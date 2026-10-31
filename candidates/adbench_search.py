#!/usr/bin/env python3
"""Read-only progress/preflight and Slurm searches for the six-agent plan.

Searches use Gate 1 feedback. Formal Gate 2/3 evaluation is run by Wangjia.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

SHARED = Path("/work/hdd/bcjw/wzhan/ADBench")
CASE_COUNTS = {"qwen3_0_6b": 8, "qwen3_5_4b": 15, "llama_3_2_1b": 8,
               "qwen2_vl_2b": 6, "convnextv2_tiny": 8, "dit_s_2": 3,
               "torchrec_dlrm": 2, "olmoe_1b_7b": 8}
MODELS = tuple(CASE_COUNTS)
AGENTS = ("openevolve", "gepa", "geak", "codex")
SMOKE_CASES = ("rmsnorm/rmsnorm", "qwen3_swiglu_mlp/swiglu_mlp")
LLM_MODEL = "gpt-5.5"
OUTER_LIMIT = 6000
SAFETY_SECONDS = 180


def read_json(path: Path):
    return json.loads(path.read_text())


def cases(agent: str, only_model: str | None):
    registered = read_json(SHARED / "cases.json")
    for model in ((only_model,) if only_model else MODELS):
        if model not in registered:
            raise ValueError(f"{model} missing from cases.json")
        task_root = SHARED / agent / model / "tasks"
        tasks = [read_json(p) for p in sorted(task_root.glob("*/task.json"))]
        ids = [t["case_id"] for t in tasks]
        if len(ids) != CASE_COUNTS[model] or len(set(ids)) != len(ids):
            raise ValueError(f"Expected {CASE_COUNTS[model]} unique shared packs for {model}; "
                             f"found {len(ids)} under {task_root}. Do not rebuild packs.")
        # Use an explicit selection in the shared index when provided. Otherwise
        # the prebuilt packs (validated against the plan's counts) define cases.
        selection = registered[model].get("cases", registered[model].get("case_ids"))
        if selection is not None:
            expected = {x if isinstance(x, str) else x["case_id"] for x in selection}
            if set(ids) != expected:
                raise ValueError(f"Task packs do not match cases.json for {model}")
        for task in tasks:
            if task.get("model") != model or task.get("level") not in (1, 2):
                raise ValueError(f"Invalid model/level in task pack: {model} {task['case_id']}")
            yield model, task["case_id"]


def run_dir(agent: str, model: str, case_id: str) -> Path:
    return SHARED / agent / model / "runs" / agent / case_id.replace("/", "__")


def status(path: Path) -> str:
    if not path.exists():
        # A log alone can be left by a failure before the ledger was created.
        return "unfinished" if path.with_suffix(".log").exists() else "pending"
    try:
        state = read_json(path / "run.json").get("status")
        if not isinstance(state, str) or state in ("", "running", "unfinished"):
            return "unfinished"
        submission = read_json(path / "submission.json")
        if "submission" not in submission:
            return "unfinished"
        if submission["submission"] and not (path / "submission.py").is_file():
            return "unfinished"
        return state  # A terminal no-program/packaging failure is never retried.
    except (OSError, ValueError, AttributeError, TypeError):
        return "unfinished"


def packaged(path: Path) -> bool:
    try:
        return bool(read_json(path / "submission.json").get("submission")) and (path / "submission.py").is_file()
    except (OSError, ValueError, AttributeError):
        return False


def ledger_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def smoke_longest(agent: str) -> float:
    root = SHARED / "smoke" / agent / "qwen3_0_6b"
    elapsed = []
    for case_id in SMOKE_CASES:
        path = root / "runs" / agent / case_id.replace("/", "__")
        if status(path) in ("pending", "unfinished") or not packaged(path):
            raise ValueError(f"Smoke search/packaging incomplete: {path}")
        data = read_json(path / "run.json")
        config = data.get("config", {})
        check_model(agent, path, config)
        calls = ledger_count(path / "api_calls.jsonl")
        budget = 2 if agent == "geak" else 10
        unit = "optimization_directions" if agent == "geak" else "api_requests"
        if config.get("budget") != budget or config.get("budget_unit") != unit:
            raise ValueError(f"Smoke budget mismatch: {path}")
        if calls < 1 or (agent != "geak" and calls > 10):
            raise ValueError(f"Smoke request count invalid ({calls}): {path}")
        wall = float(data.get("wall_seconds") or 0)
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError(f"Smoke wall time missing: {path}")
        elapsed.append(wall)
    evaluation = root / "evaluation" / agent / "evaluation.json"
    result = read_json(evaluation)
    if result.get("suspended") or not result.get("summary"):
        raise ValueError(f"Smoke evaluation not finished; rerun evaluate unchanged: {evaluation}")
    if any(not result.get("cases", {}).get(cid, {}).get("submitted") for cid in SMOKE_CASES):
        raise ValueError(f"Smoke evaluation does not include both submissions: {evaluation}")
    return max(elapsed)


def check_model(agent: str, path: Path, config: dict) -> None:
    if config.get("model") != LLM_MODEL or (agent == "geak" and
            (config.get("backend") != "codex" or config.get("api") != "openai")):
        raise ValueError(f"Model/backend mismatch: {path}; expected {LLM_MODEL}"
                         + (" through Codex/OpenAI" if agent == "geak" else "")
                         + ". Preserve this run separately; do not mix it into this plan.")


def agent_flags(agent: str) -> list[str]:
    if agent == "gepa":
        return ["--model", LLM_MODEL, "--eval-timeout", "900"]
    if agent == "codex":
        return ["--model", LLM_MODEL, "--effort", "high",
                "--time-limit", "4800", "--eval-timeout", "900"]
    if agent == "geak":
        return ["--model", LLM_MODEL, "--backend", "codex", "--effort", "high",
                "--geak-budget", "2", "--time-limit", "4800"]
    return ["--model", LLM_MODEL]


def job_end_time() -> float:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise ValueError("Searches must run inside sbatch; use the login-node launcher.")
    info = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(part.split("=", 1) for part in info.split() if "=" in part)
    return datetime.fromisoformat(fields["EndTime"]).timestamp()


@contextmanager
def agent_lock(agent: str):
    # Covers direct submissions and overlapping all-model/model-specific jobs.
    with (SHARED / agent / ".batch-search.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another batch is already searching for {agent}") from None
        yield


def kill_group(proc: subprocess.Popen) -> None:
    # The bench parent can exit before its driver/evaluator descendants do.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    until = time.monotonic() + 5
    while time.monotonic() < until:
        proc.poll()  # Reap the group leader if it has exited.
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def states(agent, selected):
    counts = Counter()
    for model, cid in selected:
        path = run_dir(agent, model, cid)
        state = status(path)
        if state not in ("pending", "unfinished"):
            check_model(agent, path, read_json(path / "run.json").get("config", {}))
        counts[state] += 1
    return counts


def validate_workers(agent: str, workers: int) -> None:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if workers > 1 and agent not in ("gepa", "openevolve"):
        raise ValueError("Shared-GPU workers are supported only for GEPA and OpenEvolve")


def shared_gpu_lock() -> str:
    configured = os.environ.get("ADBENCH_GPU_LOCK")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if "," in gpu:
            raise ValueError("--workers shares one GPU; expose exactly one CUDA device")
        directory = SHARED / "gpu-locks"
        path = directory / f"adbench-{socket.gethostname()}-{os.environ.get('SLURM_JOB_ID', 'manual')}-{gpu}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def launch_search(agent, model, case_id, remaining, gpu_lock=None):
    env = os.environ.copy()
    env.update(ADBENCH_ROOT=str(SHARED / agent), ADBENCH_MODEL=LLM_MODEL, ADBENCH_BUDGET="10")
    env["EVOGRAD_HF_CACHE"] = (str(SHARED.parent / "hf_cache") if
                               model in ("qwen3_0_6b", "qwen3_5_4b") else
                               "/work/hdd/bcjw/wzhan/evograd_verification/hf_cache")
    if gpu_lock:
        env["ADBENCH_GPU_LOCK"] = gpu_lock
    command = [sys.executable, "candidates/bench.py", "run", "--model", model,
               "--agent", agent, "--case-id", case_id, "--budget", "10",
               "--time-limit", str(OUTER_LIMIT), "--", *agent_flags(agent)]
    print(f"START {agent} {model} {case_id}; {remaining/60:.1f} min left", flush=True)
    return subprocess.Popen(command, env=env, start_new_session=True)


def run(agent: str, only_model: str | None, skip_smoke: bool = False, workers: int = 1) -> int:
    validate_workers(agent, workers)
    key = "OPENAI_API_KEY"
    if not os.environ.get(key, "").strip():
        raise ValueError(f"{key} is missing from the exported environment; load ~/.adbench_keys.txt before searching")
    deadline = job_end_time()
    with agent_lock(agent):
        if skip_smoke:
            with (SHARED / agent / "NOTES.md").open("a") as notes:
                notes.write(f"\n- {datetime.now().astimezone().isoformat(timespec='seconds')}: "
                            f"Slurm job {os.environ['SLURM_JOB_ID']} searched "
                            f"{only_model or 'all models'} with --skip-smoke. "
                            "Smoke verification was bypassed; the next-case guard is "
                            "20 minutes plus a 3-minute cleanup margin.\n")
        return run_locked(agent, only_model, deadline, skip_smoke, workers)


def run_locked(agent: str, only_model: str | None, deadline: float,
               skip_smoke: bool = False, workers: int = 1) -> int:
    validate_workers(agent, workers)
    selected = list(cases(agent, only_model))
    counts = states(agent, selected)
    if counts["unfinished"]:
        for model, case_id in selected:
            path = run_dir(agent, model, case_id)
            if status(path) == "unfinished":
                print(f"ERROR: unfinished {path}; inspect the run and log before continuing.", flush=True)
        return 2
    if not counts["pending"]:
        print("No pending searches. DONE.", flush=True)
        return 0
    if skip_smoke:
        cutoff = 20 * 60
        print(f"{agent}: {len(selected)} cases; smoke check skipped; "
              f"stop starting cases below {(cutoff+SAFETY_SECONDS)/60:.1f} min remaining "
              "(20-minute minimum plus cleanup margin)", flush=True)
    else:
        longest = smoke_longest(agent)
        cutoff = max(2 * longest, 20 * 60)
        print(f"{agent}: {len(selected)} cases; longest smoke {longest/60:.1f} min; "
              f"stop starting cases below {(cutoff+SAFETY_SECONDS)/60:.1f} min remaining", flush=True)
    if workers > 1:
        lock = shared_gpu_lock()
        print(f"Sharing one GPU across {workers} case searches; lock: {lock}", flush=True)
        with (SHARED / agent / "NOTES.md").open("a") as notes:
            notes.write(f"\n- {datetime.now().astimezone().isoformat(timespec='seconds')}: "
                        f"job {os.environ.get('SLURM_JOB_ID', 'unknown')} uses --workers {workers} "
                        f"for {only_model or 'all models'}, serialized GPU evaluations via {lock}. "
                        "Search wall times include lock waiting; per-case budgets and limits are unchanged.\n")
        return run_parallel(agent, selected, deadline, cutoff, workers, lock)
    started = 0
    failed = False
    for model, case_id in selected:
        path = run_dir(agent, model, case_id)
        state = status(path)
        if state == "unfinished":
            print(f"ERROR: unfinished {path}; stopping before another case.", flush=True)
            failed = True
            break
        if state != "pending":
            continue
        remaining = deadline - time.time()
        if remaining < cutoff + SAFETY_SECONDS:
            print(f"Time cutoff reached ({remaining/60:.1f} min left).", flush=True)
            break
        proc = launch_search(agent, model, case_id, remaining)
        started += 1
        try:
            rc = proc.wait(timeout=min(OUTER_LIMIT + 60, remaining - SAFETY_SECONDS))
        except (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt):
            print(f"INTERRUPTED {model} {case_id}; preserve its directory and .log. "
                  "Inspect before applying the plan's aborted-run procedure.", flush=True)
            failed = True
            break
        finally:
            kill_group(proc)
        current = status(path)
        print(f"END {model} {case_id}: exit {rc}, status {current}, packaged {packaged(path)}", flush=True)
        if current in ("pending", "unfinished") or rc != 0 or not packaged(path):
            print("Stopping after a failed or incomplete search; inspect the run and log.", flush=True)
            failed = True
            break
    counts = states(agent, selected)
    print(f"Started {started} case(s); {counts['pending']} pending and "
          f"{counts['unfinished']} unfinished remain for {only_model or 'all models'}.", flush=True)
    if not counts["pending"] and not counts["unfinished"]:
        print("DONE: every selected case has a finished search.", flush=True)
    return 2 if failed else 0


def run_parallel(agent, selected, deadline, cutoff, workers, lock):
    pending = [(m, cid) for m, cid in selected if status(run_dir(agent, m, cid)) == "pending"]
    active = {}
    started = 0
    failed = False
    stop_starting = False
    try:
        while active or (pending and not stop_starting):
            for proc, (model, cid, began) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    if time.monotonic() - began > OUTER_LIMIT + 60:
                        raise InterruptedError(f"Per-case timeout: {model}:{cid}")
                    continue
                kill_group(proc)
                del active[proc]
                path = run_dir(agent, model, cid)
                current = status(path)
                print(f"END {model} {cid}: exit {rc}, status {current}, packaged {packaged(path)}", flush=True)
                if rc != 0 or current in ("pending", "unfinished") or not packaged(path):
                    print("Search failed; starting no new cases. Allowing other active searches to finish.", flush=True)
                    failed = stop_starting = True
            remaining = deadline - time.time()
            if active and remaining <= SAFETY_SECONDS:
                raise InterruptedError("Allocation cleanup margin reached")
            if not stop_starting and remaining < cutoff + SAFETY_SECONDS:
                print(f"Time cutoff reached ({remaining/60:.1f} min left); draining active searches.", flush=True)
                stop_starting = True
            while pending and not stop_starting and len(active) < workers:
                model, cid = pending.pop(0)
                if status(run_dir(agent, model, cid)) != "pending":
                    raise ValueError(f"Case changed before launch: {model}:{cid}")
                proc = launch_search(agent, model, cid, deadline - time.time(), lock)
                active[proc] = (model, cid, time.monotonic())
                started += 1
            if active:
                time.sleep(0.2)
    except (InterruptedError, KeyboardInterrupt) as exc:
        print(f"INTERRUPTED: {exc}; preserving active run directories and ledgers.", flush=True)
        failed = True
    finally:
        for proc, (model, cid, _) in active.items():
            print(f"Stopping active search {model}:{cid}; inspect its run and log.", flush=True)
            kill_group(proc)
    counts = states(agent, selected)
    print(f"Started {started} case(s); {counts['pending']} pending and "
          f"{counts['unfinished']} unfinished remain.", flush=True)
    if not counts['pending'] and not counts['unfinished']:
        print("DONE: every selected case has a finished search.", flush=True)
    return 2 if failed else 0


def report(agent: str | None, only_model: str | None) -> int:
    for name in ((agent,) if agent else AGENTS):
        selected = list(cases(name, only_model))
        counts = states(name, selected)
        paths = [run_dir(name, model, cid) for model, cid in selected]
        print(f"{name}: {len(selected)} planned, {sum(map(packaged, paths))} kernels packaged, "
              f"{sum(ledger_count(p/'api_calls.jsonl') for p in paths)} API requests, "
              f"{sum(ledger_count(p/'evaluations.jsonl') for p in paths)} search evaluations; "
              + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        for (model, cid), path in zip(selected, paths):
            if status(path) == "unfinished":
                print(f"  Active or interrupted: {model}:{cid}; check squeue and logs.")
    print("Kernel packaging is not a formal Gate 2/3 verdict. Costs above exclude archived attempts.")
    return 0


def plan(agent: str, model: str | None, skip_smoke: bool = False, workers: int = 1) -> int:
    validate_workers(agent, workers)
    selected = list(cases(agent, model))
    counts = states(agent, selected)
    terminal = len(selected) - counts["pending"] - counts["unfinished"]
    print(f"{agent} / {model or 'all models'}: {len(selected)} planned; {terminal} finished; "
          f"{counts['pending']} pending; {counts['unfinished']} active/interrupted", flush=True)
    for m, cid in selected:
        if status(run_dir(agent, m, cid)) == "unfinished":
            print(f"  Inspect active/interrupted: {m}:{cid}", flush=True)
    if counts["unfinished"]:
        print("Inspect unfinished runs before requesting another GPU job.", flush=True)
        return 4
    if not counts["pending"]:
        print("DONE: every selected case has a finished search.", flush=True)
        return 3
    m, cid = next((m, cid) for m, cid in selected if status(run_dir(agent, m, cid)) == "pending")
    if skip_smoke:
        print(f"Next pending: {m}:{cid}; smoke check skipped; "
              f"next-case guard {20*60+SAFETY_SECONDS} sec", flush=True)
    else:
        longest = smoke_longest(agent)
        print(f"Next pending: {m}:{cid}; longest smoke {longest/60:.1f} min", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "run", "status", "plan", "check-smoke"))
    parser.add_argument("--agent", choices=AGENTS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="bypass smoke preflight; use a 20-minute next-case guard plus cleanup")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent cases sharing one GPU (GEPA/OpenEvolve only; default 1)")
    parser.add_argument("--partition", choices=("ghx4", "ghx4-interactive"),
                        help="submit only: select the sbatch file (default: ghx4)")
    args = parser.parse_args()
    if args.workers != 1 and args.action not in ("submit", "run", "plan"):
        parser.error("--workers applies only to submit, run and plan")
    if args.partition is not None and args.action != "submit":
        parser.error("--partition applies only to submit; run uses its existing Slurm allocation")
    if args.skip_smoke and args.action not in ("submit", "run", "plan"):
        parser.error("--skip-smoke applies only to submit, run and plan")
    if args.action != "status" and not args.agent:
        parser.error(f"{args.action} requires --agent")
    try:
        if args.action == "submit":
            launcher = Path(__file__).resolve().with_name("submit_adbench_search.sh")
            command = ["bash", str(launcher), args.agent]
            if args.model:
                command.append(args.model)
            command.extend(["--partition", args.partition or "ghx4"])
            if args.skip_smoke:
                command.append("--skip-smoke")
            command.extend(["--workers", str(args.workers)])
            return subprocess.call(command)
        if args.action == "run":
            def interrupted(signum, frame):
                raise InterruptedError(f"Received signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            return run(args.agent, args.model, args.skip_smoke, args.workers)
        if args.action == "plan":
            return plan(args.agent, args.model, args.skip_smoke, args.workers)
        if args.action == "check-smoke":
            print(f"Smoke records complete; longest search: {smoke_longest(args.agent)/60:.1f} min")
            print("Also confirm actual model routing in agent logs; API ledgers count requests, not model names.")
            return 0
        return report(args.agent, args.model)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
