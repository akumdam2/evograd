#!/usr/bin/env python3
"""Evaluate a model across one or all agents, including each T3 combination.

Run inside a GPU allocation with the experiment's Python environment active:
    python candidates/evaluate_model.py convnextv2_tiny --agent openevolve
Search records come from ADBench; evaluated submissions come from ADBench-redeploy.
Exit 3 means suspended: rerun the identical command in the next allocation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
MODELS = ("qwen3_0_6b", "qwen3_5_4b", "llama_3_2_1b", "qwen2_vl_2b",
          "convnextv2_tiny", "dit_s_2", "torchrec_dlrm", "olmoe_1b_7b")
CASE_COUNTS = dict(zip(MODELS, (8, 15, 8, 6, 8, 3, 2, 8)))
AGENTS = ("openevolve", "gepa", "kernelagent", "codex", "claude_code", "geak",
          "evograd", "seed")
GROUP_AGENTS = ("openevolve", "gepa", "codex")
REUSE_FLAGS = ("share_gate3_native", "share_gate2_calibration_native", "share_gate2_verification_reference")
SHARED = Path("/work/hdd/bcjw/wzhan/ADBench")
REDEPLOY = Path("/work/hdd/bcjw/wzhan/ADBench-redeploy")
VERIFICATION = Path("/work/hdd/bcjw/wzhan/evograd_verification")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_program(directory, cid, agent):
    record = read_json(directory / "submission.json")
    if record.get("case_id") != cid or record.get("agent") != agent or not record.get("submission"):
        raise ValueError(f"Invalid repackaged submission record: {directory}")
    program = directory / "submission.py"
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    if digest != record.get("program_sha256"):
        raise ValueError(f"Submission hash mismatch: {program}")
    return program, digest


def kernel_block(program):
    lines = program.read_bytes().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if b"EVOLVE-BLOCK-START" in line]
    ends = [i for i, line in enumerate(lines) if b"EVOLVE-BLOCK-END" in line]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(f"Expected one EVOLVE-BLOCK to compare kernel code: {program}")
    return b"".join(lines[starts[0] + 1:ends[0]])


def prepare(args):
    """Read frozen packs and submissions without importing candidate code."""
    registered = read_json(args.shared_root / "cases.json")[args.model]
    model_root = args.shared_root / args.agent / args.model
    tasks = [read_json(p) for p in sorted((model_root / "tasks").glob("*/task.json"))]
    if not tasks:
        raise ValueError(f"No task packs under {model_root / 'tasks'}")
    ids = [t["case_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs in task packs")
    for task in tasks:
        if task.get("model") != args.model or task.get("level") not in (1, 2):
            raise ValueError(f"Invalid task pack: {task['case_id']}")
    selection = registered.get("cases", registered.get("case_ids"))
    if selection is not None:
        expected = {item if isinstance(item, str) else item["case_id"] for item in selection}
        if set(ids) != expected:
            raise ValueError("Task packs do not match the selected cases in cases.json")
    elif len(ids) != CASE_COUNTS[args.model]:
        raise ValueError(f"Expected {CASE_COUNTS[args.model]} task packs for {args.model}; found {len(ids)}")

    corpus = args.corpus or Path(registered["corpus"])
    if not corpus.is_absolute():
        corpus = args.shared_root / corpus
    if corpus.is_dir():
        corpus = corpus / "manifest.json"
    if not corpus.is_file():
        raise ValueError(f"Corpus manifest not found: {corpus}; use --corpus PATH")

    candidates, failures = {}, {}
    for cid in sorted(ids):
        directory = model_root / "runs" / args.agent / cid.replace("/", "__")
        run = read_json(directory / "run.json")
        status = run.get("status")
        if not isinstance(status, str) or status in ("", "running", "unfinished", "pending"):
            raise ValueError(f"Search has not finished: {directory}")
        record = read_json(directory / "submission.json")
        if record.get("case_id") != cid or record.get("agent") != args.agent:
            raise ValueError(f"Submission identity mismatch: {directory}")
        if "submission" not in record:
            raise ValueError(f"Incomplete submission record: {directory}")
        if not record["submission"]:
            failures[cid] = {"status": status, "record": str(directory / "submission.json"),
                             "packaging_error": record.get("packaging_error")}
            continue
        original, original_digest = checked_program(directory, cid, args.agent)
        submission_dir = (args.submission_root / args.agent / args.model / "runs" /
                          args.agent / cid.replace("/", "__"))
        program, digest = checked_program(submission_dir, cid, args.agent)
        if program.resolve() != original.resolve() and kernel_block(program) != kernel_block(original):
            raise ValueError(f"Kernel EVOLVE-BLOCK changed during repackaging: {program}")
        candidates[cid] = {"path": str(program.resolve()), "sha256": digest}
        if program.resolve() != original.resolve():
            candidates[cid].update(original_path=str(original.resolve()), original_sha256=original_digest)
    if not candidates:
        raise ValueError("No packaged submissions; retain the search failure records")
    return {"model": args.model, "agent": args.agent, "corpus": str(corpus.resolve()),
            "selected_cases": sorted(ids), "candidates": candidates,
            "search_failures": failures}


def environment(model):
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO / "src"), env.get("PYTHONPATH"))))
    defaults = {
        "CXX": "/usr/bin/g++", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(VERIFICATION / "hf_home"),
        "EVOGRAD_BENCHMARK_DATA_ROOT": str(VERIFICATION / "data"),
        "EVOGRAD_LLAMA_HF_CACHE": "/work/hdd/bcjw/kramesh/hub",
        "EVOGRAD_HF_CACHE": ("/work/hdd/bcjw/wzhan/hf_cache"
                             if model in ("qwen3_0_6b", "qwen3_5_4b")
                             else str(VERIFICATION / "hf_cache")),
        "TRITON_CACHE_DIR": str(Path(tempfile.gettempdir()) / env.get("USER", "adbench") / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(Path(tempfile.gettempdir()) / env.get("USER", "adbench") / "inductor"),
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--agent", choices=(*AGENTS, "all"), default="openevolve",
                        help="one agent, or all = openevolve, gepa, codex")
    parser.add_argument("--no-weight-cache", action="store_true",
                        help="disable the job-local checkpoint snapshot in all-agent mode")
    for name in REUSE_FLAGS:
        parser.add_argument("--" + name.replace("_", "-"), action="store_true",
                            help="opt in to " + name.replace("_", " "))
    parser.add_argument("--shared-root", type=Path, default=SHARED,
                        help="search root containing cases.json and agent directories")
    parser.add_argument("--submission-root", type=Path, default=REDEPLOY,
                        help="repackaged submissions with agent/model/runs/agent/case layout; "
                             "default /work/hdd/bcjw/wzhan/ADBench-redeploy; no fallback to originals")
    parser.add_argument("--corpus", type=Path, help="override cases.json's corpus; directory or manifest")
    parser.add_argument("--out-root", type=Path, default=Path.home() / "adbench-evaluation-trials")
    parser.add_argument("--deadline-min", type=float, default=45,
                        help="evaluation deadline; leave headroom before allocation ends (default 45)")
    parser.add_argument("--tier3-blocks", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print command; no GPU needed")
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline_min) or args.deadline_min <= 0 or args.tier3_blocks < 1:
        parser.error("deadline must be positive and finite; tier3-blocks must be positive")
    args.shared_root = args.shared_root.expanduser().resolve()
    args.submission_root = args.submission_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.corpus:
        args.corpus = args.corpus.expanduser().resolve()
    try:
        agents = list(GROUP_AGENTS) if args.agent == "all" else [args.agent]
        if not agents:
            raise ValueError(f"No agent directories found for {args.model}")
        selections = []
        for agent in agents:
            single = argparse.Namespace(**vars(args))
            single.agent = agent
            selections.append(prepare(single))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1

    state = args.out_root / args.model / "state"
    jobs = [{"selection": selection,
             "out": str(args.out_root / args.model / "evaluation" / selection["agent"])}
            for selection in selections]
    reuse = {name: getattr(args, name) for name in REUSE_FLAGS}
    plan = {"model": args.model, "state": str(state), "jobs": jobs}
    if any(reuse.values()):
        plan["calibration_reuse"] = reuse
    plan_path = args.out_root / args.model / "all_agents.json"
    if args.agent == "all":
        plan.update(execution="case-major-shared-baselines/1", weight_cache=not args.no_weight_cache)
        command = [sys.executable, "-u", "-m", "evograd.evaluation.model_group",
                   "--plan", str(plan_path), "--deadline-min", str(args.deadline_min),
                   "--tier3-blocks", str(args.tier3_blocks)]
        if args.no_weight_cache:
            command.append("--no-weight-cache")
        print(f"Agents: {', '.join(agents)}; case-major on one GPU; shared controls and timing baselines", flush=True)
    else:
        selection = selections[0]
        command = [sys.executable, "-u", "-m", "evograd.evaluation.protocol", args.model,
                   "--corpus", selection["corpus"], "--out", jobs[0]["out"], "--state", str(state),
                   "--deadline-min", str(args.deadline_min), "--tier3-blocks", str(args.tier3_blocks)]
        command += ["--" + name.replace("_", "-") for name, on in reuse.items() if on]
        for cid, program in selection["candidates"].items():
            command += ["--candidate", f"{cid}={program['path']}"]
    for job in jobs:
        selection = job["selection"]
        print(f"{selection['agent']}/{args.model}: {len(selection['candidates'])} submissions / "
              f"{len(selection['selected_cases'])} selected cases", flush=True)
        for cid, failure in selection["search_failures"].items():
            print(f"Search produced no submission: {cid} ({failure['status']})", flush=True)
        print(f"Results: {job['out']}/evaluation.json", flush=True)
    if args.model == "olmoe_1b_7b":
        print("OLMoE has no model-scope verification gates; report this limitation with T3.", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    # Serialize access to this model's reusable controls across agents.
    state.mkdir(parents=True, exist_ok=True)
    with (state / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another evaluation is using {state}; wait for it to finish.", file=sys.stderr)
            return 1
        for job in jobs:
            existing = Path(job["out"]) / "evaluation.json"
            if existing.exists():
                recorded = read_json(existing).get("settings", {}).get("calibration_reuse", {})
                if any(bool(recorded.get(name, False)) != reuse[name] for name in REUSE_FLAGS):
                    print("Calibration reuse options changed; choose a new --out-root.", file=sys.stderr)
                    return 1
            selection_path = Path(job["out"]) / "selection.json"
            if selection_path.exists() and read_json(selection_path) != job["selection"]:
                print("Frozen selection changed; choose a new --out-root.", file=sys.stderr)
                return 1
        if args.agent == "all":
            if plan_path.exists():
                saved_plan = read_json(plan_path)
                # Reports above are authoritative for reuse settings. An older
                # launcher may have written a plan before the child rejected
                # those settings; allow repairing that descriptor without
                # discarding the existing measurements or training controls.
                identity = {k: v for k, v in plan.items() if k != "calibration_reuse"}
                saved_identity = {k: v for k, v in saved_plan.items() if k != "calibration_reuse"}
                if saved_identity != identity:
                    changed = sorted(k for k in set(identity) | set(saved_identity)
                                     if identity.get(k) != saved_identity.get(k))
                    print(f"Agent group changed ({', '.join(changed)}); choose a new --out-root.",
                          file=sys.stderr)
                    return 1
            plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        for job in jobs:
            selection_path = Path(job["out"]) / "selection.json"
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(json.dumps(job["selection"], indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(command, cwd=REPO, env=environment(args.model)).returncode
    if result == 3:
        print("Suspended at the deadline. Rerun this same command in the next allocation.")
    elif result == 0:
        print(f"Evaluation finished. Results: {args.out_root / args.model / 'evaluation'}")
        if any(selection["search_failures"] for selection in selections):
            print("Retain the search failure rows in selection.json when reporting coverage.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
