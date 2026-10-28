#!/usr/bin/env python3
"""The benchmark run of one agent on one model: tasks, runs, the protocol, the summary.

    python candidates/bench.py tasks     --model M --case-id C [C ...]
    python candidates/bench.py run       --model M --agent A [--budget N] [--case-id C ...] [-- agent flags]
    python candidates/bench.py evaluate  --model M --agent A --corpus MANIFEST [--deadline-min 50]
    python candidates/bench.py summarize --model M --agent A [A ...] [--out summary.md]

Each case of the model (``evograd evaluate``'s unit: an observed configuration
under the model's reference) gets a task pack; the agent runs its own loop on
each pack (``candidates/<agent>/run.py --no-score``) and its best program is
repackaged from the evolve block into ``submission.py``; every submission of the
agent then goes through the one protocol, ``evograd evaluate``, and
``summarize`` reports per level what the protocol counts. ``adbench_tools
score``/``compare`` (one configuration, seed acceptance, Tier 1 only) stay as
diagnostics and are not run here.

Layout under ``$ADBENCH_ROOT/<model>`` (default ``results/adbench/<model>``):

    tasks/<case>/                 task packs (case id with / as __)
    runs/<agent>/<case>/          the agent's run directory, + submission.py/.json
    runs/<agent>/<case>.log       the agent's output
    evaluation/<agent>/           evograd evaluate's DIR (evaluation.json)
    state/                        Gate 3 controls and Gate 2 calibrations, shared by every agent
    summary.{json,md}             summarize's output

Every step is resumable: a run directory that already holds a finished run is
kept, and ``evograd evaluate`` reuses its stages (``--deadline-min`` stops it
at a stage boundary with exit 3; rerun the same command).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from _common import DEFAULT_BUDGET, METERED_AGENTS  # noqa: E402

SUBMISSION = "submission.py"
SUBMISSION_RECORD = "submission.json"


def root_for(model: str, root: str | None) -> Path:
    base = Path(root or os.environ.get("ADBENCH_ROOT") or REPO / "results" / "adbench")
    return (base / model).resolve()


def case_dir(case_id: str) -> str:
    return case_id.replace("/", "__")


def model_cases(model: str) -> dict[str, object]:
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import observed_cases

    return {c.case_id: c for c in observed_cases(load(model))}


def task_dirs(root: Path, case_ids: list[str]) -> dict[str, Path]:
    return {cid: root / "tasks" / case_dir(cid) for cid in case_ids}


def selected_cases(args) -> list[str]:
    """``--case-id`` as given, else every case that has a task pack."""
    if args.case_id:
        return list(args.case_id)
    tasks = sorted((args.root / "tasks").glob("*/task.json"))
    return [json.loads(p.read_text())["case_id"] for p in tasks]


# ── tasks ────────────────────────────────────────────────────────────────────


def cmd_tasks(args) -> int:
    from evograd.adbench_tools.task import build_task, load

    cases = model_cases(args.model)
    unknown = sorted(set(args.case_id) - set(cases))
    if unknown:
        raise SystemExit(f"{args.model} has no cases {unknown}; known: {sorted(cases)}")
    for cid, directory in task_dirs(args.root, args.case_id).items():
        if (directory / "task.json").is_file():
            task = load(directory)
            if task.spec.get("case_id") != cid or task.spec.get("model") != args.model:
                raise SystemExit(f"{directory} holds {task.task_id}, not {args.model} {cid}")
            print(f"kept {task.task_id}: {directory}")
            continue
        task = build_task(None, directory, model=args.model, case_id=cid, library=args.library)
        print(f"built {task.task_id}: {directory} (L{task.spec['level']}, "
              f"primitives {list(task.allowed_primitives)})")
    return 0


# ── run ──────────────────────────────────────────────────────────────────────


def submit(task, run_dir: Path, agent: str) -> dict:
    """``submission.py``: the agent's best program rebuilt from its evolve block.

    The first step of ``adbench_tools score``, on its own: nothing the agent's
    loop did outside the block reaches the protocol, and ``fixed_code_intact``
    records whether it tried.
    """
    from evograd.adbench_tools.evaluate import fixed_code_intact, prepare_program

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record = {"agent": agent, "task_id": task.task_id, "case_id": task.spec["case_id"],
              "level": task.spec["level"], "status": run.get("status"),
              "best_program": run.get("best_program"), "wall_seconds": run.get("wall_seconds")}
    best = run.get("best_program")
    if not best or not Path(best).is_file():
        record["submission"] = None
    else:
        source = Path(best).read_text(encoding="utf-8")
        record["fixed_code_intact"] = fixed_code_intact(task, source)
        try:
            packaged = prepare_program(task, source, run_dir / "final", origin=agent,
                                       repackage=True)
        except ValueError as exc:  # PackagingError: the module is not a pair
            record.update({"submission": None, "packaging_error": str(exc)})
        else:
            text = packaged.read_text(encoding="utf-8")
            (run_dir / SUBMISSION).write_text(text, encoding="utf-8")
            record["submission"] = str(run_dir / SUBMISSION)
            record["program_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (run_dir / SUBMISSION_RECORD).write_text(json.dumps(record, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    return record


def cmd_run(args) -> int:
    from evograd.adbench_tools.task import load

    driver = HERE / args.agent / "run.py"
    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")
    if any(flag == "--budget" or flag.startswith("--budget=") for flag in args.agent_args):
        raise SystemExit("set --budget on bench.py, not in the agent flags")
    if not driver.is_file():
        raise SystemExit(f"no candidate driver {driver}")
    failures = 0
    for cid, directory in task_dirs(args.root, selected_cases(args)).items():
        if not (directory / "task.json").is_file():
            raise SystemExit(f"no task pack for {cid} at {directory}; run `tasks` first")
        task = load(directory)
        run_dir = args.root / "runs" / args.agent / case_dir(cid)
        run_json = run_dir / "run.json"
        done = run_json.is_file() and json.loads(run_json.read_text()).get("status") != "running"
        if done and not args.redo:
            config = json.loads(run_json.read_text()).get("config", {})
            if args.agent in METERED_AGENTS and (
                    config.get("budget_unit") != "api_requests" or
                    config.get("budget") != args.budget):
                raise SystemExit(f"{run_dir} was not run with the requested {args.budget}-request "
                                 "budget; use a fresh --root or explicitly --redo")
            print(f"[{args.agent}] {cid}: kept finished run at {run_dir}", flush=True)
        else:
            if run_dir.exists():
                if not args.redo:
                    raise SystemExit(f"{run_dir} contains an unfinished run; refusing to erase its "
                                     "API budget ledger. Recover that run or use a fresh --root; "
                                     "--redo explicitly starts a new trial.")
                import shutil
                shutil.rmtree(run_dir)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(driver), "--task", str(directory), "--run-dir",
                       str(run_dir), "--budget", str(args.budget), "--no-score", *args.agent_args]
            print(f"[{args.agent}] {cid}: {' '.join(command)}", flush=True)
            with open(run_dir.with_suffix(".log"), "w", encoding="utf-8") as log:
                env = {**os.environ, "PYTHONPATH": os.pathsep.join(
                    [str(REPO / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                           timeout=args.time_limit or None)
            if completed.returncode != 0:
                print(f"[{args.agent}] {cid}: driver exited {completed.returncode} "
                      f"(see {run_dir.with_suffix('.log')})", flush=True)
        if not run_json.is_file():
            failures += 1
            continue
        record = submit(task, run_dir, args.agent)
        print(f"[{args.agent}] {cid}: submission {record.get('submission') or 'none'} "
              f"({record.get('packaging_error') or record['status']})", flush=True)
        failures += record.get("submission") is None
    return 1 if failures else 0


# ── evaluate ─────────────────────────────────────────────────────────────────


def submissions(root: Path, agent: str) -> dict[str, Path]:
    found = {}
    for record_path in sorted((root / "runs" / agent).glob(f"*/{SUBMISSION_RECORD}")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("submission"):
            found[record["case_id"]] = Path(record["submission"])
    return found


def cmd_evaluate(args) -> int:
    chosen = submissions(args.root, args.agent)
    if args.case_id:
        chosen = {cid: path for cid, path in chosen.items() if cid in args.case_id}
    if not chosen:
        raise SystemExit(f"no submissions of {args.agent} under {args.root / 'runs' / args.agent}")
    out = args.root / "evaluation" / args.agent
    command = [sys.executable, "-m", "evograd.evaluation.protocol", args.model,
               "--corpus", str(args.corpus), "--out", str(out), "--state", str(args.root / "state"),
               "--tier3-blocks", str(args.tier3_blocks)]
    if args.deadline_min is not None:
        command += ["--deadline-min", str(args.deadline_min)]
    for cid, path in sorted(chosen.items()):
        command += ["--candidate", f"{cid}={path}"]
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "PYTHONPATH": os.pathsep.join([str(REPO / "src"),
                                          os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    print(" ".join(command), flush=True)
    return subprocess.run(command, env=env).returncode


# ── summarize ────────────────────────────────────────────────────────────────


def cost(root: Path, agent: str) -> dict:
    """What the agent spent over its runs, from the ledgers."""
    from evograd.adbench_tools.ledger import RunLedger

    total = {"runs": 0, "evaluations": 0, "llm_calls_recorded": 0,
             "unmetered_runs": 0, "wall_seconds": 0.0}
    for run_json in sorted((root / "runs" / agent).glob("*/run.json")):
        summary = RunLedger(run_json.parent).summary()
        total["runs"] += 1
        total["evaluations"] += summary["evaluations"]
        if summary["llm_calls_recorded"] is None:
            total["unmetered_runs"] += 1
        else:
            total["llm_calls_recorded"] += summary["llm_calls_recorded"]
        total["wall_seconds"] += float(summary.get("wall_seconds") or 0.0)
    return total


def _cell(value, fmt="{:.2f}"):
    return "-" if value is None else fmt.format(value)


def summary_table(rows: dict[str, dict]) -> str:
    lines = ["| agent | level / tier | submitted | violation | executable | gate 1 | accepted | coverage "
             "| speedup vs eager | vs compile | evaluations | API requests | wall h |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for agent, entry in rows.items():
        summary, spent = entry["summary"], entry["cost"]
        calls = "unmetered" if spent.get("unmetered_runs") else str(spent["llm_calls_recorded"])
        for level in ("level1", "level2"):
            s = summary[level]
            f = s["funnel"]
            if not f["configurations"]:
                continue
            lines.append(
                f"| {agent} | L{level[-1]} / T2 | {f['submitted']}/{f['configurations']} | {f['violation']} "
                f"| {f['executable']} | {f['gate1']} | {f['accepted']} | {_cell(s['coverage'])} "
                f"| {_cell(s['speedup']['eager']['geomean'])} "
                f"| {_cell(s['speedup']['torch_compile']['geomean'])} "
                f"| {spent['evaluations']} | {calls} "
                f"| {spent['wall_seconds'] / 3600:.2f} |")
        for model, m in summary["level3"]["models"].items():
            lines.append(
                f"| {agent} | L3 / T3 ({model}) | {len(m['cases'])} combined | - | - | - "
                f"| {'yes' if m['accepted'] else 'no'} | - | {_cell(m['speedup'])} "
                f"| {_cell(m['speedup_vs_torch_compile'])} | - | - | - |")
    singles = []
    for agent, entry in rows.items():
        for model, cases in entry["summary"]["level2"].get("tier3_single_site", {}).items():
            for cid, case in cases.items():
                timed = case.get("tier3") or {}
                interval = timed.get("ci95") or {}
                ci = f"[{_cell(interval.get('low'))}, {_cell(interval.get('high'))}]"
                singles.append(
                    f"| {agent} | {model}: {cid} | {'yes' if case.get('accepted') else 'no'} "
                    f"| {_cell(timed.get('step_ms'))} | {_cell(timed.get('speedup'))} | {ci} "
                    f"| {'yes' if case.get('positive') else 'no'} |")
    if singles:
        lines += ["", "L2 single-site Tier 3 (against eager)", "",
                  "| agent | case | accepted | step ms | speedup | 95% interval | positive |",
                  "|---|---|---|---|---|---|---|", *singles]
    return "\n".join(lines) + "\n"


def cmd_summarize(args) -> int:
    from evograd.evaluation.protocol import summarize

    rows = {}
    for agent in args.agent:
        path = args.root / "evaluation" / agent / "evaluation.json"
        if not path.is_file():
            print(f"{agent}: no {path}; run `evaluate` first", file=sys.stderr)
            continue
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        try:
            summary = summarize([evaluation])
        except ValueError as exc:
            raise SystemExit(f"{agent}: {exc}") from exc
        rows[agent] = {"summary": summary, "cost": cost(args.root, agent),
                       "suspended": evaluation.get("suspended")}
    if not rows:
        return 1
    table = summary_table(rows)
    out = args.out or args.root / "summary.md"
    out.write_text(f"Model `{args.model}`\n\n{table}", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print(table)
    for agent, entry in rows.items():
        if entry["suspended"]:
            print(f"note: {agent}'s evaluation is suspended ({entry['suspended']}); rerun `evaluate`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="a verified model (evograd evaluate's)")
        p.add_argument("--root", default=None, help="default $ADBENCH_ROOT/<model>")

    p = sub.add_parser("tasks", help="task packs for the model's observed cases")
    common(p)
    p.add_argument("--case-id", nargs="+", required=True, metavar="TASK/COMPONENT")
    p.add_argument("--library", default="auto")
    p.set_defaults(run=cmd_tasks)

    p = sub.add_parser("run", help="one agent on every task pack (or --case-id), no scoring")
    common(p)
    p.add_argument("--agent", required=True, help="a directory under candidates/")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="model API requests per agent/case, Claude Code's included "
                        "(default $ADBENCH_BUDGET or 10; GEAK: only counted, its "
                        "optimization directions are max(2, N // 5))")
    p.add_argument("--case-id", nargs="*", default=None)
    p.add_argument("--time-limit", type=int, default=0, help="seconds per case, 0 = none")
    p.add_argument("--redo", action="store_true",
                   help="start a new trial, replacing an existing run and its API ledger")
    p.add_argument("agent_args", nargs=argparse.REMAINDER,
                   help="flags for the agent's run.py, after --")
    p.set_defaults(run=cmd_run)

    p = sub.add_parser("evaluate", help="evograd evaluate on the agent's submissions")
    common(p)
    p.add_argument("--agent", required=True)
    p.add_argument("--corpus", type=Path, required=True, help="real-payload corpus manifest")
    p.add_argument("--case-id", nargs="*", default=None)
    p.add_argument("--deadline-min", type=float, default=None)
    p.add_argument("--tier3-blocks", type=int, default=10)
    p.set_defaults(run=cmd_evaluate)

    p = sub.add_parser("summarize", help="per-level table over agents")
    common(p)
    p.add_argument("--agent", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(run=cmd_summarize)

    args = parser.parse_args(argv)
    args.root = root_for(args.model, args.root)
    if getattr(args, "agent_args", None) and args.agent_args[0] == "--":
        args.agent_args = args.agent_args[1:]
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
