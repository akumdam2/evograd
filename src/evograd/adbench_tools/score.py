"""Final measurement of one agent's program, and the cross-agent comparison.

``score`` re-measures the program an agent returned, with nothing from the
agent's own loop trusted:

1. **Acceptance** -- ``evograd.evaluation.common.seed`` through
   ``verify_candidate(seed_acceptance=True)``, the gate the unified frontend's
   seed must pass before ``evograd`` publishes it: the deployment entry,
   every declared output/gradient against the oracle, and the timed shape.
2. **Timing** -- the full evaluator with more warmup and repetitions than the
   in-loop evaluations (defaults 25/100; both recorded).
3. **Ledger** -- the run's evaluation count, recorded LLM usage and wall time.

The result is ``<run_dir>/final.json`` (schema ``adbench-tools-result/1``).
``compare`` reads any number of them and prints a Markdown table, refusing to
mix tasks or statements.

Tier-2 and Tier-3 follow-ups take the packaged artifact this writes
(``final_program.py``); ``final.json`` carries the commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from evograd.adbench_tools import RESULT_SCHEMA
from evograd.adbench_tools.evaluate import evaluate_source, fixed_code_intact, prepare_program
from evograd.adbench_tools.ledger import RunLedger
from evograd.adbench_tools.task import load

FINAL_FILE = "final.json"
FINAL_PROGRAM = "final_program.py"
_METRICS = ("combined_score", "speedup", "full_step_speedup", "backward_speedup",
            "weighted_speedup", "forward_ms", "backward_from_saved_ms",
            "raw_forward_backward_full_step_ms", "baseline_latency_ms",
            "baseline_full_step_ms", "baseline_raw_full_step_ms", "saved_bytes", "input_bytes")


def score(task_dir: str | Path, program: str | Path, *, run_dir: str | Path,
          agent: str | None = None, warmup: int = 25, reps: int = 100,
          timeout: int = 1800, contract: str | Path | None = None) -> dict[str, Any]:
    from evograd.pipelines.shared.runner import report_passed, verify_candidate

    task = load(task_dir)
    run_dir = Path(run_dir).resolve()
    ledger = RunLedger(run_dir) if (run_dir / "run.json").is_file() else None
    agent = agent or (ledger.record["agent"] if ledger else "unknown")
    source = Path(program).read_text(encoding="utf-8")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "agent": agent,
        "task_id": task.task_id,
        "task_dir": str(task.root),
        "statement_sha256": task.spec.get("statement_sha256"),
        "source_program": str(Path(program).resolve()),
        "measurement": {"warmup": warmup, "reps": reps},
    }
    # Rebuilt from the block body: whatever the agent's loop did outside the
    # block (OpenEvolve diffs may reach it) does not reach the final measurement.
    result["fixed_code_intact"] = fixed_code_intact(task, source)
    try:
        packaged = prepare_program(task, source, run_dir / "final", origin=agent,
                                   repackage=True)
    except ValueError as exc:
        result.update({"accepted": False, "correct": False, "combined_score": None,
                       "failure": {"stage": "packaging", "error": str(exc)}})
        return _write(run_dir, result, ledger)
    final_program = run_dir / FINAL_PROGRAM
    final_program.write_text(packaged.read_text(encoding="utf-8"), encoding="utf-8")
    result["program"] = str(final_program)
    result["program_sha256"] = hashlib.sha256(final_program.read_bytes()).hexdigest()

    acceptance = verify_candidate(
        python=sys.executable, op_name=task.op_name, program_path=final_program,
        declaration=task.declaration, forward=task.op().forward,
        log_dir=run_dir / "final" / "acceptance", timeout=timeout, seed_acceptance=True,
    )
    result["accepted"] = report_passed(acceptance)
    (run_dir / "final" / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True, default=str), encoding="utf-8")

    record = evaluate_source(task, final_program.read_text(encoding="utf-8"),
                             run_dir=None, origin=f"{agent}:final", timeout=timeout,
                             warmup=warmup, reps=reps, contract=contract)
    (run_dir / "final" / "evaluation.json").write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    result["correct"] = record.correct
    result["metrics"] = {k: record.metrics[k] for k in _METRICS if k in record.metrics}
    result["combined_score"] = record.combined_score if record.correct else None
    if not record.correct:
        result["failure"] = {"stage": "evaluation", "artifacts": record.artifacts.get("failure")}
    benchmark = record.artifacts.get("benchmark") if isinstance(record.artifacts, dict) else None
    if isinstance(benchmark, dict):
        result["primitives"] = benchmark.get("primitives")
    result["follow_up"] = {
        # Tier 2 runs the operator's own declared benchmark cases, not the
        # task's one configuration: the same deeper check for every agent.
        "tier2": (f"evograd tier2-bench --op {task.op_name} --candidate {final_program} "
                  f"--baseline none --dtype {task.workload['dtype']}"),
        "tier3": (f"evograd tier3-bench --model <model> --patch-set {agent}:<site>={final_program}"
                  "  # for operators bound to a Tier-3 site"),
    }
    return _write(run_dir, result, ledger)


def _write(run_dir: Path, result: dict[str, Any], ledger: RunLedger | None) -> dict[str, Any]:
    if ledger is not None:
        result["run"] = {"config": ledger.record.get("config"), **ledger.summary()}
    path = run_dir / FINAL_FILE
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return result


def compare(paths: list[Path]) -> str:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if not results:
        raise ValueError("no results to compare")
    tasks = {r["task_id"] for r in results}
    statements = {r.get("statement_sha256") for r in results}
    if len(tasks) != 1:
        raise ValueError(f"results are for different tasks: {sorted(tasks)}")
    if len(statements) != 1:
        raise ValueError("results were produced from different problem statements; "
                         "rebuild the task pack once and rerun")

    def cell(value, fmt="{:.3f}"):
        return "-" if value is None else fmt.format(value)

    rows = ["| agent | accepted | correct | combined_score | bwd speedup | full-step speedup "
            "| saved/input | evaluations | LLM calls | wall s |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -(r.get("combined_score") or float("-inf"))):
        metrics = r.get("metrics") or {}
        run = r.get("run") or {}
        saved = (metrics["saved_bytes"] / metrics["input_bytes"]
                 if metrics.get("input_bytes") else None)
        rows.append(
            f"| {r['agent']} | {r.get('accepted')} | {r.get('correct')} "
            f"| {cell(r.get('combined_score'))} | {cell(metrics.get('speedup'))} "
            f"| {cell(metrics.get('full_step_speedup'))} | {cell(saved, '{:.2f}')} "
            f"| {run.get('evaluations', '-')} | {run.get('llm_calls_recorded', '-')} "
            f"| {run.get('wall_seconds', '-')} |")
    return f"Task `{tasks.pop()}`\n\n" + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("score", help="final measurement of one agent's program")
    one.add_argument("--task", type=Path, required=True)
    one.add_argument("--run-dir", type=Path, required=True)
    one.add_argument("--agent", default=None)
    one.add_argument("--warmup", type=int, default=25)
    one.add_argument("--reps", type=int, default=100)
    one.add_argument("--contract", type=Path, default=None,
                     help="unified_contract.json, for evograd's own unified-seed programs")
    one.add_argument("program", type=Path)
    many = sub.add_parser("compare", help="Markdown table over final.json files")
    many.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score(args.task, args.program, run_dir=args.run_dir, agent=args.agent,
                       warmup=args.warmup, reps=args.reps, contract=args.contract)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("accepted") and result.get("correct") else 1
    print(compare(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
