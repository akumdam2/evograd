"""What every candidate driver shares: flags, the run ledger, the final score.

Each ``candidates/<agent>/run.py`` is::

    args = parse(parser)                     # the common flags below + its own
    task, ledger = start(args, agent, config)
    best = <run the agent; return the path of its best module or None>
    finish(args, task, ledger, best)         # final.json via evograd.adbench_tools.score

The common flags are the parity knobs. ``--budget`` is the number of candidate
programs the agent may have evaluated; each driver maps it onto its agent's own
control (iterations, max_evals, workers x rounds, directions) and records the
mapping and the actual consumption, because the agents do not all count the
same unit. ``--model`` and ``--api-base`` select one OpenAI-compatible
endpoint for every agent that can use one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

DEFAULT_MODEL = os.environ.get("ADBENCH_MODEL", "gpt-5.6-sol")
DEFAULT_API_BASE = os.environ.get("ADBENCH_API_BASE", "https://api.openai.com/v1")


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=Path, required=True, help="task pack (python -m evograd.adbench_tools task)")
    p.add_argument("--run-dir", type=Path, required=True, help="fresh directory for this run")
    p.add_argument("--budget", type=int, default=30,
                   help="candidate programs the agent may have evaluated (default 30)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM for every agent call")
    p.add_argument("--api-base", default=DEFAULT_API_BASE,
                   help="OpenAI-compatible endpoint (key from $OPENAI_API_KEY)")
    p.add_argument("--seed", type=int, default=0, help="the agent's own random seed")
    p.add_argument("--no-score", action="store_true",
                   help="skip the final measurement (run `python -m evograd.adbench_tools score` later)")
    return p


def start(args, agent: str, config: dict):
    from evograd.adbench_tools.ledger import RunLedger
    from evograd.adbench_tools.task import load

    task = load(args.task)
    if (args.run_dir / "run.json").exists():
        raise SystemExit(f"{args.run_dir} already holds a run; use a fresh --run-dir")
    if args.budget < 1:
        raise SystemExit("--budget must be >= 1")
    config = {"budget": args.budget, "model": args.model, "api_base": args.api_base,
              "seed": args.seed, **config}
    ledger = RunLedger(args.run_dir, agent=agent, task=task, config=config)
    print(f"[{agent}] task {task.task_id} -> {ledger.root}", flush=True)
    return task, ledger


def finish(args, task, ledger, best: Path | None, *, status: str | None = None,
           contract: Path | None = None, **extra) -> int:
    status = status or ("completed" if best else "no_program")
    ledger.finish(status=status, best_program=best, **extra)
    if best is None:
        print(f"[{ledger.record['agent']}] produced no program", flush=True)
        return 1
    print(f"[{ledger.record['agent']}] best program: {best}", flush=True)
    if args.no_score:
        return 0
    from evograd.adbench_tools.score import score

    result = score(task.root, best, run_dir=ledger.root, agent=ledger.record["agent"],
                   contract=contract)
    print(json.dumps({k: result.get(k) for k in
                      ("agent", "accepted", "correct", "combined_score", "metrics")},
                     indent=2, default=str), flush=True)
    return 0 if result.get("accepted") and result.get("correct") else 1


def unified_seed(task, ledger, *, eval_timeout: int = 180) -> Path | None:
    """evograd's frontend on the task: ``evograd seed unified`` at the task's
    exact configuration, from the pack's own declaration. No LLM call. Returns
    the accepted seed artifact, or ``None`` (``<run>/seed/failure.json`` says why)."""
    from dataclasses import replace

    from evograd.benchmark import get_task, load_task
    from evograd.pipelines.unified.synthesize import (
        UnifiedSeedConfig,
        select_workload,
        synthesize_unified_seed,
    )

    spec = task.spec
    if spec["base_declaration"]:
        # As `evograd seed unified --declaration` does: keep the absolute reference.
        base = replace(load_task(spec["base_declaration"]), declaration=spec["base_declaration"])
    else:
        base = get_task(task.op_name)
    workload = select_workload(base, dtype=spec["workload"]["dtype"], dims=spec["workload"]["dims"])
    seed_dir = ledger.root / "seed"
    status = synthesize_unified_seed(UnifiedSeedConfig(
        op=base, workload=workload, output_dir=seed_dir, eval_timeout=eval_timeout))
    seed = seed_dir / "initial_program_autograd_pair.py"
    if status != 0 or not seed.is_file():
        return None
    ledger.log_evaluation({"kind": "seed_acceptance", "origin": "unified", "correct": True,
                           "program": str(seed)})
    return seed


def import_evaluations_from_archive(ledger, archive: Path, origin: str) -> int:
    """Copy an OpenEvolve program archive's index into the run ledger."""
    from evograd.adbench_tools.ledger import read_lines

    count = 0
    for entry in read_lines(archive / "index.jsonl"):
        metrics = entry.get("metrics") or {}
        ledger.log_evaluation({
            "kind": "full", "origin": origin,
            "correct": float(metrics.get("correct", entry.get("correct", 0.0)) or 0.0) == 1.0,
            "combined_score": metrics.get("combined_score"),
            "program": str(archive / entry["file"]) if entry.get("file") else None,
            "duplicate": bool(entry.get("duplicate")),
        })
        count += 1
    return count
