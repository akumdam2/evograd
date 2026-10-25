#!/usr/bin/env python3
"""evograd on an adbench-tools task: the unified frontend, then OpenEvolve.

    python candidates/evograd/run.py --task TASK --run-dir RUN --budget 30

1. ``evograd seed unified`` at the task's exact configuration (the same
   ``select_workload`` call the task pack made, so the seed's scoped
   declaration is the task's). No LLM call.
2. ``evograd evolve`` -- ``evograd.evolve.run.run_evolve`` -- on that seed with
   ``--budget`` iterations, the task's model/endpoint, every evaluated program
   archived, and the seed's unified contract enforced (as ``evograd evolve``
   always does for a unified seed).
3. The final measurement shared by every agent, with the unified contract.

Budget mapping: one OpenEvolve iteration = one LLM proposal = one evaluation,
so ``iterations = budget``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "evograd"


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--scoring", default=None, help="default: the task's scoring policy")
    p.add_argument("--eval-timeout", type=int, default=180,
                   help="seed acceptance timeout, as `evograd seed unified`")
    args = p.parse_args(argv)
    task, ledger = _common.start(args, AGENT, {
        "frontend": "unified", "backend": "openevolve", "iterations": args.budget,
    })

    from evograd.evolve.run import run_evolve
    from evograd.pipelines.unified.contract import CONTRACT_FILE

    spec = task.spec
    seed = _common.unified_seed(task, ledger, eval_timeout=args.eval_timeout)
    if seed is None:
        return _common.finish(args, task, ledger, None, status="seed_failed",
                              failure=str(ledger.root / "seed" / "failure.json"))

    evolve_dir = ledger.root / "evolve"
    best = evolve_dir / "evolved_best_program.py"
    try:
        rc = run_evolve(
            task.op(), seed_path=seed, output_dir=evolve_dir,
            scoring=args.scoring or spec["scoring"], iterations=args.budget,
            save_best_to=best, primary_model=args.model, secondary_model=args.model,
            api_base=args.api_base, performance_baseline=spec["performance_baseline"],
            save_programs=True,
        )
    finally:
        _common.import_evaluations_from_archive(ledger, evolve_dir / "programs", "openevolve")
    if rc != 0 or not best.is_file():
        return _common.finish(args, task, ledger, None, status="evolve_failed")
    return _common.finish(args, task, ledger, best, contract=evolve_dir / CONTRACT_FILE,
                          seed=str(seed))


if __name__ == "__main__":
    raise SystemExit(main())
