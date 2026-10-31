#!/usr/bin/env python3
"""OpenEvolve without the evograd frontend.

    python candidates/openevolve/run.py --task TASK --run-dir RUN --budget 30

The evograd agent minus its frontend: the same backend
(``evograd.evolve.run.run_evolve``), the same OpenEvolve configuration
(``render_config`` for the task's operator: models, islands, archive,
diff-based evolution, timeouts), the same evaluator and environment, the same
budget. What differs is only what the frontend provides:

* The initial program is the task's skeleton -- the pair's signatures with
  ``NotImplementedError`` bodies and the forward reference as comments --
  packaged as an artifact (so the deployment layer sits outside the evolve
  block exactly as it does in evograd's seed). It fails evaluation by
  construction; the first correct program is the LLM's.
* ``--prompt statement`` (default) sets the system message to the task's full
  problem statement: evograd's system message verbatim, then the reference,
  configuration, deliverable and rules every external agent gets.
  ``--prompt evograd`` keeps evograd's system message alone -- the strict
  ablation, in which the rules arrive only as evaluator feedback.

The shared proxy caps actual model requests, including retries; native
``iterations = budget`` remains an additional stopping limit, as for evograd.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "openevolve"


def write_config(task, args, path: Path) -> None:
    import yaml

    from evograd.evolve.run import render_config

    config = yaml.safe_load(render_config(
        task.op(), iterations=args.budget, primary_model=args.model,
        secondary_model=args.model, api_base=args.api_base))
    if args.prompt == "statement":
        config["prompt"]["system_message"] = task.statement
    config["random_seed"] = args.seed
    if os.environ.get("ADBENCH_GPU_LOCK"):
        # This watchdog includes waiting in our evaluator. The isolated GPU
        # child still has its original 850s timeout after lock acquisition;
        # bench.py enforces the shared 6000s end-to-end case limit.
        config["evaluator"]["timeout"] = max(config["evaluator"]["timeout"], 6000)
    path.write_text(yaml.safe_dump(config, sort_keys=False, width=100), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--prompt", choices=("statement", "evograd"), default="statement")
    args = p.parse_args(argv)
    task, ledger = _common.start(args, AGENT, {
        "frontend": None, "seed": "skeleton", "prompt": args.prompt,
        "iterations": args.budget,
    })

    from evograd.adbench_tools.artifact import package
    from evograd.evolve.run import run_evolve

    seed = ledger.root / "seed.py"
    seed.write_text(package(task, (task.root / "skeleton.py").read_text(encoding="utf-8"),
                            origin="skeleton"), encoding="utf-8")
    config = ledger.root / "openevolve_config.yaml"
    write_config(task, args, config)

    evolve_dir = ledger.root / "evolve"
    best = evolve_dir / "evolved_best_program.py"
    rc = 1
    try:
        rc = run_evolve(
            task.op(), seed_path=seed, output_dir=evolve_dir, scoring=task.spec["scoring"],
            iterations=args.budget, config_path=config, save_best_to=best,
            primary_model=args.model, secondary_model=args.model, api_base=args.api_base,
            performance_baseline=task.spec["performance_baseline"], save_programs=True,
            extra_env={"EVOGRAD_ALLOWED_PRIMITIVES": ",".join(task.allowed_primitives)},
        )
    except Exception:
        if not _common.budget_exhausted(args):
            raise
    finally:
        _common.import_evaluations_from_archive(ledger, evolve_dir / "programs", "openevolve")
    if _common.budget_exhausted(args):
        return _common.finish(args, task, ledger, _common.best_evaluated(ledger,
                              best if best.is_file() else None))
    if rc != 0 or not best.is_file():
        return _common.finish(args, task, ledger, None, status="evolve_failed")
    return _common.finish(args, task, ledger, best)


if __name__ == "__main__":
    raise SystemExit(main())
