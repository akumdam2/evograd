#!/usr/bin/env python3
"""evograd on an adbench-tools task: a seed frontend, then OpenEvolve.

    python candidates/evograd/run.py --task TASK --run-dir RUN --budget 30 [--frontend unified]

1. The seed, at the task's exact configuration (``--frontend``):
   - ``unified`` (default): ``evograd seed unified`` -- the same ``select_workload``
     call the task pack made, so the seed's scoped declaration is the task's. Its
     library calls are fixed outside the evolve block by the unified contract.
   - ``a``, ``b``, ``d``: ``evograd seed a|b|d`` on the pack's declaration with the
     task's grant. Only A calls a model.
2. ``evograd evolve`` -- ``evograd.evolve.run.run_evolve`` -- on that seed with the
   task's model/endpoint, every evaluated program archived, and the code outside
   the seed's evolve block held fixed (a unified seed's contract, which also pins
   its library call sites; for A, B and D that code alone). For A, B and D the
   system message documents the task's grant, which is also the run's ceiling.
3. The final measurement shared by every agent, with that contract.

The shared proxy caps actual model requests at ``budget``, including retries and
A's seed synthesis. Evolution's native iteration limit is what synthesis left.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: E402

AGENT = "evograd"
FRONTENDS = ("unified", "a", "b", "d")
#: Seed verification timeout for A, B and D: real-size oracle checks, as the
#: GPU seed check ran them.
PIPELINE_EVAL_TIMEOUT = 900


def pipeline_seed(task, ledger, args) -> Path | None:
    """``evograd seed a|b|d`` on the pack's declaration with the task's grant.
    Returns the published seed, or ``None`` (``<run>/seed.log`` says why)."""
    from evograd.pipelines.shared.runner import evograd_env

    seed_dir = ledger.root / "seed"
    command = [sys.executable, "-m", "evograd.cli", "seed", args.frontend, "--op", task.op_name,
               "--declaration", f"{task.root / 'declaration.py'}:op",
               "--output-dir", str(seed_dir), "--eval-timeout", str(PIPELINE_EVAL_TIMEOUT)]
    for name in task.allowed_primitives:
        command += ["--allow-primitive", name]
    if args.frontend == "a":
        # The metered endpoint and its placeholder key; the proxy holds the real one.
        command += ["--api-base", args.api_base, "--api-key", "adbench-local", "--model", args.model]
    else:
        command += ["--dtype", task.spec["workload"]["dtype"]]
    if args.frontend == "d":
        command += ["--static-shapes", "--no-autotune"]
    with open(ledger.root / "seed.log", "w", encoding="utf-8") as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=evograd_env()).returncode
    seed = seed_dir / ("best" if args.frontend == "a" else "") / "initial_program_autograd_pair.py"
    # D publishes its seed before the static rules and the oracle run, so a file
    # alone does not mean the seed passed.
    if status != 0 or not seed.is_file():
        return None
    ledger.log_evaluation({"kind": "seed_acceptance", "origin": args.frontend, "correct": True,
                           "program": str(seed)})
    return seed


def main(argv: list[str] | None = None) -> int:
    p = _common.parser(__doc__)
    p.add_argument("--frontend", choices=FRONTENDS, default="unified")
    p.add_argument("--scoring", default=None, help="default: the task's scoring policy")
    p.add_argument("--eval-timeout", type=int, default=180,
                   help="unified seed acceptance timeout, as `evograd seed unified`")
    args = p.parse_args(argv)
    task, ledger = _common.start(args, AGENT, {
        "frontend": args.frontend, "backend": "openevolve", "iterations": args.budget,
    })

    from evograd.evolve.run import run_evolve
    from evograd.pipelines.unified.contract import CONTRACT_FILE

    spec = task.spec
    granted = task.allowed_primitives if args.frontend != "unified" else ()
    if args.frontend == "unified":
        seed = _common.unified_seed(task, ledger, eval_timeout=args.eval_timeout)
        failure = ledger.root / "seed" / "failure.json"
    else:
        seed = pipeline_seed(task, ledger, args)
        failure = ledger.root / "seed.log"
    if seed is None:
        return _common.finish(args, task, ledger, None, status="seed_failed", failure=str(failure))

    seed_requests = args._budget_proxy.calls_used
    iterations = args.budget - seed_requests
    extra = {"seed": str(seed), "seed_requests": seed_requests, "iterations": iterations}
    if iterations < 1:
        return _common.finish(args, task, ledger, seed, **extra)
    evolve_dir = ledger.root / "evolve"
    best = evolve_dir / "evolved_best_program.py"
    contract = evolve_dir / CONTRACT_FILE
    rc = 1
    try:
        rc = run_evolve(
            task.op(), seed_path=seed, output_dir=evolve_dir,
            scoring=args.scoring or spec["scoring"], iterations=iterations,
            save_best_to=best, primary_model=args.model, secondary_model=args.model,
            api_base=args.api_base, performance_baseline=spec["performance_baseline"],
            save_programs=True, allowed_primitives=granted, pin_fixed_code=True,
        )
    except Exception:
        if not _common.budget_exhausted(args):
            raise
    finally:
        _common.import_evaluations_from_archive(ledger, evolve_dir / "programs", "openevolve")
    contract = contract if contract.is_file() else None  # absent if evolve failed early
    if _common.budget_exhausted(args):
        return _common.finish(args, task, ledger, _common.best_evaluated(ledger, seed),
                              contract=contract, **extra)
    if rc != 0 or not best.is_file():
        return _common.finish(args, task, ledger, None, status="evolve_failed")
    return _common.finish(args, task, ledger, best, contract=contract, **extra)


if __name__ == "__main__":
    raise SystemExit(main())
