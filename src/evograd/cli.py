"""evograd — evolved backward kernels.

    evograd ops                                  # list declared operators
    evograd seed a --op rmsnorm --output-dir ... # generate a seed (pipeline a|b|c)
    evograd verify --op rmsnorm seed.py          # check a candidate vs the oracle
    evograd evolve --op rmsnorm --seed seed.py --output-dir ...
    evograd bench --op rmsnorm --candidate best.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINES = {
    "a": ("evograd.pipelines.a_atenir_llm.cli", "Pipeline A — AtenIR + LLM"),
    "b": ("evograd.pipelines.b_dispatch.cli", "Pipeline B — LLM-free handwritten dispatch"),
    "c": ("evograd.pipelines.c_forward_only.cli", "Pipeline C — forward-source-only ablation"),
}


def _seed(argv: list[str]) -> int:
    if not argv or argv[0] not in _PIPELINES:
        names = ", ".join(f"{k} ({v[1]})" for k, v in _PIPELINES.items())
        print(f"usage: evograd seed {{a|b|c}} [pipeline args]\npipelines: {names}")
        return 2
    import importlib

    module = importlib.import_module(_PIPELINES[argv[0]][0])
    return module.main(argv[1:])


def _verify(argv: list[str]) -> int:
    from evograd.opdecl.verify_cli import main

    return main(argv)


def _evolve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evograd evolve")
    parser.add_argument("--op", required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scoring", default="speed_memory")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--save-best-to", type=Path, default=None)
    parser.add_argument("--primary-model", default="gpt-4o-mini")
    parser.add_argument("--secondary-model", default="gpt-4o")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--benchmark-suite", default=None)
    parser.add_argument("--baseline", default="pytorch_autograd")
    parser.add_argument(
        "--benchmark-dtype",
        action="append",
        choices=("float32", "float16", "bfloat16"),
    )
    args = parser.parse_args(argv)

    from evograd.evolve.run import run_evolve
    from evograd.ops import get_op

    return run_evolve(
        get_op(args.op),
        seed_path=args.seed,
        output_dir=args.output_dir,
        scoring=args.scoring,
        iterations=args.iterations,
        config_path=args.config,
        save_best_to=args.save_best_to,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        api_base=args.api_base,
        benchmark_suite=args.benchmark_suite,
        benchmark_dtypes=tuple(args.benchmark_dtype) if args.benchmark_dtype else None,
        performance_baseline=args.baseline,
    )


def _bench(argv: list[str]) -> int:
    from evograd.bench.cli import main

    return main(argv)


def _ops(argv: list[str]) -> int:
    from evograd.ops import OPS

    for name, op in sorted(OPS.items()):
        grads = ", ".join(op.grad_names())
        print(f"{name:20s} grads: {grads:45s} correctness: {len(op.correctness):2d} benchmark: {len(op.benchmark):2d}")
    return 0


_COMMANDS = {
    "ops": _ops,
    "seed": _seed,
    "verify": _verify,
    "evolve": _evolve,
    "bench": _bench,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in _COMMANDS:
        print(__doc__.strip())
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
