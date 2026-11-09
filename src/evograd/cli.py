"""evograd — evolved backward kernels.

    evograd ops                                  # list declared operators
    evograd seed unified --pack PACK --output-dir ...  # generate a seed (pipeline unified|a|b|c|d)
    evograd evolve --op rmsnorm --seed seed.py --output-dir ...
    evograd ncu --op rmsnorm --candidate best.py --output-dir ...
    evograd evaluate qwen3_0_6b --corpus C/manifest.json --out RUN --candidate CASE=prog.py
    evograd summarize RUN_A/evaluation.json RUN_B/evaluation.json  # the benchmark protocol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PIPELINES = {
    "unified": ("evograd.pipelines.unified.cli", "Unified — compiler-aware deterministic AtenIR synthesis"),
    "a": ("evograd.pipelines.a_atenir_llm.cli", "Pipeline A — AtenIR + LLM"),
    "b": ("evograd.pipelines.b_dispatch.cli", "Pipeline B — LLM-free handwritten dispatch"),
    "c": ("evograd.pipelines.c_forward_only.cli", "Pipeline C — forward-source-only ablation"),
    "d": ("evograd.pipelines.d_inductor.cli", "Pipeline D — LLM-free Inductor capture"),
}


def _seed(argv: list[str]) -> int:
    if not argv or argv[0] not in _PIPELINES:
        names = ", ".join(f"{k} ({v[1]})" for k, v in _PIPELINES.items())
        print(f"usage: evograd seed {{unified|a|b|c|d}} [pipeline args]\npipelines: {names}")
        return 2
    import importlib

    module = importlib.import_module(_PIPELINES[argv[0]][0])
    return module.main(argv[1:])


def _evolve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="evograd evolve")
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None, help="external path.py:op declaration")
    parser.add_argument("--forward", default=None, help="override the declaration forward")
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scoring", default="speed_memory")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--save-best-to", type=Path, default=None)
    parser.add_argument(
        "--save-programs",
        action="store_true",
        help=(
            "keep every evaluated candidate under <output-dir>/programs, with an "
            "index.jsonl of their metrics (default: only the best is kept)"
        ),
    )
    parser.add_argument("--primary-model", default="gpt-5.6-sol")
    parser.add_argument("--secondary-model", default="gpt-5.6-sol")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--benchmark-suite", default=None)
    parser.add_argument(
        "--baseline",
        default="auto",
        help=(
            "auto, pytorch_autograd (eager), torch_compile, "
            "torch_compile_max_autotune, or a declaration baseline such as cublas_pair"
        ),
    )
    parser.add_argument(
        "--feature-dim",
        action="append",
        help="MAP-Elites feature dimension (repeatable); "
        "default: complexity + saved_memory_ratio",
    )
    parser.add_argument("--feature-bins", type=int, default=10)
    parser.add_argument(
        "--num-islands",
        type=int,
        default=2,
        help="MAP-Elites islands; 1 concentrates all cell competition",
    )
    parser.add_argument(
        "--archive-size",
        type=int,
        default=20,
        help="elite archive size; must not exceed the feature grid cell count",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="profile and attempt one accepted-only NCU-guided refinement of the final best",
    )
    parser.add_argument("--ncu-model", default=None)
    parser.add_argument("--ncu-timeout", type=int, default=120)
    parser.add_argument("--ncu-optimizer-timeout", type=int, default=360)
    parser.add_argument("--ncu-skip-at-roofline-pct", type=float, default=95.0)
    parser.add_argument(
        "--benchmark-dtype",
        action="append",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help=(
            "evolve a dtype specialist: gate correctness on this dtype only, and "
            "measure on it unless --benchmark-dtype says otherwise (Pipeline D seeds)"
        ),
    )
    args = parser.parse_args(argv)

    from evograd.evolve.run import run_evolve
    from dataclasses import replace
    from evograd.benchmark import get_task, load_task

    op = load_task(args.declaration) if args.declaration else get_task(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    if args.forward:
        op = replace(op, forward=args.forward)
        op.validate()
    return run_evolve(
        op,
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
        dtype=args.dtype,
        performance_baseline=args.baseline,
        save_programs=args.save_programs,
        feature_dimensions=tuple(args.feature_dim) if args.feature_dim else None,
        feature_bins=args.feature_bins,
        num_islands=args.num_islands,
        archive_size=args.archive_size,
        ncu=args.ncu,
        ncu_model=args.ncu_model,
        ncu_timeout=args.ncu_timeout,
        ncu_optimizer_timeout=args.ncu_optimizer_timeout,
        ncu_skip_at_roofline_pct=args.ncu_skip_at_roofline_pct,
    )


def _evaluate(argv: list[str]) -> int:
    """``evograd evaluate MODEL ...``: the benchmark protocol for one model's candidates."""
    from evograd.evaluation.protocol import main

    return main(argv)


def _summarize(argv: list[str]) -> int:
    from evograd.evaluation.protocol import summarize_main

    return summarize_main(argv)


def _ncu(argv: list[str]) -> int:
    from evograd.ncu.refine import main

    return main(argv)


def _ops(argv: list[str]) -> int:
    from evograd.benchmark import TASKS

    # Sort by benchmark level so the task hierarchy is what you see first;
    # unclassified declarations (scaffolded or user-supplied) sort last.
    for name, op in sorted(
        TASKS.items(), key=lambda item: (item[1].level or 99, item[1].family or "", item[0])
    ):
        level = f"L{op.level}" if op.level else "-"
        print(
            f"{name:26s} {level:3s} {op.family or '-':12s} "
            f"correctness: {len(op.correctness):2d} benchmark: {len(op.benchmark):3d}"
        )
        print(f"{'':26s}     grads: {', '.join(op.grad_names())}")
    return 0


_COMMANDS = {
    "ops": _ops,
    "seed": _seed,
    "evolve": _evolve,
    "evaluate": _evaluate,
    "summarize": _summarize,
    "ncu": _ncu,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in _COMMANDS:
        print(__doc__.strip())
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
