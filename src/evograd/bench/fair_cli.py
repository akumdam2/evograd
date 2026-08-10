"""Versioned final-report benchmark with symmetric provider timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"evograd_fair_bench_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", default="layernorm")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument(
        "--identity-control",
        action="store_true",
        help="use the exact baseline provider as both candidate and baseline",
    )
    parser.add_argument(
        "--baseline",
        choices=("auto", "liger", "pytorch_autograd"),
        default="auto",
    )
    parser.add_argument("--suite", default=None)
    parser.add_argument(
        "--dtype",
        action="append",
        dest="dtypes",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.blocks < 1:
        parser.error("--blocks must be >= 1")

    from evograd.bench.fair import (
        candidate_provider,
        liger_provider,
        pytorch_autograd_provider,
        renamed_provider,
        run_fair_benchmarks,
    )
    from evograd.opdecl.baselines import verify_performance_baseline
    from evograd.opdecl.verify import verify
    from evograd.ops import get_op

    op = get_op(args.op)
    baseline_name = args.baseline
    if baseline_name == "auto":
        baseline_name = (
            "liger" if "liger" in op.performance_baselines else "pytorch_autograd"
        )
    if baseline_name == "liger":
        baseline = liger_provider(op)
        verify_performance_baseline(op, "liger")
    else:
        baseline = pytorch_autograd_provider(op)
    if args.identity_control:
        candidate = renamed_provider(baseline, "candidate")
    else:
        module = _load_module(args.candidate)
        correctness = verify(op, module)
        if not correctness.ok:
            raise RuntimeError(
                "candidate correctness failed:\n"
                + json.dumps(correctness.to_dict(), indent=2)
            )
        candidate = candidate_provider(op, module)

    report = run_fair_benchmarks(
        op,
        candidate,
        baseline,
        workloads=op.benchmark_workloads(
            suite=args.suite,
            dtypes=tuple(args.dtypes) if args.dtypes else None,
        ),
        warmup=args.warmup,
        reps=args.reps,
        blocks=args.blocks,
        seed=args.seed,
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"full report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
