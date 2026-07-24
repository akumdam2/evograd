"""Standalone benchmark CLI: measure a candidate against the autograd baseline.

    python -m evograd.bench.cli --op layernorm --candidate seed.py --out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_bench_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None, help="external path.py:op declaration")
    parser.add_argument("--forward", default=None, help="override the declaration forward")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suite", default=None, help="named benchmark suite from the declaration")
    parser.add_argument(
        "--baseline",
        default="auto",
        help="auto, pytorch_autograd, or a declaration-provided baseline such as liger",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        dest="dtypes",
        choices=("float32", "float16", "bfloat16"),
        help="benchmark dtype subset (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write full JSON report here")
    args = parser.parse_args(argv)
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.reps is not None and args.reps < 1:
        parser.error("--reps must be >= 1")

    from evograd.bench.harness import DEFAULT_REPS, DEFAULT_WARMUP, run_benchmarks
    from evograd.ops import get_op, load_op

    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    if args.forward:
        op = replace(op, forward=args.forward)
        op.validate()
    report = run_benchmarks(
        op,
        _load_module(args.candidate),
        warmup=args.warmup if args.warmup is not None else DEFAULT_WARMUP,
        reps=args.reps if args.reps is not None else DEFAULT_REPS,
        device=args.device,
        workloads=op.benchmark_workloads(
            suite=args.suite, dtypes=tuple(args.dtypes) if args.dtypes else None
        ),
        performance_baseline=args.baseline,
    )
    aggregate = report["aggregate"]
    summary = {
        key: aggregate[key]
        for key in (
            "speedup_vs_baseline_backward",
            "speedup_vs_baseline_full_step",
            "geomean_speedup_vs_baseline_backward",
            "geomean_min_speedup_per_case",
            "worst_case_min_speedup",
            "saved_memory_ratio",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"full report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
