"""Standalone benchmark CLI: measure a candidate against the autograd baseline.

    python -m evograd.bench.cli --op layernorm --candidate seed.py --out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any


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
        help=(
            "auto, pytorch_autograd (eager), torch_compile, "
            "torch_compile_max_autotune, or a declaration-provided baseline such "
            "as liger"
        ),
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

    # Everything up to run_benchmarks can fail too — an unknown op, a candidate
    # that raises on import, a --dtype the declared benchmark suite has no cases
    # for. Those used to escape as a bare traceback on stderr, leaving no report
    # behind; capture them in the same shape run_benchmarks uses for setup errors.
    try:
        op = load_op(args.declaration) if args.declaration else get_op(args.op)
        if op.name != args.op:
            parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
        if args.forward:
            op = replace(op, forward=args.forward)
            op.validate()
        module = _load_module(args.candidate)
        workloads = op.benchmark_workloads(
            suite=args.suite, dtypes=tuple(args.dtypes) if args.dtypes else None
        )
    except Exception as exc:
        report = {
            "op": args.op,
            "ok": False,
            "aggregate": {},
            "cases": [],
            "error": {
                "phase": "setup",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        return _finish(report, args.out)

    report = run_benchmarks(
        op,
        module,
        warmup=args.warmup if args.warmup is not None else DEFAULT_WARMUP,
        reps=args.reps if args.reps is not None else DEFAULT_REPS,
        device=args.device,
        workloads=workloads,
        performance_baseline=args.baseline,
        on_error="record",
    )
    return _finish(report, args.out)


def _finish(report: dict[str, Any], out: Path | None) -> int:
    """Print a summary (or the failure), write the report, pick an exit code."""
    aggregate = report.get("aggregate") or {}
    if aggregate:
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

    for failure in _failures(report):
        print(failure, file=sys.stderr)

    if out:
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"full report: {out}")
    return 0 if report.get("ok") else 1


def _failures(report: dict[str, Any]) -> list[str]:
    """One human-readable block per failure, most useful line first."""
    blocks = []
    setup_error = report.get("error")
    if setup_error:
        blocks.append(
            f"benchmark setup FAILED ({setup_error['phase']}): "
            f"{setup_error['error_type']}: {setup_error['error_message']}\n"
            f"{setup_error['traceback']}"
        )
    for case in report.get("cases") or []:
        if case.get("ok"):
            continue
        error = case.get("error") or {}
        dims = ", ".join(f"{k}={v}" for k, v in (case.get("dims") or {}).items())
        blocks.append(
            f"case FAILED [{dims} {case.get('dtype')}]: "
            f"{error.get('error_type')}: {error.get('error_message')}\n"
            f"{error.get('traceback', '')}"
        )
    return blocks


if __name__ == "__main__":
    sys.exit(main())
