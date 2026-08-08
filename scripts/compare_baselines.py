"""Bench one candidate against several baselines in a single process.

    python scripts/compare_baselines.py --op layernorm --dtype float16 \
        --candidate ~/evograd_runs/layernorm/D/initial_program_autograd_pair.py

Running them in one process matters: the candidate is re-timed per baseline on
the same CUDA context, so the candidate column doubles as a noise estimate — if
it moves more than a few percent between rows, treat the speedups as soft.

Baselines default to eager PyTorch and torch.compile. Add `--baseline liger`
where a declaration provides it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--declaration", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--dtype", action="append", dest="dtypes", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--baseline",
        action="append",
        dest="baselines",
        help="repeatable; default: pytorch_autograd torch_compile",
    )
    parser.add_argument("--out", type=Path, default=None, help="write all reports here")
    args = parser.parse_args(argv)

    from evograd.bench.harness import DEFAULT_REPS, DEFAULT_WARMUP, run_benchmarks
    from evograd.ops import get_op, load_op

    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    module = _load_module(args.candidate)
    workloads = op.benchmark_workloads(
        suite=args.suite, dtypes=tuple(args.dtypes) if args.dtypes else None
    )
    baselines = args.baselines or ["pytorch_autograd", "torch_compile"]

    reports = {}
    rows = []
    for name in baselines:
        report = run_benchmarks(
            op,
            module,
            warmup=args.warmup if args.warmup is not None else DEFAULT_WARMUP,
            reps=args.reps if args.reps is not None else DEFAULT_REPS,
            device=args.device,
            workloads=workloads,
            performance_baseline=name,
            on_error="record",
        )
        reports[name] = report
        if not report["ok"]:
            error = report.get("error") or next(
                (c.get("error") for c in report["cases"] if not c.get("ok")), {}
            )
            rows.append((name, None, error))
            continue
        rows.append((name, report["aggregate"], None))

    print()
    header = (
        f"{'baseline':<28}{'candidate bwd':>14}{'baseline bwd':>14}"
        f"{'bwd speedup':>13}{'full-step':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, aggregate, error in rows:
        if aggregate is None:
            reason = f"{error.get('error_type', 'failed')}: {error.get('error_message', '')}"
            print(f"{name:<28}{reason[:60]}")
            continue
        print(
            f"{name:<28}"
            f"{aggregate['backward_from_saved_ms']:>14.4f}"
            f"{aggregate['baseline_backward_ms']:>14.4f}"
            f"{aggregate['speedup_vs_baseline_backward']:>13.2f}"
            f"{aggregate['speedup_vs_baseline_raw_full_step']:>11.2f}"
        )

    resolved = [r for _n, r, _e in rows if r is not None]
    if resolved:
        print(
            f"\nsaved/input memory ratio: {resolved[0]['saved_memory_ratio']:.3f} "
            "(candidate; identical across baselines)"
        )
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
        print(f"full reports: {args.out}")
    return 0 if all(r["ok"] for r in reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
