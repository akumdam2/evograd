"""Run the benchmark suite across operators and emit the cross-operator report.

    evograd suite --candidates candidates/ --out results/
    evograd suite --level 1 --level 2 --baseline liger --out results/

Each operator needs a candidate program implementing its autograd pair. They are
found by name inside ``--candidates``: ``<dir>/<op>.py``, or
``<dir>/<op>/<anything>.py``. An operator with no candidate is reported as
uncovered rather than skipped — "we did not run it" and "it has no speedup" are
different claims, and only the first one is honest about a missing file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from evograd.bench.suite import SuiteReport, TaskResult, task_from_benchmark_report, write_report


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_suite_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_candidate(root: Path, op_name: str) -> Path | None:
    direct = root / f"{op_name}.py"
    if direct.is_file():
        return direct
    nested = root / op_name
    if nested.is_dir():
        candidates = sorted(nested.glob("*.py"))
        if candidates:
            return candidates[0]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="directory holding one candidate program per operator",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--level",
        type=int,
        action="append",
        dest="levels",
        choices=(1, 2, 3),
        help="restrict to these benchmark levels (repeatable; default: all)",
    )
    parser.add_argument(
        "--op",
        action="append",
        dest="ops",
        help="restrict to these operators (repeatable)",
    )
    parser.add_argument("--baseline", default="auto")
    parser.add_argument("--suite", default=None, help="named benchmark suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    args = parser.parse_args(argv)

    import torch

    from evograd.bench.fair import environment_fingerprint
    from evograd.bench.harness import DEFAULT_REPS, DEFAULT_WARMUP, run_benchmarks
    from evograd.ops import OPS

    # Fail here rather than after minutes of work. The harness times with CUDA
    # events, so without a device every operator would grind through
    # allocation and correctness at full benchmark sizes only to fail at the
    # first timed region — slowly, and once per operator.
    if not torch.cuda.is_available():
        parser.error(
            "the benchmark suite needs a CUDA device: timing uses CUDA events "
            "and the declared shapes are sized for one. Correctness alone runs "
            "anywhere via `evograd verify`."
        )

    selected = {
        name: op
        for name, op in OPS.items()
        if op.level is not None
        and (not args.levels or op.level in args.levels)
        and (not args.ops or name in args.ops)
    }
    if not selected:
        parser.error("no operator matched the selection")

    try:
        environment = environment_fingerprint()
    except Exception:  # no CUDA, or torch without a device
        environment = {"note": "environment fingerprint unavailable"}

    report = SuiteReport(environment=environment)
    for name, op in sorted(selected.items(), key=lambda i: (i[1].level, i[1].family, i[0])):
        candidate = _find_candidate(args.candidates, name)
        if candidate is None:
            report.tasks.append(
                TaskResult(
                    op=name,
                    level=op.level,
                    family=op.family,
                    baseline=args.baseline,
                    cases_total=len(op.benchmark_workloads(suite=args.suite)),
                    error=f"no candidate program found under {args.candidates}",
                )
            )
            print(f"{name:28s} no candidate", file=sys.stderr)
            continue
        try:
            module = _load_module(candidate)
            benchmark = run_benchmarks(
                op,
                module,
                warmup=args.warmup if args.warmup is not None else DEFAULT_WARMUP,
                reps=args.reps if args.reps is not None else DEFAULT_REPS,
                device=args.device,
                workloads=op.benchmark_workloads(suite=args.suite),
                performance_baseline=args.baseline,
                on_error="record",
            )
            task = task_from_benchmark_report(name, op.level, op.family, benchmark)
        except Exception as exc:
            task = TaskResult(
                op=name,
                level=op.level,
                family=op.family,
                baseline=args.baseline,
                cases_total=len(op.benchmark_workloads(suite=args.suite)),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(traceback.format_exc(limit=4), file=sys.stderr)
        report.tasks.append(task)
        marker = f"{task.speedup:.3f}x" if task.speedups else "—"
        print(
            f"{name:28s} L{op.level} {marker:>9s} "
            f"coverage {task.cases_ok}/{task.cases_total}",
            file=sys.stderr,
        )

    paths = write_report(report, args.out)
    print(json.dumps(report.to_dict()["overall"], indent=2, sort_keys=True))
    print(f"report: {paths['markdown']}")
    return 0 if all(task.ok for task in report.tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
