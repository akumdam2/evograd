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

from evograd.bench.suite import (
    SuiteReport,
    TaskResult,
    task_from_benchmark_report,
    task_from_fair_report,
    task_from_tier3_report,
    write_report,
)


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--candidates",
        type=Path,
        help="directory holding one candidate program per operator",
    )
    source.add_argument(
        "--candidate-baseline",
        help=(
            "run a reviewed pair baseline (liger, cublas_pair, ...) as the "
            "candidate on every operator that declares it. Produces the suite's "
            "reference line without needing a generated program per operator. "
            "Operators without that baseline are reported as uncovered."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--level",
        type=int,
        action="append",
        dest="levels",
        choices=(1, 2, 3, 4),
        help="restrict to these benchmark levels (repeatable; default: all)",
    )
    parser.add_argument(
        "--tier3-report",
        action="append",
        default=[],
        metavar="WORKLOAD=PATH",
        help=(
            "fold a finished tier-3 training-step report into the suite as "
            "that level-4 workload's row (e.g. alphafold3=results/t3.json). "
            "Level-4 tasks are measured by `evograd tier3-bench`, not by this "
            "command; without a report here they appear as uncovered."
        ),
    )
    parser.add_argument(
        "--op",
        action="append",
        dest="ops",
        help="restrict to these operators (repeatable)",
    )
    # Deliberately not defaulted to "auto" here: with --candidate-baseline liger,
    # "auto" would also resolve to liger and every operator would report a
    # perfect 1.000x against itself, which reads as a finished run rather than a
    # mistake. Resolved below once the candidate source is known.
    parser.add_argument("--baseline", default=None)
    parser.add_argument(
        "--protocol",
        choices=("fair", "fast"),
        default="fair",
        help=(
            "fair (default): the final-report protocol — L2 cleared before every "
            "timed region, batched CUDA events, randomized provider order, "
            "mutation checks. fast: the low-overhead harness the evolutionary "
            "search uses, for iterating only. Published numbers must come from "
            "fair; the fast harness measured 17%% run-to-run drift on small "
            "kernels."
        ),
    )
    parser.add_argument("--suite", default=None, help="named benchmark suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    args = parser.parse_args(argv)

    if args.baseline is None:
        # A baseline run measures the reference line against eager PyTorch; a
        # candidate run may pick the strongest available comparison.
        args.baseline = "pytorch_autograd" if args.candidate_baseline else "auto"
    if args.candidate_baseline and args.candidate_baseline == args.baseline:
        parser.error(
            f"--candidate-baseline {args.candidate_baseline} would be timed "
            "against itself, reporting 1.0x everywhere. Use "
            "`evograd tier1-bench --identity-control` for that check."
        )

    import torch

    from evograd.bench.tier1 import (
        candidate_provider,
        declared_provider,
        environment_fingerprint,
        pytorch_autograd_provider,
        run_fair_benchmarks,
        torch_compile_provider,
        verify_pair_provider,
    )
    from evograd.bench.harness import DEFAULT_REPS, DEFAULT_WARMUP, run_benchmarks
    from evograd.opdecl.baselines import (
        baseline_candidate_module,
        resolve_performance_baseline,
        verify_performance_baseline,
        verify_runtime_forward,
    )
    from evograd.opdecl.compiled import BUILTIN_MODES
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

    candidate_source = (
        f"the reviewed `{args.candidate_baseline}` baseline standing in as the "
        "candidate (reference line, not a generated kernel)"
        if args.candidate_baseline
        else f"candidate programs under `{args.candidates}`"
    )
    timing_protocol = (
        "evograd-final-runtime-v1 — L2 cleared before every timed region, "
        "batched CUDA events with one synchronize, randomized provider order, "
        "inputs checked for mutation outside the timed regions, median of "
        "samples. Equivalent to triton.testing.do_bench plus the order "
        "randomization and mutation check"
        if args.protocol == "fair"
        else "low-overhead evolution harness — NOT the final protocol; "
        "for iteration only, not for publication"
    )
    report = SuiteReport(
        environment=environment,
        candidate_source=candidate_source,
        timing_protocol=timing_protocol,
    )
    for name, op in sorted(selected.items(), key=lambda i: (i[1].level, i[1].family, i[0])):
        try:
            if args.candidate_baseline:
                # Raises for an operator that does not declare this baseline, or
                # declares it as a compiled (non-pair) one. Both are "we did not
                # run it", which the report must distinguish from "it has no
                # speedup".
                module = baseline_candidate_module(op, args.candidate_baseline)
            else:
                candidate = _find_candidate(args.candidates, name)
                if candidate is None:
                    raise FileNotFoundError(
                        f"no candidate program found under {args.candidates}"
                    )
                module = _load_module(candidate)
        except Exception as exc:
            reason = (
                str(exc)
                if isinstance(exc, (FileNotFoundError, ValueError))
                else f"{type(exc).__name__}: {exc}"
            )
            report.tasks.append(
                TaskResult(
                    op=name,
                    level=op.level,
                    family=op.family,
                    baseline=args.baseline,
                    cases_total=len(op.benchmark_workloads(suite=args.suite)),
                    error=reason,
                )
            )
            print(f"{name:28s} no candidate", file=sys.stderr)
            continue
        try:
            workloads = op.benchmark_workloads(suite=args.suite)
            # A production-spelled eager baseline must match the definition it
            # is verified against before its timings mean anything.
            verify_runtime_forward(op, device=args.device)
            if args.protocol == "fair":
                resolved = resolve_performance_baseline(op, args.baseline)
                if resolved == "pytorch_autograd":
                    baseline_provider = pytorch_autograd_provider(op)
                elif resolved in BUILTIN_MODES:
                    baseline_provider = torch_compile_provider(
                        op,
                        name=resolved,
                        mode=BUILTIN_MODES[resolved],
                        dynamic=False,
                    )
                    compiled_correctness = verify_pair_provider(
                        op, baseline_provider, workloads, device=args.device
                    )
                    if not compiled_correctness.ok:
                        raise RuntimeError(
                            "torch.compile baseline correctness failed:\n"
                            + json.dumps(compiled_correctness.to_dict(), indent=2)
                        )
                else:
                    # Never trust a baseline's timings before its numbers match
                    # the oracle.
                    verify_performance_baseline(op, resolved, device=args.device)
                    baseline_provider = declared_provider(op, resolved)
                fair = run_fair_benchmarks(
                    op,
                    candidate_provider(op, module),
                    baseline_provider,
                    workloads=workloads,
                    device=args.device,
                    # One block. What remains — L2 cleared before every timed
                    # region, batched events with a single synchronize, median
                    # of the samples — is exactly what triton.testing.do_bench
                    # does, and what KernelBench, TritonBench and FastKernels
                    # measure with. Repeated blocks exist to feed the block
                    # bootstrap; with one block the resampling has nothing to
                    # draw from and every interval collapses to zero width, so
                    # asking for three would triple the cost to produce a
                    # statistic the suite does not report.
                    blocks=1,
                    **(
                        {"warmup": args.warmup} if args.warmup is not None else {}
                    ),
                    **({"reps": args.reps} if args.reps is not None else {}),
                )
                task = task_from_fair_report(
                    name,
                    op.level,
                    op.family,
                    resolved,
                    fair,
                    backward_may_overwrite=tuple(
                        getattr(op, "backward_may_overwrite", ()) or ()
                    ),
                    op=op,
                )
            else:
                benchmark = run_benchmarks(
                    op,
                    module,
                    warmup=args.warmup if args.warmup is not None else DEFAULT_WARMUP,
                    reps=args.reps if args.reps is not None else DEFAULT_REPS,
                    device=args.device,
                    workloads=workloads,
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

    # Level-4 workloads: measured by tier3-bench, reported here. A declared
    # workload with no supplied report is a coverage fact, not an omission.
    from evograd.ops import WORKLOADS

    tier3_reports = {}
    for entry in args.tier3_report:
        workload_name, _, path = entry.partition("=")
        if not path or workload_name not in WORKLOADS:
            parser.error(
                f"--tier3-report wants WORKLOAD=PATH with a declared workload "
                f"({sorted(WORKLOADS)}), got {entry!r}"
            )
        tier3_reports[workload_name] = Path(path)

    for name, decl in sorted(WORKLOADS.items()):
        if args.levels and decl.level not in args.levels:
            continue
        if args.ops and name not in args.ops:
            continue
        if name in tier3_reports:
            tier3 = json.loads(tier3_reports[name].read_text(encoding="utf-8"))
            task = task_from_tier3_report(name, decl.family, tier3)
        else:
            task = TaskResult(
                op=name,
                level=decl.level,
                family=decl.family,
                baseline="eager",
                tier="model",
                cases_total=len(decl.benchmark),
                error=(
                    "not measured in this run; produce a report with "
                    f"`evograd tier3-bench --workload {name}` and pass it via "
                    f"--tier3-report {name}=<path>"
                ),
            )
        report.tasks.append(task)
        marker = f"{task.speedup:.3f}x" if task.speedups else "—"
        print(
            f"{name:28s} L{decl.level} {marker:>9s} "
            f"coverage {task.cases_ok}/{task.cases_total}",
            file=sys.stderr,
        )

    paths = write_report(report, args.out)
    print(json.dumps(report.to_dict()["overall"], indent=2, sort_keys=True))
    print(f"report: {paths['markdown']}")
    return 0 if all(task.ok for task in report.tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
