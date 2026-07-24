"""Measure evolved generalist/specialist programs and emit shape dispatch."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from evograd.evolve.scoring import geomean
from evograd.opdecl.activity import OpDecl, Workload


def _case_key(case: dict) -> tuple:
    return tuple(sorted(case["dims"].items())), case["dtype"]


@contextmanager
def _environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _evaluate_program(
    op: OpDecl,
    path: Path,
    *,
    baseline: str,
    warmup: int,
    reps: int,
    timeout: int,
    cache_path: Path,
) -> dict:
    from evograd.evolve.evaluator import evaluate_isolated
    from evograd.opdecl.baselines import resolve_performance_baseline

    # evaluator_entry resolves EVOGRAD_OP at module import time. Point the child
    # process at the file without importing it in this (unconfigured) process.
    evaluator_file = Path(__file__).with_name("evolve") / "evaluator_entry.py"
    baseline = resolve_performance_baseline(op, baseline)
    env = {
        "EVOGRAD_OP": op.name,
        "EVOGRAD_FORWARD_OVERRIDE": op.forward,
        "EVOGRAD_SCORING": "speed_memory_min",
        "EVOGRAD_BENCHMARK_SUITE": "full",
        "EVOGRAD_BENCHMARK_WARMUP": str(warmup),
        "EVOGRAD_BENCHMARK_REPS": str(reps),
        "EVOGRAD_PERFORMANCE_BASELINE": baseline,
        "EVOGRAD_BASELINE_TIMING_CACHE_PATH": str(cache_path),
    }
    if op.declaration:
        env["EVOGRAD_DECLARATION"] = op.declaration
    with _environment(env):
        result = evaluate_isolated(str(evaluator_file), str(path), timeout=timeout)
    payload = {"metrics": result.metrics, "artifacts": result.artifacts}
    if float(result.metrics.get("correct", 0.0)) != 1.0:
        return payload
    raw = result.artifacts.get("benchmark")
    payload["benchmark"] = json.loads(raw) if isinstance(raw, str) else raw
    return payload


def _program_geomean(cases: dict[tuple, dict], which: str) -> float | None:
    values = [
        report[f"speedup_vs_baseline_{which}"]
        for report in cases.values()
        if math.isfinite(float(report[f"speedup_vs_baseline_{which}"]))
        and float(report[f"speedup_vs_baseline_{which}"]) > 0
    ]
    return geomean(values) if values else None


def _best_tag(prefix: str, measurements: dict[str, dict[tuple, dict]]) -> str | None:
    candidates = []
    for tag, cases in measurements.items():
        if tag == prefix or tag.startswith(f"{prefix}_"):
            score = _program_geomean(cases, "raw_full_step")
            if score is not None:
                candidates.append((tag, score))
    return max(candidates, key=lambda item: item[1])[0] if candidates else None


def _workload_from_case(case: dict) -> Workload:
    return Workload(dims=dict(case["dims"]), dtype=case["dtype"])


def _emit_dispatcher(
    op: OpDecl,
    programs: dict[str, Path],
    small_tag: str,
    large_tag: str,
    threshold: float,
    output: Path,
) -> None:
    small_path = str(programs[small_tag].resolve())
    large_path = str(programs[large_tag].resolve())
    source = f'''"""Auto-generated evograd shape dispatcher for {op.name}.

Routes at regime_feature < {threshold!r}. Saved state is prefixed by a plain
integer route tag, which contributes no tensor-memory bytes.
"""
import importlib.util as _ilu
from types import SimpleNamespace as _Namespace

from evograd.ops import get_op as _get_op

_OP = _get_op({op.name!r})
_THRESHOLD = {threshold!r}
_SMALL_PATH = {small_path!r}
_LARGE_PATH = {large_path!r}


def _load(label, path):
    spec = _ilu.spec_from_file_location("evograd_dispatch_" + label, path)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROGRAMS = (_load("small", _SMALL_PATH), _load("large", _LARGE_PATH))


def _lookup(module, generic, prefixed):
    value = getattr(module, generic, None)
    return value if value is not None else getattr(module, prefixed)


def _route(args, kwargs):
    dims = {{}}
    for index, declared in enumerate(_OP.args):
        shape = getattr(declared, "shape", None)
        if shape is None:
            continue
        if index < len(args):
            value = args[index]
        elif declared.name in kwargs:
            value = kwargs[declared.name]
        else:
            raise TypeError(f"missing required argument {{declared.name!r}}")
        tokens = [part.strip() for part in shape[1:-1].split(",") if part.strip()]
        if len(tokens) != value.dim():
            raise ValueError(f"{{declared.name}} rank mismatch: {{tuple(value.shape)}} vs {{shape}}")
        for token, actual in zip(tokens, value.shape):
            if token.isdigit():
                if int(token) != int(actual):
                    raise ValueError(f"{{declared.name}} fixed dimension {{token}} != {{actual}}")
            elif token in dims and dims[token] != int(actual):
                raise ValueError(f"inconsistent dimension {{token}}: {{dims[token]}} != {{actual}}")
            else:
                dims[token] = int(actual)
    feature = float(_OP.regime_feature(_Namespace(dims=dims)))
    return 0 if feature < _THRESHOLD else 1


def {op.forward_fn_name}(*args, **kwargs):
    route = _route(args, kwargs)
    forward = _lookup(_PROGRAMS[route], "forward_with_saved", {op.forward_fn_name!r})
    output, saved = forward(*args, **kwargs)
    saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
    return output, (route, *saved)


def {op.backward_fn_name}(dout, saved_tensors, *args, **kwargs):
    route, *saved = tuple(saved_tensors)
    backward = _lookup(_PROGRAMS[int(route)], "backward_from_saved", {op.backward_fn_name!r})
    return backward(dout, tuple(saved), *args, **kwargs)
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")


def _render_markdown(op: OpDecl, report: dict) -> str:
    baseline = report["baseline"].replace("_", " ")
    dispatch_score = report.get("threshold_dispatch_geomean")
    full_score = report.get("full_geomean")
    lines = [
        f"# {op.name} autograd-pair benchmark vs {baseline}",
        "",
        f"Measured baseline: **{baseline}**.",
        "",
        "| deployment | full-step geomean |",
        "| --- | ---: |",
    ]
    if full_score is not None:
        lines.append(f"| full-trained generalist | {full_score:.3f} |")
    if dispatch_score is not None:
        lines.append(f"| small/large threshold dispatch | {dispatch_score:.3f} |")
    lines.extend(
        [
            "",
            f"Best programs: full=`{report.get('full_best')}`, "
            f"small=`{report.get('small_best')}`, large=`{report.get('large_best')}`.",
            f"Regime collapsed: **{report.get('regime_collapsed', True)}**.",
            "",
        ]
    )
    return "\n".join(lines)


def dispatch(
    op: OpDecl,
    programs: dict[str, Path],
    *,
    output_dir: Path,
    baseline: str = "auto",
    timeout: int = 850,
    warmup: int = 10,
    reps: int = 50,
) -> dict:
    if not programs:
        raise ValueError("at least one tagged program is required")
    for tag, path in programs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{tag}: program not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / ".baseline_timing_cache.json"
    measurements: dict[str, dict[tuple, dict]] = {}
    failures = {}
    measured_baseline = None
    for tag, path in programs.items():
        payload = _evaluate_program(
            op,
            path,
            baseline=baseline,
            warmup=warmup,
            reps=reps,
            timeout=timeout,
            cache_path=cache_path,
        )
        benchmark = payload.get("benchmark")
        if not benchmark:
            measurements[tag] = {}
            failures[tag] = payload
            continue
        measured_baseline = benchmark["performance_baseline"]
        measurements[tag] = {
            _case_key(case): case for case in benchmark["cases"]
        }

    full_tag = _best_tag("full", measurements)
    small_tag = _best_tag("small", measurements)
    large_tag = _best_tag("large", measurements)
    report = {
        "op": op.name,
        "baseline": measured_baseline or baseline,
        "per_program_geomean_full": {
            tag: _program_geomean(cases, "raw_full_step")
            for tag, cases in measurements.items()
        },
        "per_program_geomean_bwd": {
            tag: _program_geomean(cases, "backward")
            for tag, cases in measurements.items()
        },
        "failures": failures,
        "full_best": full_tag,
        "small_best": small_tag,
        "large_best": large_tag,
        "regime_split": op.regime_split,
    }
    final_program = output_dir / f"{op.name}_final_dispatched.py"

    if op.regime_feature is None or not (small_tag and large_tag):
        winner = full_tag or small_tag or large_tag
        if winner is None:
            raise RuntimeError("no candidate passed correctness and benchmarking")
        shutil.copyfile(programs[winner], final_program)
        report.update(
            {
                "regime_collapsed": True,
                "best_threshold": None,
                "full_geomean": report["per_program_geomean_full"].get(full_tag),
            }
        )
    else:
        common_keys = set(measurements[small_tag]) & set(measurements[large_tag])
        features = sorted(
            {
                float(
                    op.regime_feature(
                        _workload_from_case(measurements[small_tag][key])
                    )
                )
                for key in common_keys
            }
        )
        thresholds = [0.0]
        thresholds.extend(
            math.sqrt(left * right)
            for left, right in zip(features, features[1:])
            if left > 0 and right > 0
        )
        thresholds.append(float("inf"))

        def score(threshold):
            speedups = []
            for key in common_keys:
                case = measurements[small_tag][key]
                feature = float(op.regime_feature(_workload_from_case(case)))
                tag = small_tag if feature < threshold else large_tag
                speedup = measurements[tag][key]["speedup_vs_baseline_raw_full_step"]
                if speedup > 0 and math.isfinite(speedup):
                    speedups.append(speedup)
            return geomean(speedups) if speedups else None

        scored = [(threshold, score(threshold)) for threshold in thresholds]
        scored = [(threshold, value) for threshold, value in scored if value is not None]
        threshold, dispatch_score = max(scored, key=lambda item: item[1])
        collapsed = threshold in (0.0, float("inf"))
        if collapsed:
            winner = large_tag if threshold == 0.0 else small_tag
            shutil.copyfile(programs[winner], final_program)
        else:
            _emit_dispatcher(
                op, programs, small_tag, large_tag, threshold, final_program
            )
        report.update(
            {
                "regime_collapsed": collapsed,
                "best_threshold": threshold if math.isfinite(threshold) else None,
                "threshold_dispatch_geomean": dispatch_score,
                "full_geomean": report["per_program_geomean_full"].get(full_tag),
            }
        )

    report["final_program"] = str(final_program)
    report_path = output_dir / f"{op.name}_dispatch_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / f"RESULTS_{op.name}_vs_{report['baseline']}.md"
    markdown_path.write_text(_render_markdown(op, report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None)
    parser.add_argument(
        "--program", action="append", default=[], help="tag=path; repeatable"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="auto")
    parser.add_argument("--timeout", type=int, default=850)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args(argv)
    programs = {}
    for value in args.program:
        if "=" not in value:
            parser.error(f"--program must be tag=path, got {value!r}")
        tag, raw_path = value.split("=", 1)
        programs[tag] = Path(raw_path)
    from evograd.ops import get_op, load_op

    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    report = dispatch(
        op,
        programs,
        output_dir=args.output_dir,
        baseline=args.baseline,
        timeout=args.timeout,
        warmup=args.warmup,
        reps=args.reps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
