"""Generic OpenEvolve evaluator: one implementation for every operator.

``build_evaluate(op, policy)`` returns the ``evaluate(program_path)`` callable
OpenEvolve expects. Correctness is a hard gate (checked against the autograd
oracle derived from the declaration); only correct candidates are benchmarked
and scored by the chosen :class:`ScoringPolicy`.

Replaces the per-bench ``evaluator_autograd_pair*.py`` families (7 files for
layernorm alone) — metric names and artifact schema are preserved so old and
new runs are comparable.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from typing import Any

import torch

from evograd.opdecl.activity import OpDecl
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle
from evograd.bench.harness import (
    DEFAULT_REPS,
    DEFAULT_WARMUP,
    describe_saved,
    normalize_saved,
    run_benchmarks,
    saved_bytes,
)
from evograd.evolve.scoring import (
    CUSTOM_FEATURE_DIMENSION_DEFAULTS,
    ScoringPolicy,
    regime_speedup_metrics,
    score_from_aggregate,
    shape_specialization_from_cases,
)

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # standalone use without openevolve installed

    class EvaluationResult:  # type: ignore[no-redef]
        def __init__(self, metrics: dict, artifacts: dict):
            self.metrics = metrics
            self.artifacts = artifacts


CAPTURE_NATIVE_OUTPUT = os.environ.get(
    "EVOGRAD_CAPTURE_NATIVE_OUTPUT",
    os.environ.get("AUTOGRAD_PAIR_CAPTURE_NATIVE_OUTPUT", "1"),
).lower() not in (
    "0",
    "false",
    "no",
)
NATIVE_OUTPUT_TAIL_BYTES = int(
    os.environ.get(
        "EVOGRAD_NATIVE_OUTPUT_TAIL_BYTES",
        os.environ.get("AUTOGRAD_PAIR_NATIVE_OUTPUT_TAIL_BYTES", "65536"),
    )
)
EVAL_ISOLATION_TIMEOUT = int(os.environ.get("EVOGRAD_EVAL_ISOLATION_TIMEOUT", "850"))

try:
    _PRISTINE_STDOUT_FD = os.dup(1)
    _PRISTINE_STDERR_FD = os.dup(2)
except OSError:
    _PRISTINE_STDOUT_FD = _PRISTINE_STDERR_FD = -1


def _restore_std_fds() -> None:
    if _PRISTINE_STDOUT_FD < 0:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os.dup2(_PRISTINE_STDOUT_FD, 1)
    os.dup2(_PRISTINE_STDERR_FD, 2)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: dict[str, float], artifacts: dict[str, Any]) -> EvaluationResult:
    # Custom MAP-Elites feature dimensions must exist in every result;
    # OpenEvolve raises when a configured dimension is missing from a
    # program's metrics, and failure paths otherwise omit them.
    for dim, default in CUSTOM_FEATURE_DIMENSION_DEFAULTS.items():
        metrics.setdefault(dim, default)
    return EvaluationResult(metrics=metrics, artifacts={k: _json(v) for k, v in artifacts.items()})


def archive_program(
    program_path: str, result: EvaluationResult, directory: str | os.PathLike
) -> str | None:
    """Keep a copy of one evaluated candidate under ``directory``.

    OpenEvolve only hands back the single best program, and its own checkpoints
    hold the database rather than browsable sources, so every other candidate
    the run paid an LLM call and a GPU evaluation for is otherwise gone. Called
    once per evaluation from the parent process (never the isolation child,
    which would double-write).

    Filenames are ``<UTC timestamp>_<score>_<sha1>.py`` so a directory listing
    sorts chronologically, and identical code evaluated twice is stored once.
    Every evaluation — including the ones that scored below zero — appends a
    line to ``index.jsonl``, which is the record of what actually happened.
    Archiving must never take a run down: any failure here is swallowed.
    """
    try:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        source = pathlib.Path(program_path).read_bytes()
        digest = hashlib.sha1(source).hexdigest()[:12]

        metrics = result.metrics or {}
        score = float(metrics.get("combined_score", 0.0))
        existing = next(iter(directory.glob(f"*_{digest}.py")), None)
        if existing is None:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            target = directory / f"{stamp}_{score:+011.3f}_{digest}.py"
            target.write_bytes(source)
            target.with_suffix(".json").write_text(
                _json({"metrics": metrics, "artifacts": result.artifacts or {}}),
                encoding="utf-8",
            )
        else:
            target = existing

        record = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "file": target.name,
            "sha1": digest,
            "source": os.path.basename(program_path),
            "duplicate": existing is not None,
            "metrics": metrics,
        }
        # One short line, one write, O_APPEND: concurrent evaluator processes
        # interleave lines rather than corrupting them.
        with open(directory / "index.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return str(target)
    except Exception:  # pragma: no cover - archiving is never worth a failed run
        return None


def _baseline_metrics(aggregate: dict[str, Any]) -> dict[str, float]:
    """Expose selected-baseline metrics without claiming it was PyTorch."""
    return {
        "speedup": float(aggregate["speedup_vs_baseline_backward"]),
        "full_step_speedup": float(aggregate["speedup_vs_baseline_full_step"]),
        "baseline_latency_ms": float(aggregate["baseline_backward_ms"]),
        "baseline_full_step_ms": float(aggregate["baseline_full_step_ms"]),
    }


@contextmanager
def _capture_native_output():
    """Capture Triton/ptxas dumps at the fd level so doomed candidates cannot
    flood OpenEvolve's terminal with PTX reproducers."""
    if not CAPTURE_NATIVE_OUTPUT:
        yield None
        return

    sys.stdout.flush()
    sys.stderr.flush()
    fd, path = tempfile.mkstemp(prefix="evograd_native_", suffix=".log")
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        yield path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _native_output_tail(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > NATIVE_OUTPUT_TAIL_BYTES:
                handle.seek(-NATIVE_OUTPUT_TAIL_BYTES, os.SEEK_END)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        return {
            "captured": True,
            "bytes": size,
            "tail_bytes": min(size, NATIVE_OUTPUT_TAIL_BYTES),
            "tail": text,
        }
    except Exception as exc:
        return {"captured": False, "error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _load_module(program_path: str):
    module_name = f"evograd_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _max_errors(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = (candidate.float() - reference.float()).abs()
    max_abs = float(diff.max())
    max_rel = float((diff / reference.float().abs().clamp(min=1e-8)).max())
    return max_abs, max_rel


def _compare_tensor(candidate, reference, atol: float, rtol: float) -> tuple[bool, float, float]:
    """Check values and the public tensor contract (shape and dtype)."""
    if not torch.is_tensor(candidate):
        return False, float("inf"), float("inf")
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        return False, float("inf"), float("inf")
    max_abs, max_rel = _max_errors(candidate, reference)
    ok = bool(
        torch.allclose(candidate.float(), reference.float(), atol=atol, rtol=rtol)
    )
    return ok, max_abs, max_rel


def _run_correctness(op: OpDecl, module, device: str = "cuda") -> dict[str, Any]:
    fwd, bwd = lookup_pair(op, module)
    grad_names = op.grad_names()
    reports = []
    passed = 0
    passed_by_output = {name: 0 for name in grad_names}
    forward_passed = 0

    for workload in op.correctness:
        report: dict[str, Any] = {"dims": dict(workload.dims), "dtype": workload.dtype}
        try:
            inputs = make_case_inputs(op, workload, device=device)
            y_ref, expected = oracle(op, inputs)
            positional = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
            y, saved = fwd(*positional)
            kwargs = backward_inactive_kwargs(op, bwd, inputs)
            actual = bwd(inputs[op.upstream_grad_name], saved, **kwargs)
            actual = (actual,) if torch.is_tensor(actual) else tuple(actual)
            torch.cuda.synchronize()

            with torch.no_grad():
                _y2, saved2 = fwd(*positional)
                saved_tensors = normalize_saved(saved2)
            report["forward_shape"] = list(y.shape)
            report["saved_tensors"] = describe_saved(saved_tensors)
            report["saved_bytes"] = saved_bytes(saved_tensors)

            atol, rtol = op.tolerance_for(workload)
            forward_ok, forward_abs, forward_rel = _compare_tensor(y, y_ref, atol, rtol)
            report["forward_correct"] = forward_ok
            report["forward_max_abs_error"] = forward_abs
            report["forward_max_rel_error"] = forward_rel
            forward_passed += int(forward_ok)
            correct = forward_ok and len(actual) == len(grad_names)
            if len(actual) != len(grad_names):
                report["error_message"] = (
                    f"backward returned {len(actual)} gradients, expected {len(grad_names)}"
                )
            for name, got in zip(grad_names, actual):
                ref = expected[name]
                grad_atol, grad_rtol = op.tolerance_for(workload, name)
                ok, max_abs, max_rel = _compare_tensor(got, ref, grad_atol, grad_rtol)
                report[f"{name}_correct"] = ok
                report[f"{name}_max_abs_error"] = max_abs
                report[f"{name}_max_rel_error"] = max_rel
                correct = correct and ok
                passed_by_output[name] += int(ok)
            report["correct"] = correct
            passed += int(correct)
        except Exception as exc:
            report.update(
                {
                    "correct": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        reports.append(report)

    total = max(1, len(op.correctness))
    return {
        "passed": passed,
        "total": len(op.correctness),
        "partial_correctness": passed / total,
        "forward_correctness": forward_passed / total,
        **{f"{name}_correctness": passed_by_output[name] / total for name in grad_names},
        "reports": reports,
    }


def _smoke_benchmark_shapes(
    op: OpDecl,
    module,
    workloads,
    device: str = "cuda",
) -> dict[str, Any] | None:
    """Run one untimed pair invocation per benchmark shape.

    The caller's subprocess timeout turns shape-dependent Triton deadlocks into
    a rejected candidate by destroying the child CUDA context.
    """
    fwd, bwd = lookup_pair(op, module)
    selected = tuple(
        workloads
        if workloads is not None
        else (op.coverage if op.coverage else op.benchmark)
    )
    print(
        f"[smoke] correctness passed; checking {len(selected)} benchmark shapes",
        file=sys.stderr,
        flush=True,
    )
    for workload in selected:
        label = {"dims": workload.dims, "dtype": workload.dtype}
        print(f"[smoke] {label}", file=sys.stderr, flush=True)
        try:
            inputs = make_case_inputs(op, workload, device=device)
            positional = [
                inputs.get(arg.name, getattr(arg, "default", None)) for arg in op.args
            ]
            kwargs = backward_inactive_kwargs(op, bwd, inputs)
            with torch.no_grad():
                _y, saved = fwd(*positional)
                bwd(
                    inputs[op.upstream_grad_name],
                    normalize_saved(saved),
                    **kwargs,
                )
            torch.cuda.synchronize()
        except Exception as exc:
            return {
                "error_type": "BenchmarkShapeSmokeError",
                "error_message": f"{type(exc).__name__}: {exc}",
                "case": label,
                "traceback": traceback.format_exc(limit=8),
            }
    return None


def build_evaluate(
    op: OpDecl,
    policy: ScoringPolicy,
    *,
    warmup: int | None = None,
    reps: int | None = None,
    benchmark_workloads=None,
    performance_baseline: str = "pytorch_autograd",
):
    """Return the ``evaluate(program_path)`` callable OpenEvolve loads."""
    warmup = DEFAULT_WARMUP if warmup is None else warmup
    reps = DEFAULT_REPS if reps is None else reps

    def emit(
        metrics: dict[str, float],
        artifacts: dict[str, Any],
        cases: list[dict] | None = None,
    ) -> EvaluationResult:
        # Always publish regime axes so MAP-Elites configs that select them
        # never hit a missing-metric ValueError on failed candidates.
        merged = dict(metrics)
        merged.update(
            regime_speedup_metrics(
                cases,
                regime_feature=op.regime_feature,
                regime_split=float(op.regime_split or 0.0),
            )
        )
        return _result(merged, artifacts)

    def evaluate(program_path: str) -> EvaluationResult:
        _restore_std_fds()
        if not torch.cuda.is_available():
            return emit(
                {"combined_score": -1e9, "correct": 0.0},
                {
                    "failure": {
                        "error_type": "RuntimeUnavailable",
                        "error_message": "CUDA is not available",
                    }
                },
            )
        try:
            module = _load_module(program_path)
            lookup_pair(op, module)  # validate API before running anything
        except Exception as exc:
            return emit(
                {"combined_score": -1e9, "correct": 0.0},
                {
                    "failure": {
                        "error_type": "ImportOrApiError",
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                },
            )

        correctness = None
        benchmark = None
        native_output_path = None
        try:
            with _capture_native_output() as captured_path:
                native_output_path = captured_path
                correctness = _run_correctness(op, module)
                if correctness["passed"] == correctness["total"]:
                    smoke_workloads = op.coverage if op.coverage else benchmark_workloads
                    smoke_failure = _smoke_benchmark_shapes(op, module, smoke_workloads)
                    if smoke_failure is None:
                        benchmark = run_benchmarks(
                            op,
                            module,
                            warmup=warmup,
                            reps=reps,
                            geomean_weights=policy.geomean_weights,
                            workloads=benchmark_workloads,
                            performance_baseline=performance_baseline,
                        )
                    else:
                        native_output = _native_output_tail(native_output_path)
                        artifacts = {
                            "correctness": correctness,
                            "failure": smoke_failure,
                        }
                        if native_output and native_output.get("bytes", 0) > 0:
                            artifacts["native_output"] = native_output
                        return emit(
                            {
                                "combined_score": -1e6,
                                "correct": 0.0,
                                "partial_correctness": 1.0,
                            },
                            artifacts,
                        )
        except Exception as exc:
            native_output = _native_output_tail(native_output_path)
            artifacts: dict[str, Any] = {
                "failure": {
                    "error_type": "CandidateExecutionError",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            }
            if correctness is not None:
                artifacts["correctness"] = correctness
            if native_output and native_output.get("bytes", 0) > 0:
                artifacts["native_output"] = native_output
            return emit({"combined_score": -1e9, "correct": 0.0}, artifacts)

        native_output = _native_output_tail(native_output_path)
        grad_names = op.grad_names()
        if correctness["passed"] != correctness["total"]:
            metrics = {
                "combined_score": -1e6 + float(correctness["partial_correctness"]),
                "correct": 0.0,
                "partial_correctness": float(correctness["partial_correctness"]),
                "forward_correct": float(correctness["forward_correctness"]),
            }
            metrics.update(
                {
                    f"{name}_correct": float(correctness[f"{name}_correctness"])
                    for name in grad_names
                }
            )
            artifacts = {"correctness": correctness}
            if native_output and native_output.get("bytes", 0) > 0:
                artifacts["native_output"] = native_output
            return emit(metrics, artifacts)

        aggregate = benchmark["aggregate"]
        combined_score, score_details = score_from_aggregate(aggregate, policy)
        metrics = {
            "combined_score": float(combined_score),
            "correct": 1.0,
            "partial_correctness": 1.0,
            "forward_correct": 1.0,
            "forward_ms": float(aggregate["forward_ms"]),
            "backward_from_saved_ms": float(aggregate["backward_from_saved_ms"]),
            "forward_backward_full_step_ms": float(aggregate["forward_backward_full_step_ms"]),
            "raw_forward_backward_full_step_ms": float(
                aggregate["raw_forward_backward_full_step_ms"]
            ),
            "autograd_forward_backward_full_step_ms": float(
                aggregate["autograd_forward_backward_full_step_ms"]
            ),
            "baseline_raw_full_step_ms": float(aggregate["baseline_raw_full_step_ms"]),
            "saved_bytes": float(aggregate["saved_bytes"]),
            "input_bytes": float(aggregate["input_bytes"]),
        }
        metrics.update(_baseline_metrics(aggregate))
        metrics.update({k: float(v) for k, v in score_details.items()})
        metrics.update({f"{name}_correct": 1.0 for name in grad_names})
        try:
            metrics["shape_specialization"] = float(
                shape_specialization_from_cases(benchmark.get("cases") or [])
            )
        except Exception:
            pass  # _result fills the neutral default; never fail an eval over this
        benchmark["score_mode"] = policy.mode
        benchmark["scoring_policy"] = policy.name
        benchmark["performance_baseline"] = performance_baseline
        benchmark["full_step_weight"] = policy.full_step_weight
        benchmark["memory_penalty_weight"] = policy.memory_penalty_weight
        benchmark["worst_case_guard"] = policy.worst_case_guard
        benchmark["warmup"] = warmup
        benchmark["reps"] = reps
        return emit(
            metrics,
            {"correctness": correctness, "benchmark": benchmark},
            cases=benchmark["cases"],
        )

    return evaluate


def evaluate_isolated(
    evaluator_file: str,
    program_path: str,
    *,
    timeout: int | None = None,
) -> EvaluationResult:
    """Evaluate in a killable subprocess so a deadlocked GPU kernel cannot survive."""
    timeout = timeout or EVAL_ISOLATION_TIMEOUT
    fd, result_path = tempfile.mkstemp(prefix="evograd_result_", suffix=".json")
    os.close(fd)
    env = {
        **os.environ,
        "EVOGRAD_EVAL_CHILD": "1",
        "EVOGRAD_RESULT_JSON": result_path,
    }
    try:
        try:
            process = subprocess.run(
                [sys.executable, evaluator_file, program_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return _result(
                {"combined_score": -1e6, "correct": 0.0},
                {
                    "failure": {
                        "error_type": "EvaluationTimeoutKilled",
                        "error_message": (
                            f"evaluation exceeded {timeout}s and was killed"
                        ),
                        "stderr_tail": stderr[-4000:],
                    }
                },
            )
        try:
            with open(result_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return _result(
                {"combined_score": -1e9, "correct": 0.0},
                {
                    "failure": {
                        "error_type": "IsolatedEvalCrashed",
                        "error_message": (
                            f"evaluator subprocess exited rc={process.returncode} "
                            "without a result"
                        ),
                        "stdout_tail": process.stdout[-4000:],
                        "stderr_tail": process.stderr[-4000:],
                    }
                },
            )
        return EvaluationResult(
            metrics=payload["metrics"], artifacts=payload.get("artifacts", {})
        )
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def main(argv: list[str]) -> int:
    """Standalone: python -m evograd.evolve.evaluator --op X [--scoring Y] PROGRAM"""
    import argparse

    from evograd.evolve.scoring import get_policy
    from evograd.ops import get_op

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--scoring", default="speed_memory")
    parser.add_argument("program")
    args = parser.parse_args(argv)

    evaluate = build_evaluate(get_op(args.op), get_policy(args.scoring))
    result = evaluate(args.program)
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
