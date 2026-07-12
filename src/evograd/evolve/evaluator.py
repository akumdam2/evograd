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

import importlib.util
import json
import os
import sys
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from typing import Any

import torch

from evograd.opdecl.activity import OpDecl
from evograd.opdecl.bind import backward_const_kwargs, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle
from evograd.bench.harness import (
    DEFAULT_REPS,
    DEFAULT_WARMUP,
    normalize_saved,
    run_benchmarks,
    saved_bytes,
)
from evograd.evolve.scoring import ScoringPolicy, score_from_aggregate

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # standalone use without openevolve installed

    class EvaluationResult:  # type: ignore[no-redef]
        def __init__(self, metrics: dict, artifacts: dict):
            self.metrics = metrics
            self.artifacts = artifacts


CAPTURE_NATIVE_OUTPUT = os.environ.get("EVOGRAD_CAPTURE_NATIVE_OUTPUT", "1").lower() not in (
    "0",
    "false",
    "no",
)
NATIVE_OUTPUT_TAIL_BYTES = int(os.environ.get("EVOGRAD_NATIVE_OUTPUT_TAIL_BYTES", "65536"))


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: dict[str, float], artifacts: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(metrics=metrics, artifacts={k: _json(v) for k, v in artifacts.items()})


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


def _run_correctness(op: OpDecl, module, device: str = "cuda") -> dict[str, Any]:
    fwd, bwd = lookup_pair(op, module)
    grad_names = op.grad_names()
    reports = []
    passed = 0
    passed_by_output = {name: 0 for name in grad_names}

    for workload in op.correctness:
        report: dict[str, Any] = {"dims": dict(workload.dims), "dtype": workload.dtype}
        try:
            inputs = make_case_inputs(op, workload, device=device)
            _y_ref, expected = oracle(op, inputs)
            positional = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
            y, saved = fwd(*positional)
            kwargs = backward_const_kwargs(op, bwd, inputs)
            actual = bwd(inputs[op.upstream_grad_name], saved, **kwargs)
            actual = (actual,) if torch.is_tensor(actual) else tuple(actual)
            torch.cuda.synchronize()

            with torch.no_grad():
                _y2, saved2 = fwd(*positional)
                saved_tensors = normalize_saved(saved2)
            report["forward_shape"] = list(y.shape)
            report["saved_tensors"] = [
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "bytes": t.numel() * t.element_size(),
                }
                for t in saved_tensors
            ]
            report["saved_bytes"] = saved_bytes(saved_tensors)

            atol, rtol = op.tolerance_for(workload)
            correct = len(actual) == len(grad_names)
            if not correct:
                report["error_message"] = (
                    f"backward returned {len(actual)} gradients, expected {len(grad_names)}"
                )
            for name, got in zip(grad_names, actual):
                ref = expected[name]
                if got is None:
                    ok = False
                    max_abs = max_rel = float("inf")
                else:
                    max_abs, max_rel = _max_errors(got, ref)
                    ok = bool(torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol))
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
        **{f"{name}_correctness": passed_by_output[name] / total for name in grad_names},
        "reports": reports,
    }


def build_evaluate(op: OpDecl, policy: ScoringPolicy, *, warmup: int | None = None, reps: int | None = None):
    """Return the ``evaluate(program_path)`` callable OpenEvolve loads."""
    warmup = DEFAULT_WARMUP if warmup is None else warmup
    reps = DEFAULT_REPS if reps is None else reps

    def evaluate(program_path: str) -> EvaluationResult:
        if not torch.cuda.is_available():
            return _result(
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
            return _result(
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
                    benchmark = run_benchmarks(
                        op,
                        module,
                        warmup=warmup,
                        reps=reps,
                        geomean_weights=policy.geomean_weights,
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
            return _result({"combined_score": -1e9, "correct": 0.0}, artifacts)

        native_output = _native_output_tail(native_output_path)
        grad_names = op.grad_names()
        if correctness["passed"] != correctness["total"]:
            metrics = {
                "combined_score": -1e6 + float(correctness["partial_correctness"]),
                "correct": 0.0,
                "partial_correctness": float(correctness["partial_correctness"]),
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
            return _result(metrics, artifacts)

        aggregate = benchmark["aggregate"]
        combined_score, score_details = score_from_aggregate(aggregate, policy)
        metrics = {
            "combined_score": float(combined_score),
            "correct": 1.0,
            "partial_correctness": 1.0,
            "speedup": float(aggregate["speedup_vs_pytorch_autograd_backward"]),
            "full_step_speedup": float(aggregate["speedup_vs_pytorch_autograd_full_step"]),
            "forward_ms": float(aggregate["forward_ms"]),
            "backward_from_saved_ms": float(aggregate["backward_from_saved_ms"]),
            "forward_backward_full_step_ms": float(aggregate["forward_backward_full_step_ms"]),
            "autograd_forward_backward_full_step_ms": float(
                aggregate["autograd_forward_backward_full_step_ms"]
            ),
            "baseline_latency_ms": float(aggregate["pytorch_autograd_backward_ms"]),
            "baseline_full_step_ms": float(aggregate["pytorch_autograd_full_step_ms"]),
            "saved_bytes": float(aggregate["saved_bytes"]),
            "input_bytes": float(aggregate["input_bytes"]),
        }
        metrics.update({k: float(v) for k, v in score_details.items()})
        metrics.update({f"{name}_correct": 1.0 for name in grad_names})
        benchmark["score_mode"] = policy.mode
        benchmark["scoring_policy"] = policy.name
        benchmark["full_step_weight"] = policy.full_step_weight
        benchmark["memory_penalty_weight"] = policy.memory_penalty_weight
        benchmark["worst_case_guard"] = policy.worst_case_guard
        benchmark["warmup"] = warmup
        benchmark["reps"] = reps
        return _result(metrics, {"correctness": correctness, "benchmark": benchmark})

    return evaluate


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
