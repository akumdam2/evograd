"""Killable child process for one candidate and a batch of shape instances."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import traceback

import torch

from evograd.bench.harness import run_benchmarks
from evograd.evolve.evaluator import (
    _load_module,
    _run_correctness,
    _smoke_benchmark_shapes,
)
from evograd.opdecl.activity import Workload
from evograd.opdecl.bind import lookup_pair
from evograd.ops import get_op


def _failure(shape_ids: list[str], kind: str, message: str, **extra) -> dict:
    info = {"status": kind, "error": message, **extra}
    return {
        "gate_ok": False,
        "shapes": {
            shape_id: {"score": 0.0, "info": {"shape_id": shape_id, **info}}
            for shape_id in shape_ids
        },
    }


def _score_case(case: dict) -> tuple[float, dict]:
    backward = float(case["speedup_vs_baseline_backward"])
    full = float(case["speedup_vs_baseline_raw_full_step"])
    saved_ratio = float(case["saved_bytes"]) / max(float(case["input_bytes"]), 1e-9)
    penalty = 1.0 + 0.05 * saved_ratio
    score = min(backward, full) / penalty
    info = {
        "status": "ok",
        "dims": case["dims"],
        "dtype": case["dtype"],
        "score": score,
        "backward_speedup": backward,
        "full_step_speedup": full,
        "backward_ms": float(case["backward_from_saved_ms"]),
        "raw_full_step_ms": float(case["raw_forward_backward_full_step_ms"]),
        "baseline_backward_ms": float(case["baseline_backward_ms"]),
        "baseline_full_step_ms": float(case["baseline_raw_full_step_ms"]),
        "saved_memory_ratio": saved_ratio,
        "saved_tensors": case.get("saved_tensors", []),
    }
    return score, info


def run(request: dict) -> dict:
    shapes = request["shapes"]
    shape_ids = [shape["id"] for shape in shapes]
    if not torch.cuda.is_available():
        return _failure(shape_ids, "runtime_unavailable", "CUDA is not available")

    with tempfile.TemporaryDirectory(prefix="evograd_gepa_candidate_") as temporary:
        candidate_path = Path(temporary) / "candidate.py"
        candidate_path.write_text(request["source"], encoding="utf-8")
        try:
            op = get_op(request["op"])
            module = _load_module(str(candidate_path))
            lookup_pair(op, module)
        except Exception as exc:
            return _failure(
                shape_ids,
                "import_or_api_error",
                f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=8),
            )

        if request.get("run_gate", True):
            try:
                correctness = _run_correctness(op, module)
                if correctness["passed"] != correctness["total"]:
                    return _failure(
                        shape_ids,
                        "wrong_answer",
                        f"passed {correctness['passed']}/{correctness['total']} correctness cases",
                        correctness=correctness,
                    )
                smoke_failure = _smoke_benchmark_shapes(op, module, op.coverage)
                if smoke_failure is not None:
                    return _failure(
                        shape_ids,
                        "coverage_failure",
                        smoke_failure.get("error_message", "coverage smoke failed"),
                        failure=smoke_failure,
                    )
            except Exception as exc:
                return _failure(
                    shape_ids,
                    "gate_exception",
                    f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=8),
                )

        workloads = tuple(
            Workload(dims=dict(shape["dims"]), dtype=shape["dtype"]) for shape in shapes
        )
        try:
            benchmark = run_benchmarks(
                op,
                module,
                warmup=int(request["warmup"]),
                reps=int(request["reps"]),
                workloads=workloads,
                performance_baseline=request["baseline"],
            )
        except Exception as exc:
            return _failure(
                shape_ids,
                "benchmark_error",
                f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=8),
            )

    results = {}
    for shape, case in zip(shapes, benchmark["cases"], strict=True):
        score, info = _score_case(case)
        info["shape_id"] = shape["id"]
        results[shape["id"]] = {"score": score, "info": info}
    return {"gate_ok": True, "shapes": results}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} REQUEST_JSON RESULT_JSON", file=sys.stderr)
        return 2
    request_path, result_path = map(Path, argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = run(request)
    except Exception as exc:
        shape_ids = [shape["id"] for shape in request.get("shapes", [])]
        result = _failure(
            shape_ids,
            "worker_exception",
            f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
