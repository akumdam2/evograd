"""Correctness-only evaluation of one packaged artifact, in a child process.

The same gates, in the same order, as ``evograd.evolve.evaluator`` applies
before it times anything: the pair API, the static primitive rules, every
declared output and gradient against the autograd oracle, and a run at each
coverage shape. Writes ``{"metrics", "artifacts"}`` to ``$EVOGRAD_RESULT_JSON``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path


def _check(op, program: Path) -> tuple[dict, dict]:
    import torch

    from evograd.evolve.evaluator import (
        _load_module,
        _primitive_usage,
        _run_correctness,
        _smoke_benchmark_shapes,
    )
    from evograd.opdecl.bind import lookup_pair

    failed = {"combined_score": -1e9, "correct": 0.0}
    if not torch.cuda.is_available():
        return failed, {"failure": {"error_type": "RuntimeUnavailable",
                                    "error_message": "CUDA is not available"}}
    try:
        module = _load_module(str(program))
        lookup_pair(op, module)
        usage, failure = _primitive_usage(str(program), module)
        if failure is not None:
            return failed, {"failure": failure}
    except Exception as exc:
        return failed, {"failure": {"error_type": "ImportOrApiError", "error_message": str(exc),
                                    "traceback": traceback.format_exc(limit=8)}}
    try:
        correctness = _run_correctness(op, module)
    except Exception as exc:
        return failed, {"failure": {"error_type": "CorrectnessError", "error_message": str(exc),
                                    "traceback": traceback.format_exc(limit=8)}}
    if correctness["passed"] != correctness["total"]:
        return ({"combined_score": -1e6 + float(correctness.get("partial_correctness", 0.0)),
                 "correct": 0.0,
                 "partial_correctness": float(correctness.get("partial_correctness", 0.0))},
                {"correctness": correctness})
    smoke = _smoke_benchmark_shapes(op, module, op.coverage)
    if smoke is not None:
        return failed, {"correctness": correctness, "failure": smoke}
    return ({"combined_score": 0.0, "correct": 1.0, "partial_correctness": 1.0},
            {"correctness": correctness, "primitives": usage})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("program", type=Path)
    args = parser.parse_args(argv)

    from evograd.benchmark import load_task

    op = load_task(args.declaration)
    if op.name != args.op:
        raise SystemExit(f"declaration is {op.name!r}, expected {args.op!r}")
    if os.environ.get("EVOGRAD_FORWARD_OVERRIDE"):
        op = replace(op, forward=os.environ["EVOGRAD_FORWARD_OVERRIDE"])
    try:
        metrics, artifacts = _check(op, args.program)
    except Exception as exc:  # the parent must always receive a verdict
        metrics = {"combined_score": -1e9, "correct": 0.0}
        artifacts = {"failure": {"error_type": type(exc).__name__, "error_message": str(exc),
                                 "traceback": traceback.format_exc(limit=8)}}
    payload = {"metrics": metrics, "artifacts": artifacts}
    target = os.environ.get("EVOGRAD_RESULT_JSON")
    if target:
        Path(target).write_text(json.dumps(payload, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if metrics.get("correct") == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
