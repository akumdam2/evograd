"""Subprocess-friendly candidate verification, used by the seed pipelines.

Runs :func:`evograd.opdecl.verify.verify` on a candidate module and prints a
JSON report whose ``metrics.correct`` field matches what the pipelines'
retry loops expect (1.0 = every case passed). Run in a subprocess so CUDA
crashes in a candidate cannot take the pipeline down:

    python -m evograd.opdecl.verify_cli --op layernorm path/to/program.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(f"evograd_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument(
        "--declaration",
        default=None,
        help="external declaration as path.py:op",
    )
    parser.add_argument(
        "--forward",
        default=None,
        help="override the declaration's forward reference for the oracle",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("float32", "float16", "bfloat16"),
        help="verify only this declared dtype (repeatable)",
    )
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)

    from evograd.opdecl.verify import verify
    from evograd.evolve.evaluator import (
        _capture_native_output,
        _native_output_tail,
    )
    from evograd.ops import get_op, load_op

    try:
        op = load_op(args.declaration) if args.declaration else get_op(args.op)
        if op.name != args.op:
            raise ValueError(
                f"declaration name {op.name!r} does not match --op {args.op!r}"
            )
        if args.forward:
            op = replace(op, forward=args.forward)
            op.validate()
        workloads = op.correctness
        if args.dtype:
            selected = set(args.dtype)
            workloads = tuple(w for w in workloads if w.dtype in selected)
            if not workloads:
                raise ValueError(f"{op.name}: no correctness workloads for {sorted(selected)}")
        native_path = None
        native = None
        try:
            with _capture_native_output() as captured:
                native_path = captured
                report = verify(
                    op,
                    load_candidate(args.candidate),
                    device=args.device,
                    workloads=workloads,
                )
        finally:
            native = _native_output_tail(native_path)
        payload = {
            "metrics": {"correct": 1.0 if report.ok else 0.0},
            "verify": report.to_dict(),
        }
        if native and native.get("bytes", 0) > 0:
            payload["native_output"] = native
    except Exception:
        payload = {
            "metrics": {"correct": 0.0},
            "error": traceback.format_exc(limit=12),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["metrics"]["correct"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
