"""evograd seed unified: deterministic compiler-aware AtenIR synthesis."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from evograd.pipelines.unified.synthesize import UnifiedSeedConfig, select_workload, synthesize_unified_seed
from evograd.pipelines.shared.cli_common import add_op_arg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_op_arg(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--dims", nargs="+", metavar="NAME=SIZE",
                        help="fixed dimensions; default: first declared workload of the selected dtype")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-timeout", type=int, default=180)
    args = parser.parse_args(argv)
    if args.example_input:
        parser.error("unified uses --dims and --dtype to define the exact deployment configuration")
    from evograd.benchmark import get_task, load_task

    op = load_task(args.declaration) if args.declaration else get_task(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    if args.forward:
        op = replace(op, forward=args.forward)
    if args.declaration:
        original_path, _, attribute = args.declaration.partition(":")
        op = replace(op, declaration=f"{Path(original_path).resolve()}:{attribute}")
    try:
        dims = None
        if args.dims:
            dims = {}
            for binding in args.dims:
                name, separator, value = binding.partition("=")
                if not separator or name in dims:
                    raise ValueError(f"invalid or duplicate dimension: {binding}")
                dims[name] = int(value)
        workload = select_workload(op, dtype=args.dtype, dims=dims)
    except ValueError as exc:
        parser.error(str(exc))
    return synthesize_unified_seed(UnifiedSeedConfig(
        op=op, workload=workload, output_dir=args.output_dir,
        python=args.python, device=args.device, eval_timeout=args.eval_timeout,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
