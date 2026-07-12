"""CLI for Pipeline B: LLM-free hand-written dispatch autograd-pair seed.

    python -m evograd.pipelines.b_dispatch.cli \\
        --op rmsnorm --output-dir /tmp/B_rmsnorm --dtype float32 --dtype float16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.b_dispatch.synthesize import (
    HandwrittenDispatchConfig,
    synthesize_handwritten_dispatch,
)
from evograd.pipelines.shared.cli_common import add_op_arg, resolve_op


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline B — deterministic AtenIR seed via hand-written dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_op_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dtype",
        action="append",
        default=None,
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Data type(s) to verify (repeatable; default: every declared correctness dtype)",
    )
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--fp16-atol", type=float, default=5e-2)
    parser.add_argument("--fp16-rtol", type=float, default=5e-2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--no-autograd-pair-seed",
        action="store_true",
        help="Emit only dispatch_program.py, skip the autograd-pair wrapper.",
    )
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help=(
            "Extract with static shapes (legacy behavior). By default the AtenIR "
            "graph is extracted with symbolic shapes so the generated program is "
            "correct at any input shape."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    op, forward, example_input = resolve_op(args)
    return synthesize_handwritten_dispatch(
        HandwrittenDispatchConfig(
            op=op,
            forward=forward,
            example_input=example_input,
            output_dir=Path(args.output_dir).resolve(),
            dtypes=tuple(
                args.dtype
                or dict.fromkeys(workload.dtype for workload in op.correctness)
            ),
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            python=args.python,
            emit_autograd_pair_seed=not args.no_autograd_pair_seed,
            dynamic_shapes=not args.static_shapes,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
