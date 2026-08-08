"""CLI for Pipeline D: Inductor-generated autograd-pair seed.

    python -m evograd.pipelines.d_inductor.cli \\
        --op rmsnorm --output-dir /tmp/D_rmsnorm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.d_inductor.synthesize import (
    InductorSeedConfig,
    synthesize_inductor_seed,
)
from evograd.pipelines.shared.cli_common import add_op_arg, resolve_op


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline D — seed from PyTorch Inductor's own Triton kernels",
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
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to capture on. CUDA emits Triton; CPU emits C++ (plumbing test only).",
    )
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help=(
            "Capture with static shapes. By default AOTAutograd traces with dynamic "
            "shapes so the generated kernels take sizes as runtime arguments and the "
            "seed is valid across the workload grid."
        ),
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "sweep launch configs at first call (default). --no-autotune pins each "
            "kernel to the config autotuning chose during capture, giving a "
            "deterministic seed whose block sizes are explicit constants"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=120,
        help="kill a seed verification subprocess after this many seconds",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="emit the seed without checking it against the oracle",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    op, forward, _example_input = resolve_op(args)
    return synthesize_inductor_seed(
        InductorSeedConfig(
            op=op,
            forward=forward,
            output_dir=Path(args.output_dir).resolve(),
            dtypes=tuple(
                args.dtype
                or dict.fromkeys(workload.dtype for workload in op.correctness)
            ),
            python=args.python,
            device=args.device,
            dynamic_shapes=not args.static_shapes,
            autotune=args.autotune,
            eval_timeout=args.eval_timeout,
            skip_verify=args.skip_verify,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
