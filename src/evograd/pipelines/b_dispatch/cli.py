"""CLI for Pipeline B: LLM-free hand-written dispatch autograd-pair seed.

    python -m evograd.pipelines.b_dispatch.cli \\
        --pack <task pack> --output-dir /tmp/B_rmsnorm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.b_dispatch.synthesize import (
    HandwrittenDispatchConfig,
    synthesize_handwritten_dispatch,
)
from evograd.pipelines.shared.cli_common import (
    add_exec_args,
    add_grant_arg,
    add_pack_arg,
    resolve_op,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline B — deterministic AtenIR seed via hand-written dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_pack_arg(parser)
    add_exec_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help=(
            "Extract with static shapes (legacy behavior). By default the AtenIR "
            "graph is extracted with symbolic shapes so the generated program is "
            "correct at any input shape."
        ),
    )
    # B records a grant in the deployment layer; its kernels stay handwritten.
    add_grant_arg(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    op, example_input, pack = resolve_op(args)
    return synthesize_handwritten_dispatch(
        HandwrittenDispatchConfig(
            op=op,
            forward=op.forward,
            example_input=example_input,
            output_dir=Path(args.output_dir).resolve(),
            python=args.python,
            eval_timeout=args.eval_timeout,
            dynamic_shapes=not args.static_shapes,
            allowed_primitives=tuple(args.allow_primitive),
            skip_verify=args.skip_verify,
            pack=pack,
        )
    )

if __name__ == "__main__":
    raise SystemExit(main())
