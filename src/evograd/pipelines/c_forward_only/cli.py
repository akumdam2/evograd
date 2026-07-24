"""CLI for Pipeline C: forward-source-only autograd-pair synthesis ablation.

    python -m evograd.pipelines.c_forward_only.cli \\
        --op rmsnorm --output-dir /tmp/C_rmsnorm --model gpt-5.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.c_forward_only.synthesize import (
    ForwardOnlyConfig,
    synthesize_forward_only,
)
from evograd.pipelines.shared.cli_common import (
    add_exec_args,
    add_llm_args,
    add_op_arg,
    resolve_op,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline C — forward-source-only autograd-pair ablation"
    )
    add_op_arg(parser)
    add_llm_args(parser)
    add_exec_args(parser)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    op, forward, _example_input = resolve_op(args)
    return synthesize_forward_only(
        ForwardOnlyConfig(
            op=op,
            forward=forward,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            python=args.python,
            eval_timeout=args.eval_timeout,
            dry_run=args.dry_run,
            skip_verify=args.skip_verify,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
