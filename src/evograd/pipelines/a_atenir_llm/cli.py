"""CLI for Pipeline A: AtenIR-grounded LLM autograd-pair synthesis.

    python -m evograd.pipelines.a_atenir_llm.cli \\
        --op rmsnorm --output-dir /tmp/A_rmsnorm --model gpt-5.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.pipelines.a_atenir_llm.synthesize import (
    AutogradPairConfig,
    synthesize_autograd_pair,
)
from evograd.pipelines.shared.cli_common import (
    add_exec_args,
    add_llm_args,
    add_op_arg,
    resolve_op,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline A — AtenIR autograd-pair fusion agent")
    add_op_arg(parser)
    add_llm_args(parser)
    add_exec_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lowering-context-file", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    op, forward, example_input = resolve_op(args)
    lowering_context = None
    if args.lowering_context_file:
        lowering_context = Path(args.lowering_context_file).read_text(encoding="utf-8")
    return synthesize_autograd_pair(
        AutogradPairConfig(
            op=op,
            forward=forward,
            example_input=example_input,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            python=args.python,
            lowering_context=lowering_context,
            dry_run=args.dry_run,
            skip_verify=args.skip_verify,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
