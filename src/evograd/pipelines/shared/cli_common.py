"""Shared CLI argument handling for the seed pipelines.

Every pipeline targets an operator by name (``--op``); the forward reference
and extraction example-input are derived from the declaration and only exist
as flags for overrides (e.g. pointing at an out-of-repo forward)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from evograd.opdecl.activity import OpDecl, example_input_spec
from evograd.ops import OPS, get_op, load_op


def add_op_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--op",
        required=True,
        help=f"target operator (built-ins: {', '.join(sorted(OPS))})",
    )
    parser.add_argument(
        "--declaration",
        default=None,
        help="external declaration as path.py:op",
    )
    parser.add_argument(
        "--forward",
        default=None,
        help="override the declaration's forward reference (module.path:callable)",
    )
    parser.add_argument(
        "--example-input",
        default=None,
        help="override the derived AtenIR extraction input spec",
    )


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")


def add_exec_args(parser: argparse.ArgumentParser) -> None:
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
        help="accept the first attempt without oracle verification (no-CUDA machines)",
    )


def resolve_op(args: argparse.Namespace) -> tuple[OpDecl, str, str]:
    """Return (op, forward, example_input) with derived defaults applied."""
    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    if op.name != args.op:
        raise ValueError(
            f"external declaration name {op.name!r} does not match --op {args.op!r}"
        )
    forward = args.forward or op.forward
    if forward != op.forward:
        op = replace(op, forward=forward)
        op.validate()
    example_input = args.example_input or example_input_spec(op)
    return op, forward, example_input
