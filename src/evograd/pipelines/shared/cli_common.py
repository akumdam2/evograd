"""Shared CLI argument handling for the seed pipelines.

Every pipeline targets one model case through its task pack (``--pack``, built
by ``adbench-tools build-task``): the pack's declaration is the operator, its
forward the reference, and Gate 1 on its search payloads accepts the seed. The
AtenIR extraction input is derived from the declaration and only exists as a
flag for overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evograd.opdecl.activity import OpDecl, example_input_spec
from evograd.pipelines.shared.primitives import REGISTRY


def add_pack_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="task pack of one model case; its Gate 1 search rule accepts the seed",
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
        help="publish the first attempt unverified (no-CUDA machines); its report records "
             "verification as skipped",
    )


def add_grant_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-primitive",
        action="append",
        default=[],
        choices=sorted(REGISTRY),
        help="grant one trusted Level-1 primitive to the seed (repeatable); the "
             "artifact declares the grant in its deployment layer",
    )


def resolve_op(args: argparse.Namespace) -> tuple[OpDecl, str, Path]:
    """Return (op, example_input, pack root) for ``--pack``."""
    from evograd.adbench_tools.task import load as load_pack

    task = load_pack(args.pack)
    op = task.op()
    return op, args.example_input or example_input_spec(op), task.root
