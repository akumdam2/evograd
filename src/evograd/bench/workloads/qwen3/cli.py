"""Run the canonical Qwen3-0.6B training smoke and write its JSON report.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3 \
        --out results/qwen3-smoke/canonical.json

With no arguments the canonical workload runs: Qwen3-0.6B, batch 2, sequence
2048, BF16, CUDA, SDPA, ``model.train()``, no cache, no gradient checkpointing.

Every option below shrinks or shifts the workload for debugging. Any of them
makes the run non-canonical: the report says so, the id changes, and a banner is
printed to stderr. Nothing here can turn the cache or gradient checkpointing on
-- those are refused by the spec, not by this parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_override_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The debug overrides, shared with the harvest entry point so the two
    commands cannot drift into accepting different workloads."""
    parser.add_argument("--device", default=None, help="debug override (canonical: cuda)")
    parser.add_argument("--batch-size", type=int, default=None, help="debug override (canonical: 2)")
    parser.add_argument("--seq-len", type=int, default=None, help="debug override (canonical: 2048)")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="debug override (canonical: bfloat16)",
    )
    parser.add_argument(
        "--attn",
        dest="attn_implementation",
        choices=("sdpa", "eager"),
        default=None,
        help="debug override (canonical: sdpa)",
    )
    parser.add_argument("--seed", type=int, default=None, help="debug override (canonical: 0)")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="debug override: shrink num_hidden_layers (canonical: 28)",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    add_override_arguments(parser)
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="print the resolved workload id and hashes, then exit without running",
    )
    return parser


def resolve_spec(args: argparse.Namespace):
    from .levels.level4.spec import CANONICAL

    overrides = {
        key: getattr(args, key)
        for key in ("device", "batch_size", "seq_len", "dtype", "attn_implementation", "seed")
        if getattr(args, key) is not None
    }
    if args.layers is not None:
        overrides["arch"] = {"num_hidden_layers": args.layers}
    return CANONICAL.replace(**overrides) if overrides else CANONICAL


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .levels.level4.smoke import run_smoke
    from .levels.level4.spec import WorkloadSpecError

    try:
        spec = resolve_spec(args)
    except WorkloadSpecError as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 2

    if not spec.is_canonical:
        print(
            "WARNING: non-canonical workload -- this run is a debug variant and its "
            f"numbers must not be reported as the canonical result.\n"
            f"         canonical: {spec.__class__().workload_id}\n"
            f"         this run:  {spec.workload_id}",
            file=sys.stderr,
        )

    if args.print_spec:
        print(f"workload_id   {spec.workload_id}")
        print(f"workload_hash {spec.workload_hash}")
        print(f"config_hash   {spec.config_hash}")
        print(f"canonical     {spec.is_canonical}")
        return 0

    report = run_smoke(spec)
    print(report.summary())
    if args.out is not None:
        path = report.write(args.out)
        print(f"wrote {path}")
    else:
        print(report.to_json())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
