"""Run the canonical Qwen3-0.6B training smoke and write its JSON report.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3 \
        --out results/qwen3-smoke/canonical.json

With no arguments the canonical workload runs: Qwen3-0.6B, batch 2, sequence
2048, BF16, CUDA, SDPA, ``model.train()``, no cache, no gradient checkpointing.

Every option shrinks or shifts the workload for debugging. Any of them makes the
run non-canonical: the report says so, the id changes, and a banner is printed to
stderr. Nothing here can turn the cache or gradient checkpointing on -- those are
refused by the spec, not by this parser.
"""

from __future__ import annotations

import argparse

from ..common import cli as _cli


def add_override_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The debug overrides, quoting Qwen3's own canonical values."""
    from .declaration import WORKLOAD

    return _cli.add_override_arguments(parser, WORKLOAD)


def resolve_spec(args: argparse.Namespace):
    from .declaration import WORKLOAD

    return _cli.resolve_spec(args, WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    from .declaration import WORKLOAD

    return _cli.smoke_main(WORKLOAD, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
