"""Run the canonical OLMoE-1B-7B training smoke and write its JSON report.

    PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b \
        --out results/experiments/olmoe_1b_7b_cases/<run>/smoke.json

With no arguments the canonical workload runs: OLMoE-1B-7B, batch 2, sequence
2048, BF16, CUDA, SDPA, grouped-GEMM experts, ``model.train()``, no cache, no
gradient checkpointing, random weights, synthetic tokens. Nothing is fetched.

Every option shrinks or shifts the workload for debugging and makes the run
non-canonical: the report says so, the id changes, and a banner is printed. The
experts kernel is not a command-line override; a debug variant that runs the
per-expert ``eager`` loop is built in Python
(``CANONICAL.replace(experts_implementation="eager")``).
"""

from __future__ import annotations

import argparse

from ..common import cli as _cli


def add_override_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
