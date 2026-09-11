"""Level-1 CLI for Llama-3-8B: the shared commands, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level1_cli as _common
from evograd.evaluation.workloads.llama3_8b.descriptor import WORKLOAD


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
