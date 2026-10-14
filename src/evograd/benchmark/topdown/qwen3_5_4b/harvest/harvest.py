"""Harvest the canonical Qwen3.5 training step into a workload manifest.

    PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest \
        --out results/experiments/qwen3_5_4b_cases/<run>/harvest.json

The Level-3 smoke run with the observer attached: same build, inputs, step and
gradient validation. Every layer is then held to its block kind's mandatory
boundaries, and the per-kind counts to the layer types. A failure writes
nothing.
"""

from __future__ import annotations

from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """Execute the canonical Qwen3.5 step under observation and return the manifest."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
