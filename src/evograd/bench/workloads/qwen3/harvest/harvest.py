"""Harvest the canonical Qwen3 training step into a workload manifest.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.harvest \
        --out results/qwen3-level4/harvest.json

This is the Level-4 smoke run with an observer attached. It reuses the same
build, the same inputs, the same step and the same gradient validation -- the
point of the milestone is that the harvest describes *the* canonical execution,
and a second, subtly different implementation of it would defeat that.

Unlike the smoke run, a failure here does not produce a file. A smoke report
that says "it failed" is useful; a manifest missing a boundary is worse than no
manifest, because everything derived from it would inherit the gap silently.
"""

from __future__ import annotations

import sys
from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """Execute the canonical Qwen3 step under observation and return the manifest."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
