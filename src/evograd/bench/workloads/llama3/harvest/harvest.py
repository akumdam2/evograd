"""Harvest the canonical Llama-3 training step into a workload manifest.

    PYTHONPATH=src python -m evograd.bench.workloads.llama3.harvest.harvest \
        --out results/llama3-level4/harvest.json

This is the Level-4 smoke run with an observer attached. It reuses the same
build, the same inputs, the same step and the same gradient validation -- the
harvest must describe *the* canonical execution, and a second, subtly different
implementation of it would defeat that.

Unlike the smoke run, a failure here does not produce a file: a manifest missing
a boundary is worse than no manifest, because everything derived from it would
inherit the gap silently.

**Memory.** Llama-3-8B in BF16 is ~16 GiB of weights and ~16 GiB of gradients
before activations. The Level-4 step takes no optimizer step, so it fits a
80-120 GiB card; ``--layers`` shrinks it for a smoke.
"""

from __future__ import annotations

from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """Execute the canonical Llama-3 step under observation, return the manifest."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
