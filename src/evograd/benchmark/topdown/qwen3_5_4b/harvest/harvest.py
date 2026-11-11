"""Harvest the Qwen3.5 training step the capture runs into a workload manifest.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
        python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest \
        --out results/experiments/qwen3_5_4b_cases/<run>/harvest.json

The step is the real-payload capture's first one, built by the verification
integration (:mod:`evograd.verification.integrations.qwen3_5_4b`) under the
verification settings: the pinned ``Qwen/Qwen3.5-4B`` checkpoint on the first
WikiText-2 TRAIN batch of data order 101, B1 x T2048, observed with the same
observer, checks and gradient validation as before. Every layer is then held
to its block kind's mandatory boundaries, and the per-kind counts to the layer
types. The manifest records the run (``run``), and a capture accepts only a
snapshot derived from it. ``--synthetic`` harvests the random-weight,
synthetic-token canonical spec instead (the debug overrides apply to it); so
do the tiny CPU tests, through :func:`run_harvest`. A failure writes nothing.
"""

from __future__ import annotations

from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """The random-weight, synthetic-token Qwen3.5 step under observation."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def harvest_real_run(device: str = "cuda") -> dict[str, Any]:
    """The capture's first step (pinned checkpoint, order-101 batch) under observation."""
    from evograd.verification.integrations.base import real_run
    from evograd.verification.integrations.qwen3_5_4b import MODEL

    from ..declaration import WORKLOAD

    workload, net, batch, identity = real_run(MODEL, device=device)
    return _common.run_harvest(WORKLOAD, workload.spec, model=net, batch=batch, identity=identity)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__, real_run=harvest_real_run)


if __name__ == "__main__":
    raise SystemExit(main())
