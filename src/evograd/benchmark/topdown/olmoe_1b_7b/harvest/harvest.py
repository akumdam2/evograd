"""Harvest the OLMoE training step the capture runs into a workload manifest.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
        python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest \
        --out results/experiments/olmoe_1b_7b_cases/<run>/harvest.json

The step is the real-payload capture's first one, built by the verification
integration (:mod:`evograd.verification.integrations.olmoe_1b_7b`) under the
verification settings: the pinned ``allenai/OLMoE-1B-7B-0924`` checkpoint with
grouped-GEMM experts on the first WikiText-2 TRAIN batch of data order 101,
B2 x T2048, observed with the same observer, checks and gradient validation as
before. Every layer is then held to the decoder block's mandatory boundaries
-- attention, RoPE, SDPA, the MoE block, its router and its experts -- and the
per-task counts to the layer count. The manifest records the run (``run``),
and a capture accepts only a snapshot derived from it. ``--synthetic``
harvests the random-weight, synthetic-token canonical spec instead (the debug
overrides apply to it); so do the tiny CPU tests, through :func:`run_harvest`.
A failure writes nothing.
"""

from __future__ import annotations

from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """The random-weight, synthetic-token OLMoE step under observation."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def harvest_real_run(device: str = "cuda") -> dict[str, Any]:
    """The capture's first step (pinned checkpoint, order-101 batch) under observation."""
    from ..declaration import WORKLOAD

    return _common.run_real_harvest(WORKLOAD, "olmoe_1b_7b", device=device)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__, real_run=harvest_real_run)


if __name__ == "__main__":
    raise SystemExit(main())
