"""Harvest the Llama-3 training step the capture runs into a workload manifest.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
        python -m evograd.benchmark.topdown.llama3_2_1b.harvest.harvest \
        --out results/llama3-level3/harvest.json

The step is the real-payload capture's first one, built by the verification
integration (:mod:`evograd.verification.integrations.llama3_2_1b`) under the
verification settings: the pinned ``meta-llama/Llama-3.2-1B`` checkpoint (gated,
read from ``$EVOGRAD_LLAMA_HF_CACHE``) on the first WikiText-2 TRAIN batch of
data order 101, B2 x T2048, observed with the same observer, checks and
gradient validation as before. The manifest records that run (``run``), and a
capture accepts only a snapshot derived from it. ``--synthetic`` harvests the
random-weight, synthetic-token canonical spec instead (the debug overrides
apply to it); so do the tiny CPU tests, through :func:`run_harvest`.

Unlike the smoke run, a failure here does not produce a file: a manifest missing
a boundary is worse than no manifest, because everything derived from it would
inherit the gap silently.

**Memory.** Llama-3.2-1B in BF16 is ~2.5 GiB of weights and as much again in
gradients before activations, and the Level-3 step takes no optimizer step. It
fits any current card at full depth and full sequence, which is the reason this
workload moved off Meta-Llama-3-8B: the 8B needed ``--layers`` to harvest at
all, and a reduced-depth harvest describes shapes the canonical run never had.
"""

from __future__ import annotations

from typing import Any

from ...common import cli as _cli
from ...common import harvest as _common
from ...common.spec import WorkloadSpec


def run_harvest(spec: WorkloadSpec | None = None) -> dict[str, Any]:
    """The random-weight, synthetic-token Llama-3 step under observation."""
    from ..declaration import WORKLOAD

    return _common.run_harvest(WORKLOAD, spec)


def harvest_real_run(device: str = "cuda") -> dict[str, Any]:
    """The capture's first step (pinned checkpoint, order-101 batch) under observation."""
    from evograd.verification.integrations.base import real_run
    from evograd.verification.integrations.llama3_2_1b import MODEL

    from ..declaration import WORKLOAD

    workload, net, batch, identity = real_run(MODEL, device=device)
    return _common.run_harvest(WORKLOAD, workload.spec, model=net, batch=batch, identity=identity)


def main(argv: list[str] | None = None) -> int:
    from ..declaration import WORKLOAD

    return _cli.harvest_main(WORKLOAD, argv, description=__doc__, real_run=harvest_real_run)


if __name__ == "__main__":
    raise SystemExit(main())
