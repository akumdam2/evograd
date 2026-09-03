"""The Qwen3-0.6B benchmark workload, organized by level and by tier.

One canonical execution -- Qwen3-0.6B, batch 2, sequence 2048, BF16, CUDA,
SDPA, ``model.train()``, ``use_cache=False``, no gradient checkpointing,
deterministic seed -- run as::

    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    loss.backward()

with no optimizer step. Everything in this package is derived from that one
execution: :mod:`.harvest` records what it invokes, :mod:`.levels` decomposes it
into level 4/3/2/1 tasks, and :mod:`.evaluation` holds the tier-3 machinery for
swapping an operator into the live model.

Level and tier are independent axes. A level-2 operator can be measured by a
tier-1 pair benchmark or inside a tier-3 model run; the level says what is being
asked of a kernel, the tier says how carefully the answer is checked. See
``README.md`` in this directory.

Transformers is optional. Importing this package never imports it; the failure
appears at
:func:`~evograd.bench.workloads.qwen3.levels.level4.model.require_transformers`
with the extra to install.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3 --out report.json
"""

from .levels.level4.report import SmokeReport
from .levels.level4.spec import (
    CANONICAL,
    QWEN3_0_6B,
    WorkloadSpec,
    WorkloadSpecError,
)

__all__ = [
    "CANONICAL",
    "QWEN3_0_6B",
    "Qwen3Workload",
    "SmokeReport",
    "WorkloadSpec",
    "WorkloadSpecError",
]


def __getattr__(name: str):
    # Deferred: Qwen3Workload reaches the tier-3 site adapters, which import
    # torch. The spec and the report must stay importable without it.
    if name == "Qwen3Workload":
        from .evaluation.tier3.workload import Qwen3Workload

        return Qwen3Workload
    raise AttributeError(name)
