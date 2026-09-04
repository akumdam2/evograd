"""The Meta-Llama-3-8B benchmark workload, organized by level and by tier.

One canonical execution -- Llama-3-8B, batch 2, sequence 2048, BF16, CUDA, SDPA,
``model.train()``, ``use_cache=False``, no gradient checkpointing, deterministic
seed -- run as::

    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    loss.backward()

with no optimizer step. Everything in this package derives from that one
execution: :mod:`.harvest` records what it invokes, and :mod:`.levels`
decomposes it.

Weights are randomly initialised from a written-out configuration. Llama-3 is a
gated repository, but nothing here fetches a checkpoint, a config or a
tokenizer, so no Hub token is required.

Transformers is optional. Importing this package never imports it; the failure
appears at
:func:`~evograd.bench.workloads.llama3.levels.level4.model.require_transformers`
with the extra to install.

    PYTHONPATH=src python -m evograd.bench.workloads.llama3 --out report.json

**State.** Level 4 and the harvest are implemented and shared with Qwen3's
machinery. What does not exist yet is anything that must be *derived from a
run*: there is no tracked ``harvest/snapshot.json``, because a snapshot comes
out of a harvest and none has been executed. Levels 3-1 and the tier-3 sites
follow from that snapshot; see ``README.md``.
"""

from .levels.level4.report import SmokeReport
from .levels.level4.spec import (
    CANONICAL,
    LLAMA_3_8B,
    WorkloadSpec,
    WorkloadSpecError,
)

__all__ = [
    "CANONICAL",
    "LLAMA_3_8B",
    "SmokeReport",
    "WorkloadSpec",
    "WorkloadSpecError",
]
