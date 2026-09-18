"""Llama's wiring into the shared training loop: where its batches come from.

The loop itself is :mod:`evograd.evaluation.tier3.gate.training` -- the same one
Qwen3 uses. Only two things are Llama's own here, and both are consequences of
this workload's identity rather than choices made for convenience:

**The tokens are synthetic.** This workload is randomly initialized and its data
is a deterministic pseudo-random token stream (``WorkloadSpec.data``), not text.
A loss curve over it says whether a patched model *executes and optimizes* like
the unpatched one; it says nothing about language-model quality, and its
absolute value is not comparable with a real-text run's. Every result this
module produces carries that label so a report cannot lose it.

**Train and validation batches are disjoint by construction.** Both are drawn
from the same generator, separated by a wide gap in the seed space: training
uses ``data_seed + 1000 + step`` (the offset the existing trajectory check
already uses, so a 200-step run is a longer version of the same experiment) and
validation ``data_seed + 900000 + index``. Validation batches are fixed: the
same tokens in the same order for every provider and every checkpoint.
"""

from __future__ import annotations

from typing import Any

#: Where each stream starts, relative to the workload's ``data_seed``.
TRAIN_SEED_OFFSET = 1000
VALIDATION_SEED_OFFSET = 900_000
#: Held-out batches per evaluation. Eight x (2 x 2048) is 32,768 tokens: enough
#: for a stable mean at this batch size without making validation dominate a
#: 200-step run's wall time.
VALIDATION_BATCHES = 8


class SyntheticBatches:
    """A fixed, positionally indexed stream of deterministic synthetic batches.

    The interface the shared loop expects: ``batch(index)`` and ``describe()``.
    Indexed rather than iterated, so validation cannot advance the training
    stream and re-reading a batch gives the same tokens.
    """

    def __init__(self, workload, *, offset: int, count: int, role: str):
        self._workload = workload
        self.offset = int(offset)
        self.count = int(count)
        self.role = role

    @property
    def batches_per_epoch(self) -> int:
        return self.count

    def batch(self, index: int):
        if index < 0:
            raise IndexError(f"batch index {index} is negative")
        return self._workload.batch_for(seed=self.offset + int(index))

    def describe(self) -> dict[str, Any]:
        return {
            "source": "deterministic synthetic tokens (this workload's own generator)",
            "role": self.role,
            "seed_offset": self.offset,
            "batches": self.count,
            "batch_size": self._workload.spec.batch_size,
            "seq_len": self._workload.spec.seq_len,
            "workload_id": self._workload.spec.workload_id,
            "generator_seed_base": self._workload.spec.seed,
            "note": ("synthetic tokens: an execution and numerical diagnostic, not "
                     "evidence of language-model quality, and not comparable with a "
                     "real-text loss"),
        }


def train_batches(workload, steps: int) -> SyntheticBatches:
    return SyntheticBatches(workload, offset=workload.data_seed + TRAIN_SEED_OFFSET,
                            count=steps, role="training")


def validation_batches(workload, count: int = VALIDATION_BATCHES) -> SyntheticBatches:
    """Held out from training by construction: a disjoint region of the seed space."""
    return SyntheticBatches(workload, offset=workload.data_seed + VALIDATION_SEED_OFFSET,
                            count=count, role="validation (disjoint synthetic batches)")


__all__ = ["SyntheticBatches", "TRAIN_SEED_OFFSET", "VALIDATION_BATCHES",
           "VALIDATION_SEED_OFFSET", "train_batches", "validation_batches"]
