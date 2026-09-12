"""Which reusable primitives Llama-3-8B executes, and what it does with them.

The contracts belong to :mod:`evograd.ops.level1`, which owns the mathematics
and knows nothing about this model. What is here is the Llama-specific half:
which of those primitives the canonical step runs, which Level-2 site each one
composes into, and -- in :data:`OBSERVED_BINDINGS` -- how this model's observed
cases attach to the shared contracts.

**The harvest has not been run.** This workload's snapshot does not exist yet,
so every observed suite below is empty and the primitive declarations are
unchanged by it. That is deliberate: a workload package exists before its
harvest does, the registry must stay importable in the meantime, and the suites
appear the moment the snapshot is tracked -- with no edit here and none in any
primitive. Nothing in this module fabricates harvested data, and a snapshot
that exists but fails its hash check still raises rather than being treated as
absent.

Deciding whether an implementation of one of these primitives is good enough is
not done here; that is
:mod:`evograd.evaluation.workloads.llama3_2_1b.level1`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evograd.benchmark.cases import ObservedBinding

from ...harvest.snapshot import load as load_snapshot

#: The workload these observations belong to; also the snapshot key and the
#: provenance model key.
WORKLOAD = "llama_3_2_1b"

#: Generic task -> the Level-2 contract its outputs feed, in this model.
#: Llama-3's attention has no per-head query/key RMSNorm, so ``rmsnorm`` feeds
#: only the residual fusion here where in Qwen3 it also feeds the QKV boundary.
COMPOSES_INTO = {
    "linear_no_bias": ("llama3_qkv_rope", "llama3_attention", "llama3_swiglu_mlp"),
    "rmsnorm": ("llama3_residual_rmsnorm",),
    "rope": ("llama3_qkv_rope",),
    "swiglu": ("llama3_swiglu_mlp",),
    "causal_gqa_attention": ("llama3_attention",),
    "cross_entropy": (),
}

#: The reusable primitives this model's captured step runs.
#:
#: Stated once, as the keys of :data:`COMPOSES_INTO`: every primitive Llama-3-8B
#: runs composes into some Level-2 site, and ``cross_entropy``'s empty tuple
#: records that it composes into none of *this model's* fused boundaries rather
#: than that the model does not run it.
OPERATORS: tuple[str, ...] = tuple(COMPOSES_INTO)

#: Which primitives this model binds its observed cases to, and under which
#: suite name. Qwen3-0.6B binds several of the same primitives under
#: ``qwen3_0_6b_observed``; two models on one contract is the ordinary case and
#: the suite name is what keeps the two sets of widths apart.
#:
#: Deliberately *not* mirrored into ``coverage``: Llama-3-8B's observed widths
#: are several times Qwen3-0.6B's, and making every candidate run them would
#: charge a Qwen3-targeted kernel for shapes it does not claim. They are
#: benchmark suites, selectable by name.
OBSERVED_BINDINGS: tuple[ObservedBinding, ...] = (
    ObservedBinding(WORKLOAD, "causal_gqa_attention", "llama_3_2_1b_observed"),
    ObservedBinding(WORKLOAD, "cross_entropy", "llama_3_2_1b_observed",
                    declared_tolerances=True),
    ObservedBinding(WORKLOAD, "linear_no_bias", "llama_3_2_1b_observed"),
    ObservedBinding(WORKLOAD, "rmsnorm", "llama_3_2_1b_observed"),
    ObservedBinding(WORKLOAD, "rope", "llama_3_2_1b_observed"),
    ObservedBinding(WORKLOAD, "swiglu", "llama_3_2_1b_observed",
                    declared_tolerances=True),
)


class Level1Error(RuntimeError):
    """The Level-1 mapping could not be checked."""


def mapping(snapshot_path: Path | None = None) -> dict[str, Any]:
    """What the harvest observed at Level 1, plus this model's composition.

    The same shape Qwen3-0.6B's manifest returns, read from this workload's own
    snapshot. That snapshot has not been produced yet, so this raises the
    harvest reader's "not harvested" error rather than returning an empty
    report -- an unrun harvest and an empty one are different answers, and the
    Level-1 CLI that calls this is what tells the difference.
    """
    payload = load_snapshot(snapshot_path)
    return {
        "snapshot_hash": payload["snapshot_hash"],
        "workload_id": payload["workload_id"],
        "tasks": payload["level1"],
        "composes_into": COMPOSES_INTO,
    }


__all__ = ["COMPOSES_INTO", "Level1Error", "OBSERVED_BINDINGS",
           "OPERATORS", "WORKLOAD", "mapping"]
