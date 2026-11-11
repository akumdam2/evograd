"""Which reusable primitives Qwen3-0.6B executes, and what it does with them.

The contracts themselves belong to :mod:`evograd.ops.level1`, which owns the
mathematics and knows nothing about this model. What is here is the
Qwen-specific half: which of those primitives the canonical step runs, at which
observed shapes and layouts, in which roles and how often, and which Level-2
site each one composes into.

None of those facts are written down twice. The shapes, dtypes, strides, roles
and frequencies live in the tracked harvest snapshot; this module reads them
and adds the one thing the snapshot cannot know -- the composition, which is a
statement about this benchmark's own level hierarchy rather than about the run.

It also owns :data:`OBSERVED_BINDINGS`: which primitives this model binds its
observed cases to, under which suite name, and in which order they join the
untimed coverage. The mechanics of applying a binding are generic and live in
:mod:`evograd.benchmark.cases`; the choice of what to bind is this model's, and
:mod:`evograd.benchmark.core.registry` is where the two are put together.

Deciding whether an implementation of one of these primitives is *good enough*
is not done here; that is :mod:`evograd.verification`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evograd.benchmark.cases import ObservedBinding

from ...harvest.snapshot import load as load_snapshot

#: The workload these observations belong to. Also the snapshot key and the
#: provenance model key, so it is written once.
WORKLOAD = "qwen3_0_6b"

#: Which reusable primitives this model's captured step runs, and how each
#: one's observed cases join the contract.
#:
#: This is Qwen's decision, not the binder's. The suite name is part of the
#: report contract; ``coverage`` records whether the observed cases precede or
#: follow the coverage the primitive already has, because a report lists cases
#: in declaration order and the two are not interchangeable.
#:
#: :mod:`evograd.benchmark.cases` knows how to apply all of that and nothing
#: about which model it belongs to; the registry hands this table to it.
OBSERVED_BINDINGS: tuple[ObservedBinding, ...] = (
    ObservedBinding(WORKLOAD, "causal_gqa_attention", "qwen3_0_6b_observed"),
    ObservedBinding(WORKLOAD, "cross_entropy", "qwen3_0_6b_observed"),
    ObservedBinding(WORKLOAD, "linear_no_bias", "qwen3_0_6b_observed"),
    ObservedBinding(WORKLOAD, "rmsnorm", "qwen3_0_6b_observed", coverage="prepend"),
    ObservedBinding(WORKLOAD, "rope", "qwen3_0_6b_observed"),
    ObservedBinding(WORKLOAD, "swiglu", "qwen3_0_6b_observed"),
)


#: Generic task -> the Level-2 contract its outputs feed. Checked structurally
#: by the focused tests; recorded here so the composition is stated once.
COMPOSES_INTO = {
    "linear_no_bias": ("qwen3_attention_block", "qwen3_swiglu_mlp", "qwen3_lm_head_ce"),
    "rmsnorm": ("qwen3_attention_block", "fused_add_rms_norm"),
    "rope": ("qwen3_attention_block",),
    "swiglu": ("qwen3_swiglu_mlp",),
    "causal_gqa_attention": ("qwen3_attention_block",),
    "cross_entropy": ("qwen3_lm_head_ce",),
}


class Level1Error(RuntimeError):
    """The Level-1 mapping could not be checked."""


def mapping(snapshot_path: Path | None = None) -> dict[str, Any]:
    payload = load_snapshot(snapshot_path)
    return {
        "snapshot_hash": payload["snapshot_hash"],
        "workload_id": payload["workload_id"],
        "tasks": payload["level1"],
        "composes_into": COMPOSES_INTO,
    }
