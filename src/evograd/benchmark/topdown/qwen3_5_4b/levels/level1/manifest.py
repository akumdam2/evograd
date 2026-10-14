"""Which reusable primitives Qwen3.5 executes, and where they go.

The contracts belong to :mod:`evograd.ops.level1`. What is here is this model's
half: which primitives its canonical step runs (so their observed cases are
bound under ``qwen3_5_4b_observed``), and which Level-2 site each composes into.
Six of the primitives (causal_conv1d, gated_rmsnorm, gdn_gating, gemma_rmsnorm,
l2norm, partial_rope) are the hybrid linear-attention decoder's own.

The observed shapes, roles and frequencies live in the tracked whole-model
harvest snapshot, not here. Until that snapshot exists the bindings contribute
nothing (:func:`evograd.benchmark.cases.observed_workloads_if_harvested`), and no
declaration needs editing when it appears. Every observed role must map onto one
of these primitives (:data:`...harvest.snapshot.LEVEL1_SOURCES`); an unmapped role
fails the extraction rather than being dropped.

``partial_rope`` cases read theta 1e7 from their provenance (the record's
``rope_theta``); ``cross_entropy`` and its Liger reference run on 248,320-wide
rows -- both are exercised by ``tests/qwen3_5_4b/test_level1_bindings.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evograd.benchmark.cases import ObservedBinding

WORKLOAD = "qwen3_5_4b"
SUITE = "qwen3_5_4b_observed"

#: The primitives the canonical step runs, all bound under one suite name.
OBSERVED_BINDINGS: tuple[ObservedBinding, ...] = tuple(
    ObservedBinding(WORKLOAD, task, SUITE)
    for task in (
        "causal_conv1d",
        "causal_gqa_attention",
        "cross_entropy",
        "gated_rmsnorm",
        "gdn_gating",
        "gemma_rmsnorm",
        "l2norm",
        "linear_no_bias",
        "partial_rope",
        "swiglu",
    )
)

#: Primitive -> the Level-2 contracts it composes into.
COMPOSES_INTO: dict[str, tuple[str, ...]] = {
    "linear_no_bias": ("qwen3_5_gdn_in_proj", "qwen3_5_gdn_norm_out",
                       "qwen3_5_qkv_norm_partial_rope", "qwen3_5_gated_attention",
                       "qwen3_5_swiglu_mlp"),
    "causal_conv1d": ("qwen3_5_gdn_in_proj",),
    "gdn_gating": ("qwen3_5_gdn_in_proj",),
    "l2norm": ("qwen3_5_gated_delta_rule",),
    "gated_rmsnorm": ("qwen3_5_gdn_norm_out",),
    "gemma_rmsnorm": ("qwen3_5_qkv_norm_partial_rope", "qwen3_5_residual_rmsnorm"),
    "partial_rope": ("qwen3_5_qkv_norm_partial_rope",),
    "causal_gqa_attention": ("qwen3_5_gated_attention",),
    "swiglu": ("qwen3_5_swiglu_mlp",),
    # The loss, and the lm_head before it: the lm_head_ce site is deferred, so
    # this composes into nothing declared yet.
    "cross_entropy": (),
}


def mapping(snapshot_path: Path | None = None) -> dict[str, Any]:
    from ...harvest.snapshot import load as load_snapshot

    payload = load_snapshot(snapshot_path)
    return {
        "snapshot_hash": payload["snapshot_hash"],
        "workload_id": payload["workload_id"],
        "tasks": payload["level1"],
        "composes_into": COMPOSES_INTO,
    }


__all__ = ["COMPOSES_INTO", "OBSERVED_BINDINGS", "SUITE", "WORKLOAD", "mapping"]
