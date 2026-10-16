"""Which reusable primitives OLMoE executes, and where they go.

The contracts belong to :mod:`evograd.ops.level1`. What is here is this model's
half: which primitives its canonical step runs (so their observed cases are
bound under ``olmoe_1b_7b_observed``), and which Level-2 site each composes into.
No new primitive: the attention projections, the norms, RoPE, causal SDPA, the
SwiGLU activation and the loss are all contracts the dense decoders already
bind. The two places a routed MoE differs are what is *not* here: the expert
GEMMs are three-dimensional parameters driven by a grouped GEMM, not
``nn.Linear`` modules, and the router's softmax and top-k have no primitive --
both are represented at Level 2 only (``olmoe_moe_experts``, with the router
native).

``swiglu``'s observed case is the experts' activation: under the grouped GEMM
one call per layer on the packed ``[tokens * top_k, intermediate]`` tensor.
``rmsnorm``'s observed cases include the full-width query and key norms, one
row per token over the projection width.

The observed shapes, roles and frequencies live in the tracked whole-model
harvest snapshot, not here. Until that snapshot exists the bindings contribute
nothing (:func:`evograd.benchmark.cases.observed_workloads_if_harvested`), and no
declaration needs editing when it appears.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evograd.benchmark.cases import ObservedBinding

WORKLOAD = "olmoe_1b_7b"
SUITE = "olmoe_1b_7b_observed"

#: The primitives the canonical step runs, all bound under one suite name.
OBSERVED_BINDINGS: tuple[ObservedBinding, ...] = tuple(
    ObservedBinding(WORKLOAD, task, SUITE)
    for task in (
        "causal_gqa_attention",
        "cross_entropy",
        "linear_no_bias",
        "rmsnorm",
        "rope",
        "swiglu",
    )
)

#: Primitive -> the Level-2 contracts it composes into.
COMPOSES_INTO: dict[str, tuple[str, ...]] = {
    "linear_no_bias": ("olmoe_qkv_norm_rope", "olmoe_attention"),
    "rmsnorm": ("olmoe_qkv_norm_rope", "olmoe_residual_rmsnorm"),
    "rope": ("olmoe_qkv_norm_rope",),
    "causal_gqa_attention": ("olmoe_attention",),
    "swiglu": ("olmoe_moe_experts",),
    # The loss, and the lm_head before it: no lm_head_ce site in this step.
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
