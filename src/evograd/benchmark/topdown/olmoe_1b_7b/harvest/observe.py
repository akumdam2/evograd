"""Which classes implement OLMoE's boundaries, and where its RoPE lives.

The observer is shared (:mod:`...common.observe`). What is OLMoE's is the plan:
the decoder boundaries -- ``OlmoeAttention``, ``OlmoeRMSNorm`` and, in the MLP
position, a routed mixture of experts rather than a dense MLP -- and which
module holds ``apply_rotary_pos_emb``.

The MLP position is three boundaries, not one. ``OlmoeSparseMoeBlock`` (task
``moe``) holds a router ``OlmoeTopKRouter`` (``router``: one ``[E, H]`` GEMM, a
softmax and a top-k) and the experts ``OlmoeExperts`` (``experts``: every
expert's weights in two 3-D tensors, dispatched through the configured experts
kernel). Nothing is passed as ``mlp``: the shared dense-MLP extractor would read
attributes the MoE block does not have, and the dense ``OlmoeMLP`` class in the
modeling file is never instantiated by the decoder. The experts' SiLU is found
through ``activation_owners`` and recorded under the role ``experts_act_fn``,
so it cannot merge with a dense MLP's ``act_fn``.

Mandatory boundaries are held per layer: every one of the 16 layers must have
produced each decoder task, and the per-task counts must equal the layer count.
The SiLU count is one per layer under the grouped GEMM (one call on the packed
tokens) and at least one per layer under the eager loop (one per expert that
received a token), which is the only count the experts kernel changes.
"""

from __future__ import annotations

from typing import Any

from ...common.observe import (
    DECODER_ATTRS,
    MandatoryBoundaryError,
    ModuleBoundary,
    ObservationPlan,
    check_layer_boundaries,
    decoder_boundaries,
)

MODELING = "transformers.models.olmoe.modeling_olmoe"

MODULE_BOUNDARIES: tuple[ModuleBoundary, ...] = decoder_boundaries(
    decoder_layer="OlmoeDecoderLayer",
    attention="OlmoeAttention",
    mlp=(),
    rms_norm="OlmoeRMSNorm",
    rotary_embedding="OlmoeRotaryEmbedding",
    extra=(
        ModuleBoundary("moe", "OlmoeSparseMoeBlock", DECODER_ATTRS["moe"]),
        ModuleBoundary("router", "OlmoeTopKRouter", DECODER_ATTRS["router"]),
        ModuleBoundary("experts", "OlmoeExperts", DECODER_ATTRS["experts"]),
    ),
)

#: Tasks every decoder layer must produce events for.
LAYER_TASKS: tuple[str, ...] = (
    "decoder_layer", "attention", "rope_apply", "sdpa", "linear", "rms_norm",
    "moe", "router", "experts", "silu",
)
#: And outside every layer: the loss (and its flattened cross entropy).
MODEL_TASKS: tuple[str, ...] = ("causal_cross_entropy", "cross_entropy")

MANDATORY_TASKS: tuple[str, ...] = tuple(sorted(set(LAYER_TASKS) | set(MODEL_TASKS)))

#: Tasks that run exactly once per decoder layer, whatever the experts kernel.
ONCE_PER_LAYER: tuple[str, ...] = (
    "decoder_layer", "attention", "rope_apply", "sdpa", "moe", "router", "experts",
)

PLAN = ObservationPlan(
    name="olmoe_1b_7b",
    boundaries=MODULE_BOUNDARIES,
    modeling_module=MODELING,
    mandatory_tasks=MANDATORY_TASKS,
    activation_owners=("OlmoeExperts",),
)


def check_layers(observation, spec) -> None:
    """Every layer held to the decoder block's list, and the counts to the depth.

    The counts are what the harvest *observed*; holding them to the layer
    count is a consistency check, not where they come from.
    """
    layers = int(spec.arch["num_hidden_layers"])
    for index in range(layers):
        check_layer_boundaries(observation, index, LAYER_TASKS)
    counts = observation.counts_by_task()
    wrong: dict[str, Any] = {}
    for task in ONCE_PER_LAYER:
        if counts.get(task, 0) != layers:
            wrong[task] = (counts.get(task, 0), layers)
    silu = counts.get("silu", 0)
    if getattr(spec, "experts_implementation", "grouped_mm") == "grouped_mm":
        if silu != layers:
            wrong["silu"] = (silu, layers)
    elif silu < layers:
        wrong["silu"] = (silu, f">= {layers}")
    if wrong:
        raise MandatoryBoundaryError(
            f"observed per-task counts disagree with the {layers}-layer architecture "
            f"(observed, expected): {wrong}")


__all__ = ["LAYER_TASKS", "MANDATORY_TASKS", "MODELING", "MODEL_TASKS", "MODULE_BOUNDARIES",
           "ONCE_PER_LAYER", "PLAN", "check_layers"]
