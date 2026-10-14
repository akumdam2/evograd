"""Which classes and functions implement Qwen3.5's boundaries.

The observer is shared (:mod:`...common.observe`). What is Qwen3.5's is the
plan: the decoder boundaries -- two token mixers, a dense MLP, the ``(1 + w)``
RMSNorm and the gated RMSNorm -- and the module-level functions its operators
live behind in ``modeling_qwen3_5``:

    torch_chunk_gated_delta_rule   the gated delta rule
    causal_conv1d_fn               the GDN short convolution
    l2norm                         q/k normalization inside the delta rule
    F.softplus                     inside ``g = -exp(A_log) * softplus(a + dt_bias)``,
                                   the only softplus a training step calls
    apply_rotary_pos_emb, F.scaled_dot_product_attention and the causal-LM
    loss                           the standard three, installed for every plan

Mandatory boundaries are per block kind: RoPE and SDPA run in a full-attention
layer only and the delta rule in a GDN layer only, so every one of the 32 layers
is held to its own list (:data:`BLOCK_TASKS`, :func:`check_layers`).
"""

from __future__ import annotations

from typing import Any

from ...common.observe import (
    FunctionBoundary,
    ModuleBoundary,
    ObservationPlan,
    check_layer_boundaries,
    decoder_boundaries,
)

MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"


def _gdn_attrs(module) -> dict[str, Any]:
    return {
        "num_k_heads": int(module.num_k_heads),
        "num_v_heads": int(module.num_v_heads),
        "head_k_dim": int(module.head_k_dim),
        "head_v_dim": int(module.head_v_dim),
        "conv_kernel_size": int(module.conv_kernel_size),
        "activation": str(module.activation),
        "layer_type": str(module.layer_type),
    }


def _gated_norm_attrs(module) -> dict[str, Any]:
    return {
        "normalized_size": int(module.weight.shape[-1]),
        "eps": float(module.variance_epsilon),
        "activation": str(module.activation),
    }


MODULE_BOUNDARIES: tuple[ModuleBoundary, ...] = decoder_boundaries(
    decoder_layer="Qwen3_5DecoderLayer",
    attention="Qwen3_5Attention",
    mlp="Qwen3_5MLP",
    rms_norm="Qwen3_5RMSNorm",
    rotary_embedding="Qwen3_5TextRotaryEmbedding",
    extra=(
        ModuleBoundary("linear_attention", "Qwen3_5GatedDeltaNet", _gdn_attrs),
        ModuleBoundary("rms_norm_gated", "Qwen3_5RMSNormGated", _gated_norm_attrs),
    ),
)

#: Tasks every layer must produce events for, then each block kind's own.
COMMON_TASKS = ("decoder_layer", "mlp", "linear", "rms_norm", "silu")
BLOCK_TASKS: dict[str, tuple[str, ...]] = {
    "linear_attention": COMMON_TASKS + ("linear_attention", "rms_norm_gated",
                                        "gated_delta_rule", "causal_conv1d", "l2norm",
                                        "gdn_gating"),
    "full_attention": COMMON_TASKS + ("attention", "rope_apply", "sdpa"),
}
#: And outside every layer: the loss (and its flattened cross entropy).
MODEL_TASKS = ("causal_cross_entropy", "cross_entropy")

MANDATORY_TASKS: tuple[str, ...] = tuple(sorted(
    set(BLOCK_TASKS["linear_attention"]) | set(BLOCK_TASKS["full_attention"])
    | set(MODEL_TASKS)))

FUNCTIONS: tuple[FunctionBoundary, ...] = (
    FunctionBoundary(
        "gated_delta_rule", MODELING, "torch_chunk_gated_delta_rule",
        "chunk_gated_delta_rule", tensor_kwargs=("g", "beta"),
        attrs=lambda args, kwargs: {
            "use_qk_l2norm_in_kernel": bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
            "initial_state_provided": kwargs.get("initial_state") is not None,
            "output_final_state": bool(kwargs.get("output_final_state", False)),
            "chunk_size": int(kwargs.get("chunk_size", 64)),
        },
    ),
    FunctionBoundary(
        "causal_conv1d", MODELING, "causal_conv1d_fn", "causal_conv1d_fn",
        attrs=lambda args, kwargs: {
            "activation": kwargs.get("activation"),
            "bias": (args[2] if len(args) > 2 else kwargs.get("bias")) is not None,
        },
    ),
    FunctionBoundary(
        "l2norm", MODELING, "l2norm", "l2norm",
        attrs=lambda args, kwargs: {"dim": int(kwargs.get("dim", -1)),
                                    "eps": float(kwargs.get("eps", 1e-6))},
    ),
    FunctionBoundary(
        "gdn_gating", "torch.nn.functional", "softplus", "gdn_gating",
        attrs=lambda args, kwargs: {"beta": float(kwargs.get("beta", 1.0)),
                                    "threshold": float(kwargs.get("threshold", 20.0))},
    ),
)

PLAN = ObservationPlan(
    name="qwen3_5_4b",
    boundaries=MODULE_BOUNDARIES,
    modeling_module=MODELING,
    mandatory_tasks=MANDATORY_TASKS,
    functions=FUNCTIONS,
)


def check_layers(observation, spec) -> None:
    """Every layer held to its block kind's list, and the per-kind counts to the config.

    The counts are what the harvest *observed*; holding them to the layer types
    is a consistency check, not where they come from.
    """
    from ..architecture import layer_types

    types = layer_types(spec.arch)
    for index, kind in enumerate(types):
        check_layer_boundaries(observation, index, BLOCK_TASKS[kind])
    counts = observation.counts_by_task()
    expected = {"decoder_layer": len(types),
                "gated_delta_rule": types.count("linear_attention"),
                "linear_attention": types.count("linear_attention"),
                "sdpa": types.count("full_attention"),
                "attention": types.count("full_attention"),
                "mlp": len(types)}
    wrong = {task: (counts.get(task, 0), want) for task, want in expected.items()
             if counts.get(task, 0) != want}
    if wrong:
        from ...common.observe import MandatoryBoundaryError

        raise MandatoryBoundaryError(f"observed per-task counts disagree with the layer types "
                                     f"(observed, expected): {wrong}")


__all__ = ["BLOCK_TASKS", "COMMON_TASKS", "FUNCTIONS", "MANDATORY_TASKS", "MODELING",
           "MODEL_TASKS", "MODULE_BOUNDARIES", "PLAN", "check_layers"]
