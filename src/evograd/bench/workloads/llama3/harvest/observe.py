"""Which classes implement Llama-3's boundaries, and where its RoPE lives.

The observer itself -- the hooks, the ordinals, the event records, the rule that
no tensor outlives a hook -- is shared; see :mod:`...common.observe`. This is
Llama's :class:`ObservationPlan`: five class names and one module path.

``LlamaRotaryEmbedding`` is watched but not mandatory. It caches cos/sin tables
rather than being a boundary a kernel replaces, and a Transformers release that
folds it elsewhere should downgrade the record rather than fail the harvest --
unlike ``apply_rotary_pos_emb``, whose disappearance would silently cost every
RoPE invocation.

Note what is *absent* relative to Qwen3: there is no per-head q/k normalization,
so the ``rms_norm`` boundary sees only ``input_layernorm``,
``post_attention_layernorm`` and the final ``model.norm``.
"""

from __future__ import annotations

from ...common.observe import (  # noqa: F401  (re-export)
    MANDATORY_DECODER_TASKS,
    Event,
    MandatoryBoundaryError,
    ModuleBoundary,
    Observation,
    ObservationPlan,
    ObserverError,
    TensorMeta,
    decoder_boundaries,
    describe,
    describe_all,
    layer_index_of,
    parameter_meta,
)
from ...common import observe as _common

#: Llama-3's boundary table.
MODULE_BOUNDARIES: tuple[ModuleBoundary, ...] = decoder_boundaries(
    decoder_layer="LlamaDecoderLayer",
    attention="LlamaAttention",
    mlp="LlamaMLP",
    rms_norm="LlamaRMSNorm",
    rotary_embedding="LlamaRotaryEmbedding",
)

#: Task names that must be present in a finished manifest.
MANDATORY_TASKS: tuple[str, ...] = MANDATORY_DECODER_TASKS

PLAN = ObservationPlan(
    name="llama_3_8b",
    boundaries=MODULE_BOUNDARIES,
    modeling_module="transformers.models.llama.modeling_llama",
    mandatory_tasks=MANDATORY_TASKS,
)


def observe(model, *, workload_id: str, config_hash: str, plan: ObservationPlan = PLAN):
    """Watch a Llama-3 model for the duration of the block."""
    return _common.observe(
        model, workload_id=workload_id, config_hash=config_hash, plan=plan
    )


def check_mandatory_boundaries(observation, plan: ObservationPlan = PLAN) -> None:
    """Fail rather than export a manifest missing one of Llama-3's boundaries."""
    _common.check_mandatory_boundaries(observation, plan)
