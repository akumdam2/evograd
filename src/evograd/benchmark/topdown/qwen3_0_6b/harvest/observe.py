"""Which classes implement Qwen3's boundaries, and where its RoPE lives.

The observer itself -- the hooks, the ordinals, the event records, the rule that
no tensor outlives a hook -- is the same for every decoder-only causal LM and
lives in :mod:`...common.observe`. What is Qwen3's, and all that is here, is the
:class:`ObservationPlan`: six class names and one module path.

``Qwen3RotaryEmbedding`` is watched but not mandatory. It is a cache of cos/sin
tables rather than a boundary a kernel replaces, and a Transformers release that
folds it elsewhere should downgrade the record rather than fail the harvest --
unlike ``apply_rotary_pos_emb``, whose disappearance would silently cost every
RoPE invocation.
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

#: Qwen3's boundary table.
MODULE_BOUNDARIES: tuple[ModuleBoundary, ...] = decoder_boundaries(
    decoder_layer="Qwen3DecoderLayer",
    attention="Qwen3Attention",
    mlp="Qwen3MLP",
    rms_norm="Qwen3RMSNorm",
    rotary_embedding="Qwen3RotaryEmbedding",
)

#: Task names that must be present in a finished manifest.
MANDATORY_TASKS: tuple[str, ...] = MANDATORY_DECODER_TASKS

PLAN = ObservationPlan(
    name="qwen3_0_6b",
    boundaries=MODULE_BOUNDARIES,
    modeling_module="transformers.models.qwen3.modeling_qwen3",
    mandatory_tasks=MANDATORY_TASKS,
)


def observe(model, *, workload_id: str, config_hash: str, plan: ObservationPlan = PLAN):
    """Watch a Qwen3 model for the duration of the block."""
    return _common.observe(
        model, workload_id=workload_id, config_hash=config_hash, plan=plan
    )


def check_mandatory_boundaries(observation, plan: ObservationPlan = PLAN) -> None:
    """Fail rather than export a manifest missing one of Qwen3's boundaries."""
    _common.check_mandatory_boundaries(observation, plan)
