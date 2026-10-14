"""References for the gated delta rule with in-kernel q/k L2 normalization.

The mathematics is shared with every hybrid model
(:mod:`....common.hybrid.delta_rule`): a float32 per-token recurrence on short
sequences, the float32 torch chunk form above them, and production = the torch
chunk function on the model-dtype inputs, reached through ``modeling_qwen3_5``'s
``__wrapped__`` so an importable FLA cannot replace it. No initial or final state.
"""

from __future__ import annotations

from .....common.hybrid import delta_rule as _shared
from .. import hf_modules

RECURRENT_MAX_T = _shared.RECURRENT_MAX_T
CHUNK_SIZE = _shared.CHUNK_SIZE
recurrent_gated_delta_rule = _shared.recurrent_gated_delta_rule


def gated_delta_rule_forward_ref(q, k, v, g, beta):
    return _shared.forward_ref(q, k, v, g, beta, modeling=hf_modules.modeling)


def gated_delta_rule_forward_production(q, k, v, g, beta):
    """Transformers' torch chunk implementation, as ``Qwen3_5GatedDeltaNet`` calls it."""
    return _shared.forward_production(q, k, v, g, beta, modeling=hf_modules.modeling)


__all__ = ["CHUNK_SIZE", "RECURRENT_MAX_T", "gated_delta_rule_forward_production",
           "gated_delta_rule_forward_ref", "recurrent_gated_delta_rule"]
