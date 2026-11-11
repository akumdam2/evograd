"""References for the gated full-attention module, input to output.

Shared with every hybrid model (:mod:`....common.hybrid.attention_block`):
projections, ``(1 + w)`` q/k norms, partial RoPE, causal GQA SDPA,
``sigmoid(gate)`` and o_proj. Production reaches
``modeling_qwen3_5.apply_rotary_pos_emb``; the oracle keeps every step in
float32. cos/sin are inputs: Qwen3.5's interleaved MRoPE on text positions
differs from a 1D partial RoPE by ~6e-8 in float32, so a site must never
recompute them.
"""

from __future__ import annotations

from .....common.hybrid import attention_block as _shared
from .. import hf_modules

attention_block_forward_ref = _shared.forward_ref


def attention_block_forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                       k_norm_weight, o_weight, cos, sin, eps=1e-6):
    return _shared.forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                      k_norm_weight, o_weight, cos, sin, eps,
                                      modeling=hf_modules.modeling)


__all__ = ["attention_block_forward_production", "attention_block_forward_ref"]
