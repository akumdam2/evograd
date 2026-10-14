"""References for gated attention's prefix: projections, (1 + w) q/k norms, partial RoPE.

Shared with every hybrid model (:mod:`....common.hybrid.qkv_rope`); production
reaches ``modeling_qwen3_5.apply_rotary_pos_emb``. cos/sin are inputs: Qwen3.5's
interleaved MRoPE on text positions differs from a 1D partial RoPE by ~6e-8 in
float32, so a site must never recompute them.
"""

from __future__ import annotations

from .....common.hybrid import qkv_rope as _shared
from .. import hf_modules

qkv_norm_partial_rope_forward_ref = _shared.forward_ref


def qkv_norm_partial_rope_forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                             k_norm_weight, cos, sin, eps=1e-6):
    return _shared.forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                      k_norm_weight, cos, sin, eps,
                                      modeling=hf_modules.modeling)


__all__ = ["qkv_norm_partial_rope_forward_production", "qkv_norm_partial_rope_forward_ref"]
