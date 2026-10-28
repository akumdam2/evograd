"""References for OLMoE's attention core: causal SDPA, then ``o_proj``.

The oracle is the one every decoder in the repository shares
(:mod:`....common.level2_references`): float32 scores, an explicit causal mask,
the KV heads expanded by hand. Production is the exact branch Transformers'
``sdpa_attention_forward`` takes for ``OlmoeAttention`` with no mask: a
multi-head model (OLMoE's 16/16) passes q, k, v straight through; a grouped one
gets ``enable_gqa`` when SDPA can broadcast the KV heads and ``repeat_kv``
otherwise, exactly as ``use_gqa_in_sdpa`` decides it (checked on 5.16.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .....common.level2_references import (
    attention_projection_forward_ref as attention_forward_ref,
)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, tokens, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, tokens, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, tokens, head_dim)


def attention_forward_production(q, k, v, o_weight):
    """Exactly ``sdpa_attention_forward`` for the mask-free call, then ``o_proj``."""
    batch, heads, tokens, head_dim = q.shape
    groups = heads // k.shape[1]
    key, value, sdpa_kwargs = k, v, {}
    if groups > 1:
        if k.shape[-1] == v.shape[-1] <= 256:
            sdpa_kwargs = {"enable_gqa": True}
        else:
            key, value = _repeat_kv(k, groups), _repeat_kv(v, groups)
    # bool(): under symbolic shapes (torch.compile, make_fx) `tokens > 1` is a SymBool,
    # which SDPA's is_causal refuses; Transformers' `... and ...` makes the same bool.
    attn = F.scaled_dot_product_attention(
        q, key, value, attn_mask=None, dropout_p=0.0, scale=head_dim ** -0.5,
        is_causal=bool(tokens > 1), **sdpa_kwargs,
    )
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.reshape(batch, tokens, -1).contiguous()
    return F.linear(attn, o_weight)


__all__ = ["attention_forward_production", "attention_forward_ref"]
