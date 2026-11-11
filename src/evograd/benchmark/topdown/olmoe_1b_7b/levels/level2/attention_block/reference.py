"""References for OLMoE's attention block: ``OlmoeAttention.forward`` whole.

The computation, in the module's order::

    q = q_norm(q_proj(x))     RMSNorm over the whole QO-wide projection, weight [QO]
    k = k_norm(k_proj(x))     RMSNorm over the whole KVO-wide projection, weight [KVO]
    v = v_proj(x)
    q, k, v -> view [B, T, heads, D] -> transpose -> [B, heads, T, D]
    q, k    -> RoPE over the full head with the model's cos/sin tables
    a       = causal SDPA(q, k, v) with scale D^-0.5
    out     = o_proj(a transposed to [B, T, HQ, D] and flattened to [B, T, QO])

The q/k norm runs *before* the head split, on one row of ``QO`` per token
(Qwen3 normalizes per head, Llama not at all). OLMoE-1B-7B is multi-head
attention, 16 query heads over 16 KV heads, and so is the contract: a grouped
call is refused rather than spelled.

Production is the module's code with the module calls replaced by their tensor
equivalents: ``F.linear`` for the projections, ``olmoe_rms_norm`` for
``OlmoeRMSNorm``, the modeling module's own ``apply_rotary_pos_emb``, and the
SDPA call Transformers' ``sdpa_attention_forward`` makes for a mask-free
multi-head call (checked on 5.16.1). The oracle keeps every step in float32 and
casts once. ``cos``/``sin`` are inputs, never recomputed; ``clip_qkv`` is
``None`` in the published model and the spec refuses anything else, so no
clamp is here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import hf_modules
from ..norm import olmoe_rms_norm, rms_norm_ref32


def _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_weight, cos, sin):
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, H], got {tuple(x.shape)}")
    hidden = x.shape[-1]
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight), ("v_weight", v_weight)):
        if weight.ndim != 2 or weight.shape[1] != hidden:
            raise ValueError(f"{name} must be [out, {hidden}], got {tuple(weight.shape)}")
    if k_weight.shape != v_weight.shape:
        raise ValueError("k_weight and v_weight must have the same shape")
    if q_norm_weight.shape != (q_weight.shape[0],):
        raise ValueError(f"q_norm_weight must be [{q_weight.shape[0]}], got "
                         f"{tuple(q_norm_weight.shape)}")
    if k_norm_weight.shape != (k_weight.shape[0],):
        raise ValueError(f"k_norm_weight must be [{k_weight.shape[0]}], got "
                         f"{tuple(k_norm_weight.shape)}")
    if o_weight.shape != (hidden, q_weight.shape[0]):
        raise ValueError(f"o_weight must be [{hidden}, {q_weight.shape[0]}], got "
                         f"{tuple(o_weight.shape)}")
    if sin.shape != cos.shape or cos.ndim != 3 or cos.shape[0] != 1 or cos.shape[1] != x.shape[1]:
        raise ValueError(f"cos and sin must be [1, {x.shape[1]}, D], got {tuple(cos.shape)} and "
                         f"{tuple(sin.shape)}")
    head_dim = cos.shape[-1]
    if q_weight.shape[0] % head_dim or q_weight.shape[0] != k_weight.shape[0]:
        raise ValueError(f"multi-head attention needs QO == KVO, a multiple of the head "
                         f"dimension {head_dim}; got {q_weight.shape[0]} and {k_weight.shape[0]}")
    return head_dim


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def attention_block_forward_ref(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight,
                                o_weight, cos, sin, eps=1e-5):
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_weight,
                      cos, sin)
    batch, tokens = x.shape[0], x.shape[1]
    xf = x.float()
    shape = (batch, tokens, -1, head_dim)

    query = rms_norm_ref32(xf @ q_weight.float().transpose(0, 1), q_norm_weight, eps)
    key = rms_norm_ref32(xf @ k_weight.float().transpose(0, 1), k_norm_weight, eps)
    value = xf @ v_weight.float().transpose(0, 1)
    query = query.view(shape).transpose(1, 2)
    key = key.view(shape).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)

    cos_b, sin_b = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    query = query * cos_b + _rotate_half(query) * sin_b
    key = key * cos_b + _rotate_half(key) * sin_b

    scores = (query @ key.transpose(-2, -1)) * head_dim ** -0.5
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
    weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    attn = (weights @ value).transpose(1, 2).reshape(batch, tokens, -1)
    return (attn @ o_weight.float().transpose(0, 1)).to(x.dtype)


def attention_block_forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                       k_norm_weight, o_weight, cos, sin, eps=1e-5):
    """The exact spelling ``OlmoeAttention.forward`` executes, mask-free and without a cache.

    ``OlmoeRMSNorm`` casts back to the input dtype before multiplying by the
    weight, ``apply_rotary_pos_emb`` runs entirely in the model dtype, and the
    attention output is made contiguous twice. All reproduced rather than
    improved, because this is what is timed and what every gate judges against.
    """
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_weight,
                      cos, sin)
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    query_states = olmoe_rms_norm(F.linear(x, q_weight), q_norm_weight, eps)
    key_states = olmoe_rms_norm(F.linear(x, k_weight), k_norm_weight, eps)
    value_states = F.linear(x, v_weight)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    query_states, key_states = hf_modules.modeling.apply_rotary_pos_emb(
        query_states, key_states, cos, sin)
    # bool(): under symbolic shapes (torch.compile, make_fx) `tokens > 1` is a SymBool,
    # which SDPA's is_causal refuses; Transformers' `... and ...` makes the same bool.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=None, dropout_p=0.0,
        scale=head_dim ** -0.5, is_causal=bool(query_states.shape[2] > 1))
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return F.linear(attn_output, o_weight)


__all__ = ["attention_block_forward_production", "attention_block_forward_ref"]
