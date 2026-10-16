"""References for OLMoE's attention prefix: projections, q/k RMSNorm, RoPE.

The computation ``OlmoeAttention.forward`` runs up to the attention call::

    q = q_norm(q_proj(x))     RMSNorm over the whole QO-wide projection, weight [QO]
    k = k_norm(k_proj(x))     RMSNorm over the whole KVO-wide projection, weight [KVO]
    v = v_proj(x)
    q, k, v -> view [B, T, heads, D] -> transpose -> [B, heads, T, D]
    q, k    -> RoPE over the full head with the model's cos/sin tables

This is the third q/k/v boundary in the repository and it is its own
computation: Qwen3 normalizes per head (weight ``[D]``, rows of ``D``) and
Llama does not normalize at all. The norm here runs *before* the head split,
on one row of ``QO`` per token.

Production is that code with the module calls replaced by their tensor
equivalents and the modeling module's own ``apply_rotary_pos_emb``; the oracle
keeps the projections, norms and rotation in float32 and casts each output once.
``cos``/``sin`` are inputs, never recomputed. ``clip_qkv`` is ``None`` in the
published model and the spec refuses anything else, so no clamp is here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import hf_modules
from ..norm import olmoe_rms_norm, rms_norm_ref32


def _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin):
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
    head_dim = cos.shape[-1]
    if sin.shape != cos.shape or cos.ndim != 3 or cos.shape[0] != 1:
        raise ValueError(f"cos and sin must be [1, T, D], got {tuple(cos.shape)} and "
                         f"{tuple(sin.shape)}")
    if q_weight.shape[0] % head_dim or k_weight.shape[0] % head_dim:
        raise ValueError(f"projection fan-outs {q_weight.shape[0]}/{k_weight.shape[0]} must be "
                         f"multiples of the head dimension {head_dim}")
    n_q, n_kv = q_weight.shape[0] // head_dim, k_weight.shape[0] // head_dim
    if n_q % n_kv:
        raise ValueError(f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
                         f"num_kv_heads ({n_kv})")
    return head_dim


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qkv_norm_rope_forward_ref(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight,
                              cos, sin, eps=1e-5):
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
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
    dtype = x.dtype
    return query.to(dtype), key.to(dtype), value.to(dtype)


def qkv_norm_rope_forward_production(x, q_weight, k_weight, v_weight, q_norm_weight,
                                     k_norm_weight, cos, sin, eps=1e-5):
    """The exact spelling ``OlmoeAttention.forward`` executes up to the attention call.

    ``OlmoeRMSNorm`` casts back to the input dtype before multiplying by the
    weight, and ``apply_rotary_pos_emb`` runs entirely in the model dtype. Both
    are reproduced rather than improved, because this is what is timed.
    """
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
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
    return query_states, key_states, value_states


__all__ = ["qkv_norm_rope_forward_production", "qkv_norm_rope_forward_ref"]
