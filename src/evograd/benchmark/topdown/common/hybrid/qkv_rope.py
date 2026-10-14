"""Gated attention's prefix: projections, ``(1 + w)`` q/k norms, partial RoPE.

The attention forward up to the attention call::

    q|gate = q_proj(x) viewed [B, T, HQ, 2D] and chunked -- per head, each
             head's query followed by its gate (interleaved, not two halves)
    gate   -> [B, T, HQ*D]
    q      -> q_norm (zero-centred, 1 + w) -> [B, HQ, T, D]
    k      -> k_proj -> k_norm               -> [B, HK, T, D]
    v      -> v_proj                         -> [B, HK, T, D]   (a transposed view)
    q, k   -> RoPE on the first R = cos.shape[-1] of D channels

Production is that code with the module calls replaced by their tensor
equivalents and the modeling module's own ``apply_rotary_pos_emb``; the oracle
keeps the projections, norms and rotation in float32 and casts each output once.
cos/sin are inputs, never recomputed: Qwen3.5's interleaved MRoPE and a 1D
partial RoPE differ by ~6e-8 in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.ops.level1.gemma_rmsnorm.forward_ref import gemma_rmsnorm_forward_ref


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope32(t, cos, sin):
    rotary = cos.shape[-1]
    c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    t_rot, t_pass = t[..., :rotary], t[..., rotary:]
    return torch.cat([t_rot * c + _rotate_half(t_rot) * s, t_pass], dim=-1)


def forward_ref(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin,
                eps=1e-6):
    batch, tokens = x.shape[0], x.shape[1]
    head_dim = q_norm_weight.shape[0]
    xf = x.float()

    def norm(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())

    q_full = (xf @ q_weight.float().transpose(0, 1)).view(batch, tokens, -1, 2 * head_dim)
    query, gate = torch.chunk(q_full, 2, dim=-1)
    gate = gate.reshape(batch, tokens, -1)
    query = norm(query, q_norm_weight).transpose(1, 2)
    key = norm((xf @ k_weight.float().transpose(0, 1)).view(batch, tokens, -1, head_dim),
               k_norm_weight).transpose(1, 2)
    value = (xf @ v_weight.float().transpose(0, 1)).view(batch, tokens, -1, head_dim)
    value = value.transpose(1, 2)
    query, key = _rope32(query, cos, sin), _rope32(key, cos, sin)
    dtype = x.dtype
    return query.to(dtype), key.to(dtype), value.to(dtype), gate.to(dtype)


def forward_production(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos,
                       sin, eps=1e-6, *, modeling):
    input_shape = x.shape[:-1]
    head_dim = q_norm_weight.shape[0]
    hidden_shape = (*input_shape, -1, head_dim)
    query_states, gate = torch.chunk(
        F.linear(x, q_weight).view(*input_shape, -1, head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = gemma_rmsnorm_forward_ref(
        query_states.view(hidden_shape), q_norm_weight, eps).transpose(1, 2)
    key_states = gemma_rmsnorm_forward_ref(
        F.linear(x, k_weight).view(hidden_shape), k_norm_weight, eps).transpose(1, 2)
    value_states = F.linear(x, v_weight).view(hidden_shape).transpose(1, 2)
    query_states, key_states = modeling.apply_rotary_pos_emb(
        query_states, key_states, cos, sin)
    return query_states, key_states, value_states, gate


__all__ = ["forward_production", "forward_ref"]
