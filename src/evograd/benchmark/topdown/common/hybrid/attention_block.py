"""The gated full-attention module: projections, ``(1 + w)`` q/k norms, partial
RoPE, causal GQA SDPA, the sigmoid output gate and o_proj.

``Qwen3_5Attention.forward`` from its input to its output::

    q|gate = q_proj(x) viewed [B, T, HQ, 2D] and chunked -- per head, each
             head's query followed by its gate (interleaved, not two halves)
    gate   -> [B, T, HQ*D]
    q      -> q_norm (zero-centred, 1 + w) -> [B, HQ, T, D]
    k      -> k_proj -> k_norm               -> [B, HK, T, D]
    v      -> v_proj                         -> [B, HK, T, D]   (a transposed view)
    q, k   -> RoPE on the first R = cos.shape[-1] of D channels
    a      =  causal GQA SDPA(q, k, v), scale D^-0.5 -> [B, T, HQ*D]
    y      =  o_proj(a * sigmoid(gate))

Production is that code with the module calls replaced by their tensor
equivalents, in the same order, dtypes and layouts: ``F.linear`` for the
projections, the ``(1 + w)`` norm as ``Qwen3_5RMSNorm`` computes it (float32,
one cast at the end), the modeling module's own ``apply_rotary_pos_emb``, and
the mask-free causal call Transformers' ``sdpa`` interface makes
(``is_causal=True``, ``enable_gqa=True``). The oracle keeps everything in
float32 and casts once. cos/sin are inputs, never recomputed: Qwen3.5's
interleaved MRoPE and a 1D partial RoPE differ by ~6e-8 in float32.
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


def forward_ref(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_weight, cos,
                sin, eps=1e-6):
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
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = (query @ key.transpose(-2, -1)) * head_dim ** -0.5
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
    weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    attn = (weights @ value).transpose(1, 2).reshape(batch, tokens, -1)
    gated = attn * torch.sigmoid(gate)
    return (gated @ o_weight.float().transpose(0, 1)).to(x.dtype)


def forward_production(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, o_weight,
                       cos, sin, eps=1e-6, *, modeling):
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
    tokens, heads = query_states.shape[2], query_states.shape[1]
    # bool(): under symbolic shapes (torch.compile, make_fx) the comparisons are SymBools,
    # which SDPA's bool arguments refuse; Transformers' `... and ...` makes the same bool.
    attn = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=None, dropout_p=0.0,
        scale=head_dim ** -0.5, is_causal=bool(tokens > 1),
        enable_gqa=bool(heads != key_states.shape[1]),
    )
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.reshape(*input_shape, -1).contiguous()
    attn = attn * torch.sigmoid(gate)
    return F.linear(attn, o_weight)


__all__ = ["forward_production", "forward_ref"]
