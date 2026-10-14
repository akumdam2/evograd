"""Gated attention's core: causal GQA SDPA, the sigmoid output gate, o_proj.

``y = o_proj(SDPA(q, k, v) * sigmoid(gate))`` with scale ``D^-0.5``. Production
is what Transformers' ``sdpa`` interface runs for the mask-free causal call
(``is_causal=True``, ``enable_gqa=True``), the output transposed back to
``[B, T, HQ, D]``, then the gate and the projection in the model dtype. The
oracle materializes float32 scores and keeps everything in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_ref(q, k, v, gate, o_weight):
    batch, heads, tokens, head_dim = q.shape
    groups = heads // k.shape[1]
    key = k.float().repeat_interleave(groups, dim=1)
    value = v.float().repeat_interleave(groups, dim=1)
    scores = (q.float() @ key.transpose(-2, -1)) * head_dim ** -0.5
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()
    weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    attn = (weights @ value).transpose(1, 2).reshape(batch, tokens, -1)
    gated = attn * torch.sigmoid(gate.float())
    return (gated @ o_weight.float().transpose(0, 1)).to(q.dtype)


def forward_production(q, k, v, gate, o_weight):
    batch, heads, tokens, head_dim = q.shape
    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, scale=head_dim ** -0.5,
        is_causal=tokens > 1, enable_gqa=heads != k.shape[1],
    )
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.reshape(batch, tokens, -1).contiguous()
    attn = attn * torch.sigmoid(gate)
    return F.linear(attn, o_weight)


__all__ = ["forward_production", "forward_ref"]
