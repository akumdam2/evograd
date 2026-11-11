"""PyTorch references for Llama-3.2-1B's attention block: ``LlamaAttention.forward``.

The boundary is the whole self-attention module: the normalized residual stream
``x`` in, the output projection's ``[B, T, H]`` out. Inside it, in order: the
q/k/v projections, the head-major view, RoPE on q and k, causal grouped-query
SDPA, the head merge and ``o_proj``. There is no per-head query/key RMSNorm
(that is Qwen3's block, not this one). ``cos`` and ``sin`` are Inactive tables
``LlamaRotaryEmbedding`` computes once per step and shares across the layers.

Two spellings of one contract.

``llama3_attention_block_forward_ref`` states the mathematics: the rotation in
float32, and attention with a materialized causal score matrix and a float32
softmax (the shared ``attention_projection_forward_ref``).

``llama3_attention_block_forward_production`` is the exact arithmetic the
installed Transformers runs (same ops, order, dtypes and layouts):
``apply_rotary_pos_emb`` in the model dtype, then the SDPA call
``sdpa_attention_forward`` makes on the mask-free causal path
(``attn_mask=None``, ``is_causal=True``, ``enable_gqa=True``, scale
``head_dim ** -0.5``, which equals the shared ``1 / sqrt(D)`` exactly at
``D = 64``), then ``transpose(1, 2).contiguous()``, the reshape and ``o_proj``.
It is what ``runtime_forward`` names, so the model reference every gate judges
against and the timed eager baseline are the module's own computation.
"""

import torch
import torch.nn.functional as F

from evograd.benchmark.topdown.common.level2_references import (
    attention_projection_forward_production as _attend,
    attention_projection_forward_ref as _attend_ref,
)


def _check(x, q_weight, k_weight, v_weight, cos, sin):
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, H], got {tuple(x.shape)}")
    hidden = x.shape[-1]
    for name, weight in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
    ):
        if weight.ndim != 2 or weight.shape[1] != hidden:
            raise ValueError(f"{name} must be [out, {hidden}], got {tuple(weight.shape)}")
    if k_weight.shape != v_weight.shape:
        raise ValueError("k_weight and v_weight must have the same shape")
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError(
            f"cos and sin must both be [1, T, D], got {tuple(cos.shape)} and "
            f"{tuple(sin.shape)}"
        )
    # No norm weight carries the head dimension here, so it comes from the
    # rotary table.
    head_dim = cos.shape[-1]
    if q_weight.shape[0] % head_dim or k_weight.shape[0] % head_dim:
        raise ValueError(
            f"projection fan-outs {q_weight.shape[0]}/{k_weight.shape[0]} must be "
            f"multiples of the head dimension {head_dim}"
        )
    return head_dim


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def llama3_attention_block_forward_ref(x, q_weight, k_weight, v_weight, o_weight, cos, sin):
    head_dim = _check(x, q_weight, k_weight, v_weight, cos, sin)
    shape = (*x.shape[:-1], -1, head_dim)

    q = F.linear(x, q_weight).view(shape).transpose(1, 2)
    k = F.linear(x, k_weight).view(shape).transpose(1, 2)
    v = F.linear(x, v_weight).view(shape).transpose(1, 2)

    cos_b = cos.unsqueeze(1).float()
    sin_b = sin.unsqueeze(1).float()

    def rotate(t):
        wide = t.float()
        return ((wide * cos_b) + (_rotate_half(wide) * sin_b)).to(t.dtype)

    return _attend_ref(rotate(q), rotate(k), v, o_weight)


def llama3_attention_block_forward_production(x, q_weight, k_weight, v_weight, o_weight,
                                              cos, sin):
    """``LlamaAttention.forward`` on the training path (no cache, no mask)."""
    head_dim = _check(x, q_weight, k_weight, v_weight, cos, sin)
    shape = (*x.shape[:-1], -1, head_dim)

    q = F.linear(x, q_weight).view(shape).transpose(1, 2)
    k = F.linear(x, k_weight).view(shape).transpose(1, 2)
    v = F.linear(x, v_weight).view(shape).transpose(1, 2)

    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q_embed = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (_rotate_half(k) * sin_b)
    return _attend(q_embed, k_embed, v, o_weight)


__all__ = ["llama3_attention_block_forward_production", "llama3_attention_block_forward_ref"]
