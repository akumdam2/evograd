"""PyTorch reference for Qwen3's causal grouped-query attention plus output projection.

Two spellings of one contract.

``qwen3_attention_forward_ref`` is the correctness oracle. It writes the
attention out in primitives -- expand the KV heads, score, mask, softmax in
float32, weight the values -- so the oracle differentiates the definition rather
than someone's optimization of it, and so a seed can be lowered from it. It is
emphatically *not* what anything is timed against: it materializes a
``[B, HQ, T, T]`` score matrix, which at the observed shape is
``2 x 16 x 2048 x 2048`` float32 = 512 MiB that the real execution never
allocates. Timing an eager baseline through this spelling would report how much
faster a candidate is than a strawman.

``qwen3_attention_forward_production`` is the branch Transformers actually takes
here, and is what ``runtime_forward`` names: one
``F.scaled_dot_product_attention`` with ``is_causal=True``, ``attn_mask=None``
and ``enable_gqa=True``, then the head merge and the output projection.

**The boundary.** It starts at q, k, v -- already projected, already RMSNorm'd
over the head dimension, already rotated -- and ends after ``o_proj``.
``q_proj``/``k_proj``/``v_proj``, the Q/K head-dimension norms and the rotary
embedding are deliberately *outside* it; they belong to a later
``qwen3_qkv_norm_rope`` task. Nothing here computes or consumes ``cos``/``sin``.
"""

import math

import torch
import torch.nn.functional as F


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o_weight: torch.Tensor) -> int:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be [B, heads, T, D]")
    if k.shape != v.shape:
        raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k must agree on batch, sequence and head dimension")
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    fan_in = n_q * q.shape[3]
    if o_weight.ndim != 2 or o_weight.shape[1] != fan_in:
        raise ValueError(f"o_weight must be [H, {fan_in}], got {tuple(o_weight.shape)}")
    return n_q // n_kv


def _merge_and_project(attn: torch.Tensor, o_weight: torch.Tensor) -> torch.Tensor:
    """``[B, HQ, T, D] -> [B, T, HQ*D] -> [B, T, H]``, exactly as the module does it."""
    batch, _, tokens, _ = attn.shape
    merged = attn.transpose(1, 2).contiguous().reshape(batch, tokens, -1)
    return F.linear(merged, o_weight)


def qwen3_attention_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    groups = _check(q, k, v, o_weight)
    scale = 1.0 / math.sqrt(q.shape[-1])

    # `enable_gqa=True` broadcasts the KV heads inside the kernel; here the
    # expansion is written out, because this spelling exists to state the
    # mathematics rather than to avoid the memory.
    key = k.repeat_interleave(groups, dim=1)
    value = v.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q.float(), key.float().transpose(-2, -1)) * scale
    tokens, kv_tokens = q.shape[2], key.shape[2]
    causal = torch.ones(tokens, kv_tokens, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    attn = torch.matmul(weights, value)
    return _merge_and_project(attn, o_weight)


def qwen3_attention_forward_production(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_weight: torch.Tensor,
) -> torch.Tensor:
    """The exact branch ``transformers.integrations.sdpa_attention`` takes here.

    ``attn_mask=None`` with ``is_causal=True`` is what the model passes when the
    causal pattern needs no explicit mask, and ``enable_gqa=True`` is what it
    passes when the KV heads can be broadcast rather than materialized. Both are
    read off the harvest, not assumed.
    """
    _check(q, k, v, o_weight)
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=True,
    )
    return _merge_and_project(attn, o_weight)
