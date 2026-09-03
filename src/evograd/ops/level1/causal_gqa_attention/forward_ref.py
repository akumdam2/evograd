"""References for causal grouped-query scaled dot-product attention.

Two spellings, and only one of them may ever be timed.

``causal_gqa_attention_forward_ref`` states the mathematics: expand the KV heads,
score, mask the future, softmax in float32, weight the values. It materializes a
``[B, HQ, T, T]`` score matrix -- at the observed Qwen3-0.6B shape that is
``2 x 16 x 2048 x 2048`` float32, 512 MiB the real execution never allocates --
so it is the correctness oracle and nothing else.

``causal_gqa_attention_forward_production`` is one
``F.scaled_dot_product_attention`` call with ``is_causal=True`` and
``enable_gqa=True``, which is what a training step runs and what
``runtime_forward`` names. Timing the eager baseline through the dense spelling
would report how much faster a candidate is than a strawman.

The boundary is attention alone. The output projection that usually follows it
is a separate GEMM and stays in the Level-2 ``qwen3_attention`` task.
"""

import math

import torch
import torch.nn.functional as F


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> int:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be [B, heads, T, D]")
    if k.shape != v.shape:
        raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} must match")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k must agree on batch, sequence and head dimension")
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    return n_q // n_kv


def causal_gqa_attention_forward_ref(q, k, v):
    groups = _check(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    key = k.repeat_interleave(groups, dim=1)
    value = v.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q.float(), key.float().transpose(-2, -1)) * scale
    tokens, kv_tokens = q.shape[2], key.shape[2]
    causal = torch.ones(tokens, kv_tokens, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    attn = torch.matmul(weights, value)
    if q.is_contiguous():
        return attn
    # SDPA hands the output back in the query's layout, so a decoder that passed
    # a head-major view gets one back. `matmul` always returns contiguous, so the
    # oracle would otherwise disagree with the production spelling on stride --
    # a real difference to whatever consumes the output, and one the correctness
    # gate compares. The round trip through [B, T, H, D] is differentiable and
    # reproduces exactly the stride the transpose-out-of-token-major gives.
    return attn.transpose(1, 2).contiguous().transpose(1, 2)


def causal_gqa_attention_forward_production(q, k, v):
    """The branch a real training step takes: one fused SDPA call.

    ``attn_mask=None`` with ``is_causal=True`` is what a decoder passes when the
    causal pattern needs no explicit mask, and ``enable_gqa=True`` broadcasts the
    KV heads inside the kernel rather than materializing the expansion. Both are
    read off the Qwen3-0.6B harvest, not assumed.
    """
    _check(q, k, v)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=True,
    )
