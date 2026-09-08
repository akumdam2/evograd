"""PyTorch reference for Llama-3's q/k/v projection and RoPE.

The Llama counterpart of ``qwen3_qkv_norm_rope``, and the difference between
them is the whole reason this is a second declaration rather than a second
workload on the first: **Llama-3 has no per-head query/key normalization**.
``LlamaAttention.forward`` projects, reshapes, transposes, and rotates. Qwen3
inserts an ``RMSNorm`` over the head dimension between the reshape and the
transpose, and that norm is not an optional attribute a shared kernel could
skip -- it changes the arithmetic, the gradient, and the two learned ``[D]``
weights the operator carries.

Two spellings of one contract, as everywhere else here.

``llama3_qkv_rope_forward_ref`` is the correctness oracle: the rotary
application is carried out in float32 and cast once at the end.

``llama3_qkv_rope_forward_production`` is the exact spelling
``LlamaAttention.forward`` executes -- ``apply_rotary_pos_emb`` rotates entirely
in the model dtype -- and is what ``runtime_forward`` names, so the eager
baseline is timed through what a real step runs.

**The boundary.** It starts at the already-normalized residual stream and ends
with ``(q, k, v)`` in head-major layout, ready for
``F.scaled_dot_product_attention``. SDPA and ``o_proj`` are *not* part of it;
they are ``llama3_attention``. ``cos`` and ``sin`` are inactive tables computed
once per step by ``LlamaRotaryEmbedding`` and shared by all 32 layers, so they
are inputs here and receive no gradient.

The output layout is part of the contract. ``q`` is ``[B, HQ, T, D]`` with
head-major strides because the model reaches it by
``.view(B, T, HQ, D).transpose(1, 2)``; both spellings must produce that, and
``verify_runtime_forward`` compares strides as well as values.
"""

import torch
import torch.nn.functional as F


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
    # Unlike Qwen3, there is no norm weight to read the head dimension off, so
    # it comes from the rotary table -- the only other input that carries it.
    head_dim = cos.shape[-1]
    if q_weight.shape[0] % head_dim or k_weight.shape[0] % head_dim:
        raise ValueError(
            f"projection fan-outs {q_weight.shape[0]}/{k_weight.shape[0]} must be "
            f"multiples of the head dimension {head_dim}"
        )
    n_q, n_kv = q_weight.shape[0] // head_dim, k_weight.shape[0] // head_dim
    if n_q % n_kv:
        raise ValueError(
            f"grouped-query attention needs num_q_heads ({n_q}) divisible by "
            f"num_kv_heads ({n_kv})"
        )
    return head_dim


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# Does this match ? 
def llama3_qkv_rope_forward_ref(x, q_weight, k_weight, v_weight, cos, sin):
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

    return rotate(q), rotate(k), v


def llama3_qkv_rope_forward_production(x, q_weight, k_weight, v_weight, cos, sin):
    """The exact spelling ``LlamaAttention.forward`` executes.

    ``apply_rotary_pos_emb`` runs entirely in the model dtype. Reproduced rather
    than improved, because this is what is timed.
    """
    head_dim = _check(x, q_weight, k_weight, v_weight, cos, sin)
    shape = (*x.shape[:-1], -1, head_dim)

    q = F.linear(x, q_weight).view(shape).transpose(1, 2)
    k = F.linear(x, k_weight).view(shape).transpose(1, 2)
    v = F.linear(x, v_weight).view(shape).transpose(1, 2)

    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q_embed = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_embed, k_embed, v
