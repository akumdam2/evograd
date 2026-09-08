"""PyTorch reference for Qwen3's q/k/v projection, per-head RMSNorm and RoPE.

Two spellings of one contract, as everywhere else here.

``qwen3_qkv_norm_rope_forward_ref`` is the correctness oracle. It keeps the
RMSNorm scale and the whole rotary application in float32 and casts once at the
end of each, which is deliberately more accurate than the model: Transformers
computes the RMSNorm variance in float32 but applies the learned weight *after*
casting back, and rotates entirely in BF16.

``qwen3_qkv_norm_rope_forward_production`` is that exact Transformers spelling,
and is what ``runtime_forward`` names, so the eager baseline is timed through
what a real step runs.

**The boundary.** It starts at the already-normalized residual stream and ends
with ``(q, k, v)`` in head-major layout, ready for
``F.scaled_dot_product_attention``. SDPA and ``o_proj`` are *not* part of it;
they are ``qwen3_attention``. ``cos`` and ``sin`` are inactive tables computed
once per step by the rotary embedding module and shared by all 28 layers, so
they are inputs here and receive no gradient.

The output layout is part of the contract. ``q`` is
``[B, HQ, T, D]`` with head-major strides because the model reaches it by
``.view(B, T, HQ, D).transpose(1, 2)``; both spellings must produce that, and
``verify_runtime_forward`` compares strides as well as values.
"""

import torch
import torch.nn.functional as F


def _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin):
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
    head_dim = q_norm_weight.shape[-1]
    if k_norm_weight.shape != q_norm_weight.shape:
        raise ValueError("q_norm_weight and k_norm_weight must have the same shape")
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
    if cos.shape[-1] != head_dim or sin.shape != cos.shape:
        raise ValueError(
            f"cos and sin must be [1, T, {head_dim}], got {tuple(cos.shape)} and "
            f"{tuple(sin.shape)}"
        )
    return head_dim


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qwen3_qkv_norm_rope_forward_ref(
    x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps=1e-6
):
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
    shape = (*x.shape[:-1], -1, head_dim)

    def head_rms_norm(heads, weight):
        # The whole normalization in float32, including the learned scale, then
        # one cast. The model casts before the scale; this is the more accurate
        # spelling a reference should be.
        wide = heads.float()
        variance = wide.pow(2).mean(-1, keepdim=True)
        return (weight.float() * wide * torch.rsqrt(variance + eps)).to(heads.dtype)

    q = head_rms_norm(F.linear(x, q_weight).view(shape), q_norm_weight).transpose(1, 2)
    k = head_rms_norm(F.linear(x, k_weight).view(shape), k_norm_weight).transpose(1, 2)
    v = F.linear(x, v_weight).view(shape).transpose(1, 2)

    cos_b = cos.unsqueeze(1).float()
    sin_b = sin.unsqueeze(1).float()

    def rotate(t):
        wide = t.float()
        return ((wide * cos_b) + (_rotate_half(wide) * sin_b)).to(t.dtype)

    return rotate(q), rotate(k), v


def qwen3_qkv_norm_rope_forward_production(
    x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps=1e-6
):
    """The exact spelling ``Qwen3Attention.forward`` executes.

    ``Qwen3RMSNorm`` casts back to the input dtype before multiplying by the
    weight, and ``apply_rotary_pos_emb`` runs entirely in the model dtype. Both
    are reproduced here rather than improved, because this is what is timed.
    """
    head_dim = _check(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
    shape = (*x.shape[:-1], -1, head_dim)

    def rms_norm(heads, weight):
        input_dtype = heads.dtype
        wide = heads.to(torch.float32)
        variance = wide.pow(2).mean(-1, keepdim=True)
        wide = wide * torch.rsqrt(variance + eps)
        return weight * wide.to(input_dtype)

    q = rms_norm(F.linear(x, q_weight).view(shape), q_norm_weight).transpose(1, 2)
    k = rms_norm(F.linear(x, k_weight).view(shape), k_norm_weight).transpose(1, 2)
    v = F.linear(x, v_weight).view(shape).transpose(1, 2)

    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q_embed = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_embed, k_embed, v
