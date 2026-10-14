"""References for rotary embedding applied to the leading ``R`` of ``D`` channels.

Only ``x[..., :R]`` is rotated (``rotate_half`` convention: the first half of
the rotary slice pairs with the second); ``x[..., R:]`` passes through and is
concatenated back. ``R`` is the width of the cos/sin tables.

The reference rotates in float32 and casts once; the production spelling is
HuggingFace's ``apply_rotary_pos_emb`` for one tensor, in the input dtype.
Both return a fresh contiguous tensor: the concatenation is part of the
operator, whatever layout ``x`` arrived in.
"""

import torch


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _check(x, cos, sin):
    if x.ndim != 4:
        raise ValueError(f"x must be [B, H, T, D], got {tuple(x.shape)}")
    if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[0] != 1:
        raise ValueError(f"cos/sin must be [1, T, R], got {tuple(cos.shape)}")
    rotary = cos.shape[-1]
    if rotary % 2 or rotary > x.shape[-1] or cos.shape[1] != x.shape[2]:
        raise ValueError(f"rotary width {rotary} does not fit x {tuple(x.shape)}")
    return rotary


def partial_rope_forward_ref(x, cos, sin):
    rotary = _check(x, cos, sin)
    c = cos.float().unsqueeze(1)
    s = sin.float().unsqueeze(1)
    xf = x.float()
    x_rot, x_pass = xf[..., :rotary], xf[..., rotary:]
    rotated = x_rot * c + _rotate_half(x_rot) * s
    return torch.cat([rotated, x_pass], dim=-1).to(x.dtype)


def partial_rope_forward_production(x, cos, sin):
    """HuggingFace's partial ``apply_rotary_pos_emb``, for one tensor."""
    rotary = _check(x, cos, sin)
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    x_rot, x_pass = x[..., :rotary], x[..., rotary:]
    rotated = (x_rot * c) + (_rotate_half(x_rot) * s)
    return torch.cat([rotated, x_pass], dim=-1)
