"""References for a depthwise causal 1-D convolution with an optional SiLU.

``causal_conv1d_forward_ref`` states the mathematics: every channel is filtered
by its own ``K``-tap kernel over the current and ``K - 1`` previous time steps,
with zeros before the sequence starts, accumulated in float32.

``causal_conv1d_forward_production`` is the spelling the torch fallback of
HuggingFace's ``causal_conv1d_fn`` executes: a grouped ``F.conv1d`` padded by
``K - 1`` on both sides and truncated back to ``T``.
"""

import torch
import torch.nn.functional as F


def _check(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be [B, C, T], got {tuple(x.shape)}")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(
            f"weight must be [C, K] with C={x.shape[1]}, got {tuple(weight.shape)}"
        )


def causal_conv1d_forward_ref(x, weight, silu=1):
    _check(x, weight)
    taps = weight.shape[1]
    xf = x.float()
    padded = F.pad(xf, (taps - 1, 0))
    tokens = x.shape[2]
    out = torch.zeros_like(xf)
    for j in range(taps):
        # y[t] = sum_j w[j] * x[t - (K - 1) + j]
        out = out + weight[:, j].float()[None, :, None] * padded[:, :, j : j + tokens]
    if silu:
        out = F.silu(out)
    return out.to(x.dtype)


def causal_conv1d_forward_production(x, weight, silu=1):
    """HuggingFace's ``causal_conv1d_fn`` torch fallback, without the bias."""
    _check(x, weight)
    channels, tokens = x.shape[1], x.shape[2]
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        x.to(weight.dtype), weight=weight.unsqueeze(1), bias=None,
        padding=padding, groups=channels,
    )[:, :, :tokens]
    if silu:
        out = F.silu(out)
    return out.to(x.dtype)
