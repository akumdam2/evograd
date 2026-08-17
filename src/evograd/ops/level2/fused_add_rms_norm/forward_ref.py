"""Trusted PyTorch forward for fused residual add plus RMSNorm."""

import torch


def fused_add_rms_norm_forward_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    summed = x + residual
    rstd = torch.rsqrt(summed.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (summed * rstd.to(summed.dtype)) * weight


def fused_add_rms_norm_runtime_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """The residual add plus PyTorch's fused RMSNorm.

    The definition above spells the normalization out in primitives so AtenIR
    can lower it; ``F.rms_norm`` computes the same thing in one kernel, and is
    what the eager baseline is timed through. The add stays separate — fusing it
    into the norm is exactly the optimization this task asks a candidate to
    find, so the baseline must not have it for free.
    """
    return torch.nn.functional.rms_norm(
        x + residual, (x.shape[-1],), weight=weight, eps=eps
    )
