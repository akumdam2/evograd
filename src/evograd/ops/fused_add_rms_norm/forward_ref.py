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
