"""Trusted PyTorch forward for dynamic tanh (DyT)."""

import torch


def dyt_forward_ref(
    x: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    y = gamma.float() * torch.tanh(alpha.float() * x.float()) + beta.float()
    return y.to(x.dtype)
