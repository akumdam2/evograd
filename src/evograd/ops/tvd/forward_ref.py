"""Trusted PyTorch forward for batchmean total-variation distance."""

import torch


def tvd_forward_ref(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (0.5 * (p.float() - q.float()).abs()).sum() / p.shape[0]
