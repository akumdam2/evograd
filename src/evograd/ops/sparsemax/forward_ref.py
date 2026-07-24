"""Trusted PyTorch forward for sparsemax over the last dimension."""

import torch


def sparsemax_forward_ref(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    sorted_x, _ = torch.sort(xf, dim=-1, descending=True)
    cumulative = sorted_x.cumsum(dim=-1)
    ranks = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=xf.dtype)
    support = (1.0 + ranks * sorted_x) > cumulative
    support_size = support.to(xf.dtype).sum(dim=-1, keepdim=True)
    support_sum = torch.where(support, sorted_x, torch.zeros_like(sorted_x)).sum(
        dim=-1, keepdim=True
    )
    threshold = (support_sum - 1.0) / support_size
    return torch.clamp(xf - threshold, min=0.0).to(x.dtype)
