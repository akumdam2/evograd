"""Trusted PyTorch forward for row-wise softmax."""

import torch


def softmax_forward_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)
