"""Trusted PyTorch forward for the sigmoid."""

import torch


def sigmoid_forward_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x.float()).to(x.dtype)


def sigmoid_runtime_ref(x: torch.Tensor) -> torch.Tensor:
    """The call a model makes, in the tensor's own dtype."""
    return torch.sigmoid(x)
