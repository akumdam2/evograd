"""Trusted PyTorch forward for the rectifier."""

import torch


def relu_forward_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
