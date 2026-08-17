"""Trusted PyTorch forward for squared ReLU."""

import torch


def relu_squared_forward_ref(x: torch.Tensor) -> torch.Tensor:
    relu = torch.relu(x)
    return relu * relu
