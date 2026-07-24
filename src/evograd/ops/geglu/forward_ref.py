"""Trusted PyTorch forward for tanh-approximate GeGLU."""

import torch


def geglu_forward_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    return torch.nn.functional.gelu(a.float(), approximate="tanh").to(a.dtype) * b
