"""Trusted PyTorch forward for SwiGLU."""

import torch


def swiglu_forward_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    return torch.nn.functional.silu(a.float()).to(a.dtype) * b
