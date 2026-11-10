"""Trusted PyTorch forward for the gated linear unit."""

import torch


def glu_forward_ref(x: torch.Tensor) -> torch.Tensor:
    """``first * sigmoid(gates)`` with the sigmoid in float32."""
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError(f"x must be [rows, 2 * cols], got {tuple(x.shape)}")
    first, gates = x.chunk(2, dim=-1)
    return first * torch.sigmoid(gates.float()).to(x.dtype)


def glu_runtime_ref(x: torch.Tensor) -> torch.Tensor:
    """What ``nn.GLU(dim=-1)`` calls."""
    return torch.nn.functional.glu(x, dim=-1)
