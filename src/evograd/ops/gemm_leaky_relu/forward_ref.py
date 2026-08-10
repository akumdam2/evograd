"""PyTorch reference for GEMM with a fused Leaky-ReLU epilogue."""

import torch


def gemm_leaky_relu_forward_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    negative_slope: float = 0.01,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must match b.shape[0] (K)")
    pre_activation = a @ b
    return torch.where(
        pre_activation >= 0,
        pre_activation,
        pre_activation * negative_slope,
    )
