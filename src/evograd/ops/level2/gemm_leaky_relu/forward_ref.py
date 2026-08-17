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


def gemm_leaky_relu_runtime_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    negative_slope: float = 0.01,
) -> torch.Tensor:
    """The same GEMM with PyTorch's leaky ReLU rather than a where().

    ``torch.where`` on a comparison materializes a boolean mask and a scaled
    copy; ``F.leaky_relu`` is a single elementwise kernel. Fusing it into the
    GEMM epilogue is the point of the task, so the baseline keeps them separate.
    """
    return torch.nn.functional.leaky_relu(a @ b, negative_slope=negative_slope)
