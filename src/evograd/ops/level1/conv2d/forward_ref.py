"""PyTorch reference for a contiguous NCHW convolution."""

import torch
import torch.nn.functional as F


def conv2d_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("x and weight must be 4D NCHW/OIHW tensors")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must match output channels")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("input channels must match weight input channels")
    return F.conv2d(x, weight, bias, stride=1, padding=0)
