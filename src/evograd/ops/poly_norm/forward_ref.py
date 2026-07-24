"""Trusted PyTorch forward for polynomial RMS-normalized features."""

import torch


def poly_norm_forward_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    xf = x.float()
    wf = weight.float()

    def normalize(value):
        return value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)

    y = (
        wf[0] * normalize(xf**3)
        + wf[1] * normalize(xf**2)
        + wf[2] * normalize(xf)
        + bias.float()
    )
    return y.to(x.dtype)
