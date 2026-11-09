"""Trusted PyTorch forward for a LayerNorm with a gamma and no beta."""

import torch


def layernorm_no_bias_forward_ref(x: torch.Tensor, weight: torch.Tensor,
                                  eps: float = 1e-5) -> torch.Tensor:
    """The definition in primitives, computed in float32."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must be a 1D tensor of x's hidden dimension")
    wide = x.float()
    mean = torch.mean(wide, dim=-1, keepdim=True)
    var = torch.mean((wide - mean) ** 2, dim=-1, keepdim=True)
    return ((wide - mean) * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def layernorm_no_bias_runtime_ref(x: torch.Tensor, weight: torch.Tensor,
                                  eps: float = 1e-5) -> torch.Tensor:
    """What ``nn.LayerNorm(dim, bias=False)`` calls."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, None, eps)
