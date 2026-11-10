"""Trusted PyTorch forward for a LayerNorm with no parameters."""

import torch


def layernorm_no_affine_forward_ref(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """The definition in primitives, computed in float32."""
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got shape {tuple(x.shape)}")
    wide = x.float()
    mean = torch.mean(wide, dim=-1, keepdim=True)
    var = torch.mean((wide - mean) ** 2, dim=-1, keepdim=True)
    return ((wide - mean) * torch.rsqrt(var + eps)).to(x.dtype)


def layernorm_no_affine_runtime_ref(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """What ``nn.LayerNorm(dim, elementwise_affine=False)`` calls."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), None, None, eps)
