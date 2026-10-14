"""References for row-wise L2 normalization, as FLA defines it.

``y = x * rsqrt(sum(x^2, -1) + eps)`` -- eps *inside* the root and added to the
sum rather than to the norm, which is the definition FLA and HuggingFace's
``l2norm`` share. The reference accumulates in float32; the production spelling
is HuggingFace's, in the input dtype.
"""

import torch


def l2norm_forward_ref(x, eps=1e-6):
    if x.ndim != 2:
        raise ValueError(f"x must be [rows, cols], got {tuple(x.shape)}")
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype)


def l2norm_forward_production(x, eps=1e-6):
    """``transformers.models.qwen3_5.modeling_qwen3_5.l2norm``."""
    if x.ndim != 2:
        raise ValueError(f"x must be [rows, cols], got {tuple(x.shape)}")
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm
