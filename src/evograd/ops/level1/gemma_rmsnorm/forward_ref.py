"""Reference for the zero-centred ``(1 + w)`` RMSNorm.

Normalization and scaling both happen in float32 and the result is cast back
only at the end -- ``(x * w).to(dtype)``, not ``x.to(dtype) * w`` -- which is
where it parts company with the Llama-style ``rmsnorm`` primitive as well as in
the ``1 +``. This *is* the spelling HuggingFace runs (``Gemma``/``Qwen3_5``
``RMSNorm.forward``), so there is no separate production spelling.
"""

import torch


def gemma_rmsnorm_forward_ref(x, weight, eps=1e-6):
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be [hidden] matching x's last axis")
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).type_as(x)
