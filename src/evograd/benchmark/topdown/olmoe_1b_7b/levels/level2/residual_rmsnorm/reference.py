"""References for the post-attention residual add and its RMSNorm.

Two outputs, like every residual fusion in this repository: the normalized value
feeds the MoE block and the un-normalized sum is the residual the layer adds
back at its end. Production is ``residual + branch`` in the model dtype, then
``OlmoeRMSNorm``'s arithmetic (``norm.olmoe_rms_norm``); the oracle adds and
normalizes in float32 with the weight applied in float32.
"""

from __future__ import annotations

import torch

from ..norm import olmoe_rms_norm, rms_norm_ref32


def residual_rmsnorm_forward_ref(branch, residual, weight, eps=1e-5):
    summed = residual.float() + branch.float()
    normed = rms_norm_ref32(summed, weight, eps)
    return normed.to(branch.dtype), summed.to(branch.dtype)


def residual_rmsnorm_forward_production(branch, residual, weight, eps=1e-5):
    summed = residual + branch
    return olmoe_rms_norm(summed, weight, eps), summed


__all__ = ["residual_rmsnorm_forward_production", "residual_rmsnorm_forward_ref"]
