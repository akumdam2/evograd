"""The post-attention residual add and its ``(1 + w)`` RMSNorm.

Two outputs, like every residual fusion in this repository: the normalized value
feeds the MLP and the un-normalized sum is the residual the block adds back at
its end. Production is ``residual + branch`` in the model dtype, then the
``(1 + w)`` RMSNorm arithmetic; the oracle adds and normalizes in float32.
"""

from __future__ import annotations

import torch

from evograd.ops.level1.gemma_rmsnorm.forward_ref import gemma_rmsnorm_forward_ref


def forward_ref(branch, residual, weight, eps=1e-6):
    summed = residual.float() + branch.float()
    normed = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * (1.0 + weight.float())
    return normed.to(branch.dtype), summed.to(branch.dtype)


def forward_production(branch, residual, weight, eps=1e-6):
    summed = residual + branch
    return gemma_rmsnorm_forward_ref(summed, weight, eps), summed


__all__ = ["forward_production", "forward_ref"]
