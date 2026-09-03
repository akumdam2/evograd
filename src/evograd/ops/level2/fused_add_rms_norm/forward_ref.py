"""Trusted PyTorch forward for fused residual add plus RMSNorm.

Two outputs, not one. The fusion site this task represents is the decoder
layer's residual stream:

    summed     = x + residual      # kept, and consumed again by the next block
    normalized = RMSNorm(summed)   # fed forward

A kernel that returned only ``normalized`` would have to be followed by a second
pass to recompute ``summed``, or the caller would keep the un-normalized sum
alive anyway -- which is precisely the memory traffic the fusion exists to
avoid. Returning both is what every real implementation does, Liger's included,
and it changes the backward: ``summed`` receives its own upstream gradient from
whatever consumes it downstream, and that gradient reaches ``x`` and ``residual``
without passing through the normalization at all.
"""

import torch


def fused_add_rms_norm_forward_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x + residual
    rstd = torch.rsqrt(summed.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = (summed * rstd.to(summed.dtype)) * weight
    return normalized, summed


def fused_add_rms_norm_runtime_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The residual add plus PyTorch's fused RMSNorm.

    The definition above spells the normalization out in primitives so AtenIR
    can lower it; ``F.rms_norm`` computes the same thing in one kernel, and is
    what the eager baseline is timed through. The add stays separate -- fusing it
    into the norm is exactly the optimization this task asks a candidate to
    find, so the baseline must not have it for free.

    ``summed`` is returned by both spellings, and it is the same tensor the
    normalization consumed, so a caller gets it at no extra cost.
    """
    summed = x + residual
    normalized = torch.nn.functional.rms_norm(
        summed, (x.shape[-1],), weight=weight, eps=eps
    )
    return normalized, summed
