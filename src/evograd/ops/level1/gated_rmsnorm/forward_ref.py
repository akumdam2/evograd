"""Reference for the gated RMSNorm a Gated DeltaNet applies to its output.

Written to follow ``Qwen3_5RMSNormGated.forward`` cast for cast, because the
order of the casts is part of the definition: normalize in float32, cast back to
the input dtype, scale by the weight *in that dtype*, then multiply by
``silu(gate)`` computed in float32 and cast once more. Moving any one cast
changes the result by a rounding step, so this is both the oracle and the
production spelling.
"""

import torch
import torch.nn.functional as F


def gated_rmsnorm_forward_ref(x, z, weight, eps=1e-6):
    if x.shape != z.shape:
        raise ValueError(f"x {tuple(x.shape)} and z {tuple(z.shape)} must match")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be [hidden] matching x's last axis")
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    hidden = weight * hidden.to(input_dtype)
    hidden = hidden * F.silu(z.to(torch.float32))
    return hidden.to(input_dtype)
