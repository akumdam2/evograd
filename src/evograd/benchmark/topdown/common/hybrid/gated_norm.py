"""The Gated DeltaNet's output stage: gated RMSNorm per value head, then out_proj.

``y = out_proj(RMSNormGated(o, z))``. Production follows the model's
``RMSNormGated`` cast for cast (normalize in float32, back to the input dtype,
scale, multiply by ``silu(float32(z))``, back again) and runs the projection in
the model dtype; the oracle keeps everything in float32 and casts once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.ops.level1.gated_rmsnorm.forward_ref import gated_rmsnorm_forward_ref


def forward_ref(o, z, w_norm, w_out, eps=1e-6):
    batch, tokens = o.shape[0], o.shape[1]
    of = o.float()
    normed = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    gated = normed * w_norm.float() * F.silu(z.float())
    y = gated.reshape(batch, tokens, -1) @ w_out.float().transpose(0, 1)
    return y.to(o.dtype)


def forward_production(o, z, w_norm, w_out, eps=1e-6):
    z_shape = z.shape
    core = gated_rmsnorm_forward_ref(o.reshape(-1, o.shape[-1]), z.reshape(-1, z.shape[-1]),
                                     w_norm, eps)
    core = core.reshape(z_shape)
    core = core.reshape(core.shape[0], core.shape[1], -1)
    return F.linear(core, w_out)


__all__ = ["forward_production", "forward_ref"]
