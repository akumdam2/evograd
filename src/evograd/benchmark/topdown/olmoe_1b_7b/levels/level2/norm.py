"""``OlmoeRMSNorm``'s arithmetic, as a function, for the production spellings.

Transformers computes the variance in float32, scales in float32, casts back to
the input dtype and *then* multiplies by the weight. Two sites reproduce it --
the q/k norms inside ``qkv_norm_rope`` and the post-attention norm inside
``residual_rmsnorm`` -- and they must agree bitwise with the module, so the
spelling is written once here (checked against ``modeling_olmoe`` 5.16.1).
"""

from __future__ import annotations

import torch


def olmoe_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Exactly ``OlmoeRMSNorm.forward``."""
    input_dtype = hidden_states.dtype
    wide = hidden_states.to(torch.float32)
    variance = wide.pow(2).mean(-1, keepdim=True)
    wide = wide * torch.rsqrt(variance + eps)
    return weight * wide.to(input_dtype)


def rms_norm_ref32(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The oracle: the whole normalization in float32, including the weight,
    returned in float32 for the caller to cast once."""
    wide = hidden_states.float()
    return wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


__all__ = ["olmoe_rms_norm", "rms_norm_ref32"]
