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

# The mathematics of this boundary is not Qwen3's alone -- Llama-3-8B computes
# the same thing at different widths -- so the implementation lives once in
# ``benchmark.topdown.common.level2_references`` and is re-exported here under
# the names this model's declaration points at. The declaration is unchanged:
# same ``forward``/``runtime_forward`` strings, same resolved functions, same
# numbers. What a candidate is generated against is still this model's symbol.

from evograd.benchmark.topdown.common.level2_references import (
    residual_rms_norm_forward_ref as fused_add_rms_norm_forward_ref,
    residual_rms_norm_runtime_ref as fused_add_rms_norm_runtime_ref,
)

__all__ = [
    "fused_add_rms_norm_forward_ref",
    "fused_add_rms_norm_runtime_ref",
]
