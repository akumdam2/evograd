"""Forward references for Llama-3-8B's residual-add-plus-RMSNorm boundary.

The mathematics is shared with Qwen3-0.6B and lives once in
``benchmark.topdown.common.level2_references``. It is re-exported here under
the names this model's declaration points at.
"""

from evograd.benchmark.topdown.common.level2_references import (
    residual_rms_norm_forward_ref as llama3_residual_rmsnorm_forward_ref,
    residual_rms_norm_runtime_ref as llama3_residual_rmsnorm_runtime_ref,
)

__all__ = [
    "llama3_residual_rmsnorm_forward_ref",
    "llama3_residual_rmsnorm_runtime_ref",
]
