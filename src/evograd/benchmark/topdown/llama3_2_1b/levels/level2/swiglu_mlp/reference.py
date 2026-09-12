"""Forward references for Llama-3-8B's gated MLP boundary.

The mathematics is shared with Qwen3-0.6B and lives once in
``benchmark.topdown.common.level2_references``. It is re-exported here under
the names this model's declaration points at.

``llama3_swiglu_mlp_forward_hf`` is the spelling ``LlamaMLP.forward`` executes,
without the float32 upcast the declared reference performs; verification
reports the difference rather than leaving the two contracts silently apart.
"""

from evograd.benchmark.topdown.common.level2_references import (
    gated_mlp_forward_model_dtype as llama3_swiglu_mlp_forward_hf,
    gated_mlp_forward_ref as llama3_swiglu_mlp_forward_ref,
)

__all__ = ["llama3_swiglu_mlp_forward_hf", "llama3_swiglu_mlp_forward_ref"]
