"""PyTorch reference for Qwen3's gated (SwiGLU) MLP block.

The reference accumulates the gate/up product in float32 and casts once, which
is deliberately *more* accurate than what Transformers runs: ``Qwen3MLP`` calls
``self.act_fn(...) * self.up_proj(x)`` entirely in the model dtype. A reference
exists to be the correct answer, not to reproduce a particular rounding, and
every other declaration here follows the same convention (see
``fused_moe_swiglu``). The cost of that choice is measured rather than assumed:
the Level-2 verification reports the reference against the captured
``Qwen3MLP`` invocation *and* against the BF16 spelling, so the difference
between the two is a number in a report instead of a footnote.
"""

# The mathematics of this boundary is not Qwen3's alone -- Llama-3-8B computes
# the same thing at different widths -- so the implementation lives once in
# ``benchmark.topdown.common.level2_references`` and is re-exported here under
# the names this model's declaration points at. The declaration is unchanged:
# same ``forward``/``runtime_forward`` strings, same resolved functions, same
# numbers. What a candidate is generated against is still this model's symbol.

from evograd.benchmark.topdown.common.level2_references import (
    gated_mlp_forward_ref as qwen3_swiglu_mlp_forward_ref,
    gated_mlp_forward_model_dtype as qwen3_swiglu_mlp_forward_hf,
)

__all__ = [
    "qwen3_swiglu_mlp_forward_ref",
    "qwen3_swiglu_mlp_forward_hf",
]
