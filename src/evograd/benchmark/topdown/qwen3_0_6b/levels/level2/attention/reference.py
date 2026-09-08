"""PyTorch reference for Qwen3's causal grouped-query attention plus output projection.

Two spellings of one contract.

``qwen3_attention_forward_ref`` is the correctness oracle. It writes the
attention out in primitives -- expand the KV heads, score, mask, softmax in
float32, weight the values -- so the oracle differentiates the definition rather
than someone's optimization of it, and so a seed can be lowered from it. It is
emphatically *not* what anything is timed against: it materializes a
``[B, HQ, T, T]`` score matrix, which at the observed shape is
``2 x 16 x 2048 x 2048`` float32 = 512 MiB that the real execution never
allocates. Timing an eager baseline through this spelling would report how much
faster a candidate is than a strawman.

``qwen3_attention_forward_production`` is the branch Transformers actually takes
here, and is what ``runtime_forward`` names: one
``F.scaled_dot_product_attention`` with ``is_causal=True``, ``attn_mask=None``
and ``enable_gqa=True``, then the head merge and the output projection.

**The boundary.** It starts at q, k, v -- already projected, already RMSNorm'd
over the head dimension, already rotated -- and ends after ``o_proj``.
``q_proj``/``k_proj``/``v_proj``, the Q/K head-dimension norms and the rotary
embedding are deliberately *outside* it; they belong to a later
``qwen3_qkv_norm_rope`` task. Nothing here computes or consumes ``cos``/``sin``.
"""

# The mathematics of this boundary is not Qwen3's alone -- Llama-3-8B computes
# the same thing at different widths -- so the implementation lives once in
# ``benchmark.topdown.common.level2_references`` and is re-exported here under
# the names this model's declaration points at. The declaration is unchanged:
# same ``forward``/``runtime_forward`` strings, same resolved functions, same
# numbers. What a candidate is generated against is still this model's symbol.

from evograd.benchmark.topdown.common.level2_references import (
    attention_projection_forward_ref as qwen3_attention_forward_ref,
    attention_projection_forward_production as qwen3_attention_forward_production,
)

__all__ = [
    "qwen3_attention_forward_ref",
    "qwen3_attention_forward_production",
]
