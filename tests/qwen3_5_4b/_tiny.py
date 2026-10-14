"""A tiny Qwen3.5 text architecture with every structural feature of the real one.

Same field set as the published ``text_config``, only narrower: a Gated
DeltaNet with more value heads than key heads (so the q/k repeat is exercised),
gated attention with GQA and a partial rotary factor of 0.25, a dense SwiGLU MLP,
tied embeddings, interleaved MRoPE. Four layers in the published pattern, so
layer 2 is a Gated DeltaNet layer and layer 3 a full-attention layer, as 14 and
15 are in the real model.
"""

from __future__ import annotations

from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers

TINY_OVERRIDES = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "vocab_size": 512,
    "eos_token_id": None,
    "max_position_embeddings": 512,
}

TINY_ARCH = {**with_layers(QWEN3_5_4B, 4), **TINY_OVERRIDES}

GDN_LAYER = 2
FULL_LAYER = 3
BATCH = 2
SEQ = 40  # not a multiple of the 64-token chunk: the padding path runs
