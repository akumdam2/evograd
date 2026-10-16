"""A tiny OLMoE architecture with every structural feature of the real one.

Same field set as the published configuration, only narrower: grouped-query
attention (so the group-of-one multi-head case is not the only path exercised),
full-width q/k norms, a router over four experts with two active per token,
untied embeddings. Four layers; layer 2 stands in for the real model's layer 8.
"""

from __future__ import annotations

from evograd.benchmark.topdown.olmoe_1b_7b.architecture import OLMOE_1B_7B, with_layers

TINY_OVERRIDES = {
    "hidden_size": 64,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 512,
    "eos_token_id": None,
    "pad_token_id": 1,
    "max_position_embeddings": 512,
}

TINY_ARCH = {**with_layers(OLMOE_1B_7B, 4), **TINY_OVERRIDES}

LAYER = 2
LAYERS = {"decoder_layer": LAYER}
BATCH = 2
SEQ = 24
