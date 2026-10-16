"""OLMoE-1B-7B's published architecture, written out field by field.

Copied from ``allenai/OLMoE-1B-7B-0924``'s ``config.json`` (a local copy is at
``/projects/bcjw/akumdam/models/OLMoE-1B-7B-0924/config.json``) rather than
downloaded: a reference must be reproducible on a node with no network, and a
config fetched at runtime would make every case's identity depend on what the
Hub served that day.

Every field of the file that describes the network is in :data:`OLMOE_1B_7B`,
with two departures, both stated so nothing is silent:

* ``rms_norm_eps`` is **absent** from the published file; ``OlmoeConfig``'s
  default (1e-5) is what the checkpoint trained with and is written here so the
  workload identity does not depend on a library default
  (:data:`PUBLISHED_OMITTED`).
* ``output_router_logits`` is published ``false`` and pinned **true** here
  (:data:`PUBLISHED_OVERRIDDEN`). With it on, ``OlmoeForCausalLM`` adds the
  load-balancing auxiliary loss (``router_aux_loss_coef`` 0.01 over every
  layer's router logits) to the cross entropy, which is the loss OLMoE is
  trained with; off, the step is a plain next-token loss that never trains the
  router against collapse. The published ``false`` is an inference-time
  default: it only decides whether the model *returns* the router logits.

The fields that describe the file or the run rather than the network are kept
apart in :data:`NON_ARCHITECTURAL`.

This module imports nothing heavier than the standard library.
"""

from __future__ import annotations

from typing import Any, Mapping

MODEL_NAME = "OLMoE-1B-7B"

#: The architecture, as ``config.json`` publishes it (see the module docstring
#: for the two stated departures). 6.92B parameters, 1.28B active per token --
#: ``analytic_parameter_count`` reproduces both from these numbers alone.
OLMOE_1B_7B: Mapping[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    # ``None``: no clamp on q/k/v after projection. The qkv site is declared
    # without one, and the spec refuses an architecture that sets it.
    "clip_qkv": None,
    "eos_token_id": 50279,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    # Per expert. The dense equivalent would be 8 x 1024 = 8192 for the 8
    # experts a token visits.
    "intermediate_size": 1024,
    "max_position_embeddings": 4096,
    "model_type": "olmoe",
    # The chosen experts' softmax probabilities are used as they are, not
    # renormalized to sum to one.
    "norm_topk_prob": False,
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 16,
    # Multi-head attention: as many KV heads as query heads.
    "num_key_value_heads": 16,
    "output_router_logits": True,
    "pad_token_id": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "router_aux_loss_coef": 0.01,
    "tie_word_embeddings": False,
    "vocab_size": 50304,
}

#: Published value of each field this workload pins differently.
PUBLISHED_OVERRIDDEN: Mapping[str, Any] = {"output_router_logits": False}

#: Fields the published file omits, written here at the library default.
PUBLISHED_OMITTED: Mapping[str, Any] = {"rms_norm_eps": 1e-05}

#: ``config.json`` fields that are not part of :data:`OLMOE_1B_7B`, and why:
#: ``torch_dtype`` and ``use_cache`` are run settings every case pins itself;
#: the other two describe the file.
NON_ARCHITECTURAL: Mapping[str, Any] = {
    "architectures": ["OlmoeForCausalLM"],
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
}

PUBLISHED_METADATA: Mapping[str, Any] = {
    "repository": "allenai/OLMoE-1B-7B-0924",
    "checkpoint": "13.8 GB bfloat16 safetensors in three shards (not needed: weights are random)",
    "local_config": "/projects/bcjw/akumdam/models/OLMoE-1B-7B-0924/config.json",
}

#: Every layer is the same kind of block; the name exists because the shared
#: snapshot machinery keys representative layers by block kind.
BLOCK_KIND = "decoder_layer"

#: Mid-stack of 16, as Llama-3.2-1B uses 8 of 16.
REPRESENTATIVE_LAYER = 8
REPRESENTATIVE_LAYERS: Mapping[str, int] = {BLOCK_KIND: REPRESENTATIVE_LAYER}

#: The canonical invocation: batch 2 x sequence 2048 in bfloat16, experts
#: through torch's grouped GEMM (see ``levels.level3.spec``).
CANONICAL_BATCH = 2
CANONICAL_SEQ = 2048
CANONICAL_DTYPE = "bfloat16"
CANONICAL_EXPERTS = "grouped_mm"


def layer_kind(layer_index: int, arch: Mapping[str, Any] = OLMOE_1B_7B) -> str:
    """Every OLMoE layer is a ``decoder_layer``; the index is still range-checked."""
    layers = int(arch["num_hidden_layers"])
    if not 0 <= layer_index < layers:
        raise ValueError(f"layer_index {layer_index} is outside the architecture's {layers} layers")
    return BLOCK_KIND


def with_layers(arch: Mapping[str, Any], layers: int) -> dict[str, Any]:
    """The architecture cut to its first ``layers`` layers."""
    out = dict(arch)
    out["num_hidden_layers"] = int(layers)
    return out


def config_record(arch: Mapping[str, Any] = OLMOE_1B_7B, *, name: str = "olmoe_1b_7b"):
    """The :class:`~evograd.opdecl.models.MoeModelConfig` these fields describe.

    For the published fields this equals ``evograd.opdecl.models.OLMOE_1B_7B``
    (a test holds the two equal); for a reduced architecture it is the record
    whose ``*_dims`` builders state that architecture's site shapes.
    """
    from evograd.opdecl.models import MoeModelConfig

    heads = int(arch["num_attention_heads"])
    return MoeModelConfig(
        name=name,
        hidden=int(arch["hidden_size"]),
        intermediate=int(arch["intermediate_size"]),
        n_experts=int(arch["num_experts"]),
        top_k=int(arch["num_experts_per_tok"]),
        n_heads=heads,
        n_kv_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch.get("head_dim") or int(arch["hidden_size"]) // heads),
        vocab=int(arch["vocab_size"]),
        layers=int(arch["num_hidden_layers"]),
        rope_theta=float(arch["rope_theta"]),
        rms_norm_eps=float(arch["rms_norm_eps"]),
        dtype=str(NON_ARCHITECTURAL["torch_dtype"]),
        norm_topk_prob=bool(arch["norm_topk_prob"]),
        tie_word_embeddings=bool(arch["tie_word_embeddings"]),
    )


__all__ = [
    "BLOCK_KIND", "CANONICAL_BATCH", "CANONICAL_DTYPE", "CANONICAL_EXPERTS", "CANONICAL_SEQ",
    "MODEL_NAME", "NON_ARCHITECTURAL", "OLMOE_1B_7B", "PUBLISHED_METADATA", "PUBLISHED_OMITTED",
    "PUBLISHED_OVERRIDDEN", "REPRESENTATIVE_LAYER", "REPRESENTATIVE_LAYERS", "config_record",
    "layer_kind", "with_layers",
]
