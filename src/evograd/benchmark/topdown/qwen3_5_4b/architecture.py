"""Qwen3.5-4B's published text architecture, written out field by field.

Copied from ``Qwen/Qwen3.5-4B``'s ``config.json`` -- its ``text_config``
section -- rather than downloaded: a reference must be reproducible on a node
with no network, and a config fetched at runtime would make every case's
identity depend on what the Hub served that day.

The published file describes ``Qwen3_5ForConditionalGeneration``: a 24-layer
ViT, the 32-layer text model and a one-layer MTP head. This workload is the
text model only (``Qwen3_5ForCausalLM``, which ignores ``model.visual.*`` and
``mtp.*``), so only ``text_config`` is here.

Every field of ``text_config`` that describes the network is in
:data:`QWEN3_5_4B` verbatim -- including ``layer_types``, which the published
file writes out; :func:`layer_types` derives the same
list from ``full_attention_interval`` and a test holds the two equal. The
fields that describe the file or the run rather than the network, and the two
that Transformers' Qwen3.5 code does not read (``mamba_ssm_dtype``, ``mtp_*``),
are kept apart in :data:`NON_ARCHITECTURAL` so nothing is dropped silently.

This module imports nothing heavier than the standard library.
"""

from __future__ import annotations

from typing import Any, Mapping

MODEL_NAME = "Qwen3.5-4B"

_LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
] * 8

#: The text architecture, exactly as ``config.json`` publishes ``text_config``.
#: 4.206B parameters (0.636B of them the tied embedding); 112.92M per Gated
#: DeltaNet layer and 107.48M per full-attention layer --
#: ``analytic_parameter_count`` reproduces each from these numbers alone.
QWEN3_5_4B: Mapping[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    # The attention output gate: q_proj fans out to query *and* gate per head.
    # Transformers' Qwen3.5 attention always has the gate and does not read this
    # field; it is architecture all the same, and it is passed to the config.
    "attn_output_gate": True,
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "layer_types": _LAYER_TYPES,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_5_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    # Interleaved MRoPE over 64 of the 256 head channels. For text-only input
    # its three position streams are identical, and the result differs from a
    # 1D partial RoPE by ~6e-8 in float32 -- not bitwise -- which is why every
    # site takes cos/sin as inputs rather than recomputing them.
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
    "tie_word_embeddings": True,
    "vocab_size": 248320,
}

#: ``text_config`` fields that are not part of :data:`QWEN3_5_4B`, and why.
#: ``dtype`` and ``use_cache`` are run settings every case pins itself;
#: ``mamba_ssm_dtype`` and the ``mtp_*`` pair are read by no Transformers
#: Qwen3.5 code path (the MTP head is not loaded by ``Qwen3_5ForCausalLM``).
NON_ARCHITECTURAL: Mapping[str, Any] = {
    "dtype": "bfloat16",
    "use_cache": True,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
}

#: The top-level ``config.json`` fields describing the file, for the record.
PUBLISHED_METADATA: Mapping[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "vision_config_depth": 24,
    "checkpoint": "9.32 GB bfloat16, 738 tensors (text, visual and mtp)",
}

#: The two representative blocks: the 15th layer (index 14) is a Gated DeltaNet
#: layer and the 16th (index 15) a gated full-attention layer -- mid-stack and
#: adjacent, so the two cases are the two halves of one four-layer period.
REPRESENTATIVE_LAYERS: Mapping[str, int] = {
    "linear_attention": 14,
    "full_attention": 15,
}

#: The canonical invocation: batch 1 x sequence 2048 in bfloat16 (see
#: ``levels.level3.spec`` for why batch 1).
CANONICAL_BATCH = 1
CANONICAL_SEQ = 2048
CANONICAL_DTYPE = "bfloat16"


def derived_layer_types(arch: Mapping[str, Any] = QWEN3_5_4B) -> tuple[str, ...]:
    """The per-layer token mixer, derived from ``full_attention_interval`` as
    Transformers derives it when ``layer_types`` is absent."""
    interval = int(arch["full_attention_interval"])
    return tuple(
        "linear_attention" if (i + 1) % interval else "full_attention"
        for i in range(int(arch["num_hidden_layers"]))
    )


def layer_types(arch: Mapping[str, Any] = QWEN3_5_4B) -> tuple[str, ...]:
    """The per-layer token mixer: ``layer_types`` as written, else derived."""
    written = arch.get("layer_types")
    if written:
        if len(written) != int(arch["num_hidden_layers"]):
            raise ValueError(f"layer_types has {len(written)} entries for "
                             f"{arch['num_hidden_layers']} layers")
        return tuple(written)
    return derived_layer_types(arch)


def layer_kind(layer_index: int, arch: Mapping[str, Any] = QWEN3_5_4B) -> str:
    types = layer_types(arch)
    if not 0 <= layer_index < len(types):
        raise ValueError(f"layer_index {layer_index} is outside the architecture's "
                         f"{len(types)} layers")
    return types[layer_index]


def with_layers(arch: Mapping[str, Any], layers: int) -> dict[str, Any]:
    """The architecture cut to its first ``layers`` layers, ``layer_types`` with it.

    ``num_hidden_layers`` alone cannot be overridden: the written-out
    ``layer_types`` would then disagree with it and Transformers refuses the
    config. A debug run that shrinks the depth keeps the published pattern.
    """
    out = dict(arch)
    out["num_hidden_layers"] = int(layers)
    written = list(arch.get("layer_types") or ())
    if written:
        if layers > len(written):
            raise ValueError(f"cannot extend {len(written)} written layer types to {layers} "
                             "layers")
        out["layer_types"] = written[:layers]
    return out


def config_record(arch: Mapping[str, Any] = QWEN3_5_4B, *, name: str = "qwen3_5_4b"):
    """The :class:`~evograd.opdecl.models.HybridDenseModelConfig` these fields describe.

    For the published fields this equals ``evograd.opdecl.models.QWEN3_5_4B`` (a
    test holds the two equal); for a reduced architecture it is the record whose
    ``*_dims`` builders state that architecture's site shapes.
    """
    from evograd.opdecl.models import HybridDenseModelConfig

    rope = arch["rope_parameters"]
    return HybridDenseModelConfig(
        name=name,
        hidden=int(arch["hidden_size"]),
        intermediate=int(arch["intermediate_size"]),
        layers=int(arch["num_hidden_layers"]),
        full_attention_interval=int(arch["full_attention_interval"]),
        n_heads=int(arch["num_attention_heads"]),
        n_kv_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch["head_dim"]),
        partial_rotary_factor=float(rope["partial_rotary_factor"]),
        rope_theta=float(rope["rope_theta"]),
        linear_num_key_heads=int(arch["linear_num_key_heads"]),
        linear_num_value_heads=int(arch["linear_num_value_heads"]),
        linear_key_head_dim=int(arch["linear_key_head_dim"]),
        linear_value_head_dim=int(arch["linear_value_head_dim"]),
        linear_conv_kernel_dim=int(arch["linear_conv_kernel_dim"]),
        vocab=int(arch["vocab_size"]),
        rms_norm_eps=float(arch["rms_norm_eps"]),
        dtype=str(NON_ARCHITECTURAL["dtype"]),
        tie_word_embeddings=bool(arch["tie_word_embeddings"]),
    )


__all__ = [
    "CANONICAL_BATCH",
    "CANONICAL_DTYPE",
    "CANONICAL_SEQ",
    "MODEL_NAME",
    "NON_ARCHITECTURAL",
    "PUBLISHED_METADATA",
    "QWEN3_5_4B",
    "REPRESENTATIVE_LAYERS",
    "config_record",
    "derived_layer_types",
    "layer_kind",
    "layer_types",
    "with_layers",
]
