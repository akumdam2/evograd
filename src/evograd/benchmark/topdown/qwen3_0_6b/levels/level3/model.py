"""Which Transformers classes Qwen3 builds, bound to the shared builder.

Building a model from a spec, and reading back what was actually built rather
than what was asked for, is the same procedure for every decoder-only causal LM
and lives in :mod:`....common.model`. Qwen3's contribution is two class names and
a version floor.

Transformers is an optional dependency. Nothing here imports it at module import
time, so ``import evograd.benchmark.topdown.qwen3_0_6b`` works on a machine that has
never installed it, and the failure -- when it comes -- names the extra.
"""

from __future__ import annotations

from typing import Any

import torch

from ....common.model import (  # noqa: F401  (re-export)
    DTYPES,
    MissingDependencyError,
    ModelClasses,
    check_effective_settings,
    effective_settings,
    make_inputs,
    training_step,
)
from ....common import model as _common
from .spec import WorkloadSpec

#: Qwen3 landed in Transformers 4.51.0; earlier releases have no ``Qwen3Config``.
MIN_TRANSFORMERS = (4, 51)
#: The version this milestone was developed and measured against.
TESTED_TRANSFORMERS = "5.16.1"

CLASSES = ModelClasses(
    config_class="transformers:Qwen3Config",
    model_class="transformers:Qwen3ForCausalLM",
    min_transformers=MIN_TRANSFORMERS,
    tested_transformers=TESTED_TRANSFORMERS,
    extra="evograd[qwen3]",
    label="Qwen3",
)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    return _common.require_transformers(CLASSES)


def build_config(spec: WorkloadSpec):
    """A ``Qwen3Config`` carrying the spec's architecture and run settings."""
    return _common.build_config(spec, CLASSES)


def build_model(spec: WorkloadSpec):
    """The reference model, on ``spec.device``, in train mode.

    Random initialisation goes through the shared builder. A spec whose
    ``weights`` name a pinned checkpoint loads it instead (``_load_pretrained``,
    validated against the declared architecture) and is then put into exactly the
    state the shared builder leaves a model in.
    """
    if not spec.pretrained:
        return _common.build_model(spec, CLASSES)
    spec.validate()
    require_transformers()
    model = _load_pretrained(spec).to(device=spec.device)
    model.train()
    model.gradient_checkpointing_disable()
    model.config.use_cache = spec.use_cache
    return model


def pretrained_snapshot(spec: WorkloadSpec) -> str:
    """The local directory of the pinned checkpoint. Local-only, never fetched here."""
    import os

    from huggingface_hub import snapshot_download

    repo, revision = spec.pretrained_repo
    return snapshot_download(
        repo, revision=revision, local_files_only=True,
        cache_dir=os.environ.get("EVOGRAD_HF_CACHE") or None,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )


def rope_settings(config_like) -> tuple[float | None, str]:
    """``(theta, type)`` from either RoPE spelling, so equal rotations compare equal."""
    get = (config_like.get if isinstance(config_like, dict)
           else lambda key, default=None: getattr(config_like, key, default))
    scaling = get("rope_scaling") or {}
    theta = scaling.get("rope_theta") if isinstance(scaling, dict) else None
    if theta is None:
        theta = get("rope_theta")
    rope_type = (scaling.get("rope_type") or scaling.get("type") or "default") \
        if isinstance(scaling, dict) else "default"
    return (None if theta is None else float(theta), str(rope_type))


def _load_pretrained(spec: WorkloadSpec):
    """A pinned checkpoint, validated against the architecture the spec declares.

    The declared architecture is what every operator contract, patch site and
    calibration was derived from. A checkpoint whose config disagrees -- a
    different head count, an untied embedding, a wider vocabulary -- would run
    and silently measure a different model, so each field is compared before
    the weights are trusted. The QKV site's own shapes are asserted as well,
    because that is the contract the patched kernel was generated against.
    """
    from transformers import Qwen3ForCausalLM

    location = pretrained_snapshot(spec)
    model = None
    for keyword in ("dtype", "torch_dtype"):
        try:
            model = Qwen3ForCausalLM.from_pretrained(
                location, attn_implementation=spec.attn_implementation,
                use_cache=spec.use_cache, **{keyword: DTYPES[spec.dtype]})
            break
        except TypeError:
            continue
    if model is None:  # pragma: no cover - both keywords rejected
        raise RuntimeError("Qwen3ForCausalLM.from_pretrained accepted neither dtype keyword")
    if next(model.parameters()).dtype != DTYPES[spec.dtype]:
        raise RuntimeError(
            f"checkpoint loaded in {next(model.parameters()).dtype}, spec says {spec.dtype}")

    declared = spec.arch
    loaded = model.config
    # RoPE is the one setting with two spellings: Transformers 5 keeps it under
    # `rope_scaling={"rope_theta": ..., "rope_type": ...}` with `rope_theta=None`,
    # older configs as a bare `rope_theta` with `rope_scaling=None`. The same
    # rotation, different keys -- so it is compared as (theta, type), and every
    # other field is compared literally.
    rope_keys = {"rope_theta", "rope_scaling"}
    mismatched = {
        key: (getattr(loaded, key, None), value)
        for key, value in declared.items()
        if key not in rope_keys and getattr(loaded, key, None) != value
    }
    if rope_settings(declared) != rope_settings(loaded):
        mismatched["rope"] = (rope_settings(loaded), rope_settings(declared))
    if mismatched:
        raise RuntimeError(
            "the pinned checkpoint's architecture disagrees with the declared "
            f"workload: {mismatched}")

    # Parameter sharing: the declared architecture ties the embedding to the
    # output head, and the loss/gradient bookkeeping counts parameters by name.
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    if declared.get("tie_word_embeddings") and not tied:
        raise RuntimeError("declared tie_word_embeddings=True but the loaded head is untied")

    # The QKV contract the patched kernel was generated against.
    attention = model.model.layers[0].self_attn
    head_dim = declared["head_dim"]
    expected = {
        "q_proj.weight": (declared["num_attention_heads"] * head_dim, declared["hidden_size"]),
        "k_proj.weight": (declared["num_key_value_heads"] * head_dim, declared["hidden_size"]),
        "v_proj.weight": (declared["num_key_value_heads"] * head_dim, declared["hidden_size"]),
        "q_norm.weight": (head_dim,),
        "k_norm.weight": (head_dim,),
    }
    for name, shape in expected.items():
        module_name, _, param = name.partition(".")
        actual = tuple(getattr(getattr(attention, module_name), param).shape)
        if actual != shape:
            raise RuntimeError(f"self_attn.{name} is {actual}, the QKV contract needs {shape}")
    if getattr(attention, "q_proj").bias is not None and not declared.get("attention_bias"):
        raise RuntimeError("checkpoint has attention bias; the declared contract has none")
    return model
