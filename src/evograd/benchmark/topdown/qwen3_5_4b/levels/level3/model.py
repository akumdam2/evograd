"""Which Transformers classes Qwen3.5 builds, bound to the shared builder.

Building a model from a spec and reading back what was built is the same
procedure for every decoder-only causal LM and lives in
:mod:`....common.model`. Qwen3.5's contribution is two class names, a version
floor, and one more thing to read back: the import-time kernel dispatch of
``modeling_qwen3_5`` (trap T-dispatch). ``torch_chunk_gated_delta_rule`` and
``causal_conv1d_fn`` pick FLA / ``causal_conv1d`` when the module is imported
if those packages are importable; :func:`dispatch_settings` reports what they
resolved to, and the declaration pins both to ``"torch"``.

Random weights: ``Qwen3_5ForCausalLM._from_config`` runs ``post_init``, so
every parameter is initialized by the model's own ``_init_weights``
(``dt_bias = 1``, ``A_log = log U(0.01, 16)``, ``(1 + w)`` norms at 0) under
``spec.seed``, on the CPU. A pinned checkpoint (``spec.weights``) loads the
text model of the published (vision-language) checkpoint; Transformers drops
its vision tower and MTP head and maps ``model.language_model.*``.

Transformers is optional: nothing here imports it at module import time.
"""

from __future__ import annotations

from typing import Any

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

#: ``models/qwen3_5`` first ships in Transformers 5.2.0 (checked 2026-10-10
#: against the published wheels: 5.1.0 has no ``qwen3_5`` package, 5.2.0 has
#: ``Qwen3_5TextConfig`` and ``Qwen3_5ForCausalLM``).
MIN_TRANSFORMERS = (5, 2)
#: The version this workload was developed and measured against.
TESTED_TRANSFORMERS = "5.16.1"

MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"

CLASSES = ModelClasses(
    config_class="transformers:Qwen3_5TextConfig",
    model_class="transformers:Qwen3_5ForCausalLM",
    min_transformers=MIN_TRANSFORMERS,
    tested_transformers=TESTED_TRANSFORMERS,
    extra="transformers>=5.2",
    label="Qwen3.5",
)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    return _common.require_transformers(CLASSES)


def build_config(spec: WorkloadSpec):
    """A ``Qwen3_5TextConfig`` carrying every published field and the run settings.

    ``layer_types``, ``rope_parameters`` (interleaved MRoPE, partial factor
    0.25, theta 1e7) and ``attn_output_gate`` go through as written; the shared
    builder deep-copies the architecture first, so Transformers' in-place RoPE
    normalization cannot reach the frozen declaration.
    """
    return _common.build_config(spec, CLASSES)


def build_model(spec: WorkloadSpec):
    """The reference model, on ``spec.device``, in train mode.

    Random initialisation goes through the shared builder; a spec whose
    ``weights`` name a pinned checkpoint loads it (:func:`_load_pretrained`)
    and leaves it in the state the shared builder leaves a model in.
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
    from huggingface_hub import snapshot_download
    from evograd.benchmark.topdown.common.data_paths import hf_cache_dir

    repo, revision = spec.pretrained_repo
    return snapshot_download(
        repo, revision=revision, local_files_only=True, cache_dir=hf_cache_dir("qwen3_5_4b"),
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])


def _load_pretrained(spec: WorkloadSpec):
    """The checkpoint's text model, validated against the declared architecture.

    Every declared field must equal the loaded config, and every parameter
    must be in ``spec.dtype``: the checkpoint stores ``A_log`` and the gated
    norm weights in float32, and the declared cases are bfloat16.
    """
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(
        pretrained_snapshot(spec), dtype=DTYPES[spec.dtype],
        attn_implementation=spec.attn_implementation, use_cache=spec.use_cache)
    mismatched = {key: (getattr(model.config, key, None), value)
                  for key, value in spec.arch.items() if getattr(model.config, key, None) != value}
    if mismatched:
        raise RuntimeError("the pinned checkpoint's architecture disagrees with the declared "
                           f"workload: {mismatched}")
    other = sorted({str(p.dtype) for p in model.parameters()} - {str(DTYPES[spec.dtype])})
    if other:
        raise RuntimeError(f"parameters loaded in {other}, spec says {spec.dtype}")
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    if spec.arch.get("tie_word_embeddings") and not tied:
        raise RuntimeError("declared tie_word_embeddings=True but the loaded head is untied")
    return model


def modeling():
    require_transformers()
    import importlib

    return importlib.import_module(MODELING)


def dispatch_settings(model=None) -> dict[str, Any]:
    """What ``modeling_qwen3_5``'s dispatched names resolved to at import."""
    from ....common.hybrid.dispatch import kernel_dispatch

    return kernel_dispatch(modeling())


#: The dispatch a training step must run: the torch path, never FLA.
PINNED_DISPATCH = {"gated_delta_rule_dispatch": "torch", "causal_conv1d_dispatch": "torch"}


def qwen3_5_effective_settings(model, spec: WorkloadSpec) -> dict[str, Any]:
    """The shared read-back plus the kernel dispatch."""
    out = effective_settings(model, spec)
    out.update(dispatch_settings(model))
    return out


def check_qwen3_5_settings(effective: dict[str, Any], spec: WorkloadSpec) -> list[str]:
    problems = check_effective_settings(effective, spec)
    problems += [f"{key} is {effective.get(key)!r}, pinned {want!r}"
                 for key, want in PINNED_DISPATCH.items() if effective.get(key) != want]
    return problems


__all__ = ["CLASSES", "DTYPES", "MIN_TRANSFORMERS", "MODELING", "PINNED_DISPATCH",
           "TESTED_TRANSFORMERS", "build_config", "build_model", "check_qwen3_5_settings",
           "dispatch_settings", "make_inputs", "modeling", "pretrained_snapshot",
           "qwen3_5_effective_settings", "require_transformers", "training_step"]
