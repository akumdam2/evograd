"""Which Transformers classes OLMoE builds, bound to the shared builder.

Building a model from a spec and reading back what was built is the same
procedure for every decoder-only causal LM and lives in
:mod:`....common.model`. OLMoE's contribution is two class names, a version
floor, and one more thing to read back: which experts kernel every
``OlmoeExperts`` module resolved to. The shared read-back already reports the
model-level ``experts_implementation`` and holds it to the spec; the per-module
view here is what would show a partial dispatch.

Only random weights are built. ``OlmoeForCausalLM._from_config`` runs
``post_init``, so every parameter is initialized by the model's own
``_init_weights`` (linears, the router and the 3-D expert tensors at
``normal(0, 0.02)``, norms at 1) under ``spec.seed``, on the CPU.

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

#: OLMoE has been in Transformers since 4.45, but the sites here address the
#: Transformers-5 layout -- ``OlmoeExperts`` holding every expert in one
#: ``[E, 2I, H]`` and one ``[E, H, I]`` tensor, dispatched through
#: ``experts_implementation`` -- which the 4.x ``nn.ModuleList`` of per-expert
#: MLPs does not have. 5.16.1 is the only release this package was run
#: against; a 5.x release older than the experts interface fails at build with
#: Transformers' own error rather than silently running a different model.
MIN_TRANSFORMERS = (5, 0)
#: The version this workload was developed and measured against.
TESTED_TRANSFORMERS = "5.16.1"

MODELING = "transformers.models.olmoe.modeling_olmoe"

CLASSES = ModelClasses(
    config_class="transformers:OlmoeConfig",
    model_class="transformers:OlmoeForCausalLM",
    min_transformers=MIN_TRANSFORMERS,
    tested_transformers=TESTED_TRANSFORMERS,
    extra="evograd[olmoe]",
    label="OLMoE",
)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    return _common.require_transformers(CLASSES)


def build_config(spec: WorkloadSpec):
    """An ``OlmoeConfig`` carrying every published field and the run settings.

    The shared builder deep-copies the architecture first (Transformers
    normalizes ``rope_theta`` into ``rope_parameters`` in place) and passes the
    spec's ``experts_implementation`` through as it passes the attention
    backend.
    """
    return _common.build_config(spec, CLASSES)


def build_model(spec: WorkloadSpec):
    """The reference model, on ``spec.device``, in train mode.

    Random initialisation goes through the shared builder; pinned weights are
    loaded (:func:`_load_pretrained`) with the spec's experts kernel and
    ``output_router_logits`` pinned on, as the declared architecture says.
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
    return snapshot_download(repo, revision=revision, local_files_only=True,
                             cache_dir=hf_cache_dir("olmoe_1b_7b"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])


def _load_pretrained(spec: WorkloadSpec):
    """The pinned checkpoint, validated against the declared architecture."""
    from transformers import OlmoeForCausalLM

    from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import rope_settings

    model = OlmoeForCausalLM.from_pretrained(
        pretrained_snapshot(spec), dtype=DTYPES[spec.dtype],
        attn_implementation=spec.attn_implementation,
        experts_implementation=spec.experts_implementation, use_cache=spec.use_cache,
        output_router_logits=spec.arch["output_router_logits"])
    loaded = model.config
    rope_keys = {"rope_theta", "rope_scaling"}
    mismatched = {key: (getattr(loaded, key, None), value)
                  for key, value in spec.arch.items()
                  if key not in rope_keys and getattr(loaded, key, None) != value}
    if rope_settings(loaded) != rope_settings(dict(spec.arch)):
        mismatched["rope"] = (rope_settings(loaded), rope_settings(dict(spec.arch)))
    if mismatched:
        raise RuntimeError("the pinned checkpoint's architecture disagrees with the declared "
                           f"workload: {mismatched}")
    other = sorted({str(p.dtype) for p in model.parameters()} - {str(DTYPES[spec.dtype])})
    if other:
        raise RuntimeError(f"parameters loaded in {other}, spec says {spec.dtype}")
    return model


def modeling():
    require_transformers()
    import importlib

    return importlib.import_module(MODELING)


def experts_settings(model=None) -> dict[str, Any]:
    """What every experts module actually dispatches to, read off the built model.

    ``experts_implementation_per_module`` lists the distinct values; more than
    one would mean a partially dispatched model, which the shared read-back
    (which looks at the root config only) could not see.
    """
    import torch

    per_module = sorted({
        str(impl)
        for module in (model.modules() if model is not None else ())
        if hasattr(module, "gate_up_proj")
        for impl in [getattr(getattr(module, "config", None), "_experts_implementation", None)]
    })
    return {
        "experts_implementation_per_module": per_module,
        "grouped_mm_available": bool(hasattr(torch.nn.functional, "grouped_mm")
                                     or hasattr(torch, "_grouped_mm")),
    }


def olmoe_effective_settings(model, spec: WorkloadSpec) -> dict[str, Any]:
    """The shared read-back plus the per-module experts dispatch."""
    out = effective_settings(model, spec)
    out.update(experts_settings(model))
    return out


def check_olmoe_settings(effective: dict[str, Any], spec: WorkloadSpec) -> list[str]:
    """The shared differences, plus a partial experts dispatch."""
    problems = check_effective_settings(effective, spec)
    per_module = effective.get("experts_implementation_per_module") or []
    if per_module and per_module != [spec.experts_implementation]:
        problems.append(f"experts modules dispatch to {per_module}, requested "
                        f"{spec.experts_implementation!r}")
    return problems


__all__ = ["CLASSES", "DTYPES", "MIN_TRANSFORMERS", "MODELING", "TESTED_TRANSFORMERS",
           "build_config", "build_model", "check_olmoe_settings", "experts_settings",
           "make_inputs", "modeling", "olmoe_effective_settings", "pretrained_snapshot",
           "require_transformers", "training_step"]
