"""Which Transformers classes Qwen3.5 builds, bound to the shared builder.

Building a model from a spec and reading back what was built is the same
procedure for every decoder-only causal LM and lives in
:mod:`....common.model`. Qwen3.5's contribution is two class names, a version
floor, and one more thing to read back: the import-time kernel dispatch of
``modeling_qwen3_5`` (trap T-dispatch). ``torch_chunk_gated_delta_rule`` and
``causal_conv1d_fn`` pick FLA / ``causal_conv1d`` when the module is imported
if those packages are importable; :func:`dispatch_settings` reports what they
resolved to, and the declaration pins both to ``"torch"``.

Only random weights are built. ``Qwen3_5ForCausalLM._from_config`` runs
``post_init``, so every parameter is initialized by the model's own
``_init_weights`` (``dt_bias = 1``, ``A_log = log U(0.01, 16)``, ``(1 + w)``
norms at 0) under ``spec.seed``, on the CPU.

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
    """The randomly-initialised reference model, on ``spec.device``, in train mode."""
    if spec.pretrained or spec.real_text:
        raise NotImplementedError(
            "Qwen3.5-4B implements random weights and synthetic tokens only; "
            f"weights={spec.weights!r} data={spec.data!r} are reserved")
    return _common.build_model(spec, CLASSES)


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
           "dispatch_settings", "make_inputs", "modeling", "qwen3_5_effective_settings",
           "require_transformers", "training_step"]
