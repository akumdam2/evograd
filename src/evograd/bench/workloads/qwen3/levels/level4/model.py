"""Which Transformers classes Qwen3 builds, bound to the shared builder.

Building a model from a spec, and reading back what was actually built rather
than what was asked for, is the same procedure for every decoder-only causal LM
and lives in :mod:`....common.model`. Qwen3's contribution is two class names and
a version floor.

Transformers is an optional dependency. Nothing here imports it at module import
time, so ``import evograd.bench.workloads.qwen3`` works on a machine that has
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
    """The randomly-initialised reference model, on ``spec.device``, in train mode."""
    return _common.build_model(spec, CLASSES)
