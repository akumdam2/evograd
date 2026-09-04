"""Which Transformers classes Llama-3 builds, bound to the shared builder.

Building a model from a spec, and reading back what was actually built rather
than what was asked for, is the same procedure for every decoder-only causal LM
and lives in :mod:`....common.model`. Llama's contribution is two class names.

No Hub access is needed despite Llama-3 being a gated repository: the workload
constructs from the written-out configuration with random weights, and fetches
neither checkpoint nor tokenizer.
"""

from __future__ import annotations

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

#: Llama has been in Transformers since 4.28, but this is the floor the rest of
#: the harness targets and the only range it has been exercised against; a lower
#: bound this repository has not run is a claim, not a guarantee.
MIN_TRANSFORMERS = (4, 51)
#: The version this workload was developed against.
TESTED_TRANSFORMERS = "5.16.1"

CLASSES = ModelClasses(
    config_class="transformers:LlamaConfig",
    model_class="transformers:LlamaForCausalLM",
    min_transformers=MIN_TRANSFORMERS,
    tested_transformers=TESTED_TRANSFORMERS,
    extra="evograd[llama3]",
    label="Llama-3",
)


def require_transformers():
    """Import Transformers or fail with something the reader can act on."""
    return _common.require_transformers(CLASSES)


def build_config(spec: WorkloadSpec):
    """A ``LlamaConfig`` carrying the spec's architecture and run settings."""
    return _common.build_config(spec, CLASSES)


def build_model(spec: WorkloadSpec):
    """The randomly-initialised reference model, on ``spec.device``, in train mode."""
    return _common.build_model(spec, CLASSES)
