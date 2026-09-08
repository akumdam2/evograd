"""One workload's Level-4 declaration: everything the shared machinery asks it.

Level 4 is the reference execution -- one model, one training step, run the way
training runs it. *Performing* that step, describing it, observing it and
harvesting it are the same procedure for every decoder-only causal LM, and live
in the sibling modules here. What differs is a short list of facts, and this is
the object that carries them:

    which architecture          ``spec_type`` -- its field defaults are the
                                canonical run
    which Transformers classes  ``classes``
    which boundaries to watch   ``plan``
    what to call the artifacts  ``smoke_schema``, ``manifest_schema``
    which call sites were patched ``function_wrappers``, recorded in the
                                capture scope so it describes what actually ran

Bundled rather than passed as eight arguments because every stage needs a
different subset and threading them separately is how two of them end up
disagreeing about which model is being described.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ModelClasses
from .observe import ObservationPlan
from .spec import WorkloadSpec


@dataclass(frozen=True)
class Level4Workload:
    """A model, declared once, for every Level-4 stage to read."""

    #: Registry key; also the ``Provenance.model`` key. e.g. ``"qwen3_0_6b"``.
    name: str
    #: Human-readable, for reports and banners. e.g. ``"Qwen3-0.6B"``.
    label: str
    #: The spec subclass whose field defaults are this workload's canonical run.
    spec_type: type[WorkloadSpec]
    classes: ModelClasses
    plan: ObservationPlan
    smoke_schema: str
    manifest_schema: str
    #: The exact call sites the observer patches, recorded in the capture scope.
    function_wrappers: tuple[str, ...]
    #: Dotted path of the workload package, for ``python -m`` in help text.
    package: str

    @property
    def canonical(self) -> WorkloadSpec:
        """The reference execution: the spec type's own defaults."""
        return self.spec_type()

    def resolve(self, spec: WorkloadSpec | None) -> WorkloadSpec:
        return (spec if spec is not None else self.canonical).validate()

    # ── the three things every stage does with a model ──────────────────────

    def require_transformers(self):
        from . import model as _model

        return _model.require_transformers(self.classes)

    def build_model(self, spec: WorkloadSpec):
        from . import model as _model

        return _model.build_model(spec, self.classes)

    def make_inputs(self, spec: WorkloadSpec):
        from . import model as _model

        return _model.make_inputs(spec)
