"""One workload's Level-3 declaration: everything the shared machinery asks it.

Level 3 is the reference execution -- one model, one training step, run the way
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
from typing import Any, Callable, Mapping

from .model import ModelClasses
from .observe import ObservationPlan
from .spec import WorkloadSpec


@dataclass(frozen=True)
class Level3Workload:
    """A model, declared once, for every Level-3 stage to read."""

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
    #: What the built model reports beyond the shared settings -- an
    #: import-time kernel dispatch, say -- as ``model -> {name: value}``. Read
    #: back by the smoke run and the harvest, and recorded beside the rest.
    extra_settings: Callable[[Any], dict[str, Any]] | None = None
    #: The values some of those extra settings must hold; a run that reads back
    #: anything else is refused, as a wrong dtype or backend is.
    pinned_settings: Mapping[str, Any] = field(default_factory=dict)
    #: Further checks on a finished observation, ``(observation, spec) -> None``,
    #: raising when the harvest must not be written -- per-layer mandatory
    #: boundaries for a model whose layers are not alike.
    observation_checks: Callable[[Any, WorkloadSpec], None] | None = None
    #: What the harvest deliberately does not watch, for the capture scope;
    #: ``None`` keeps the shared decoder list.
    not_observed: tuple[str, ...] | None = None

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

    def effective_settings(self, model, spec: WorkloadSpec) -> dict[str, Any]:
        """The shared read-back, plus whatever this workload adds to it."""
        from . import model as _model

        effective = _model.effective_settings(model, spec)
        if self.extra_settings is not None:
            effective.update(self.extra_settings(model))
        return effective

    def check_settings(self, effective: dict[str, Any], spec: WorkloadSpec) -> list[str]:
        """The shared differences, plus every pinned extra setting that moved."""
        from . import model as _model

        problems = _model.check_effective_settings(effective, spec)
        problems += [
            f"{key} is {effective.get(key)!r}, pinned {want!r}"
            for key, want in self.pinned_settings.items()
            if effective.get(key) != want
        ]
        return problems
