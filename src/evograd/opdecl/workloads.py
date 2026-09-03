"""Whole-model workload declarations: the level-4 tasks of the benchmark.

Levels 1–3 are :class:`~evograd.opdecl.activity.OpDecl`\\ s — forward/backward
pairs with a declared argument list, one active output, and a saved-state
contract. A level-4 task is not that shape. A whole-model training step has no
single output, nothing crossing a pair boundary, and it is not evolved as one
kernel: evolved level-1/2 kernels are patched *into* it, and the task measures
whether their local speedups survive the surrounding computation.

So level 4 gets its own declaration type. A :class:`WorkloadDecl` names the
model configuration the step trains (a registry key in
:mod:`evograd.opdecl.models`, so shapes stay provenance-checked exactly as they
are at levels 1–3), the patch sites and which declared operator each one
accepts, and the benchmark cases. The torch-facing construction — building the
model, making the batch, running the step — lives behind the ``factory``
reference, imported only when a run actually needs it, which keeps this module
importable on machines without torch like every other declaration.

Measurement is the tier-3 protocol (``bench/tier3_runner``): a full training
step, every provider on identical weights and batches. Tier and level stay
orthogonal — this module says *what* the task is, never how carefully it is
timed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evograd.opdecl.activity import Workload


@dataclass(frozen=True)
class WorkloadDecl:
    """Complete declaration of one whole-model training-step task."""

    #: Registry key, and what a report names.
    name: str
    #: ``module.path:callable`` building the torch-side training workload.
    #: The callable receives one benchmark :class:`Workload` (its ``dims`` are
    #: the shape of the run) plus ``device`` and ``seed`` keyword arguments, and
    #: returns an object satisfying ``bench.tier3_model.TrainingWorkload``.
    factory: str
    #: Aggregation group for the suite report, as on ``OpDecl``.
    family: str
    #: The model configuration this step trains: a key in
    #: ``evograd.opdecl.models.MODELS``. Provenance hangs off it — every
    #: benchmark case's dims must re-derive from this configuration.
    model: str
    #: Patch sites: ``{site_name: declared operator name}``. An evolved
    #: program is only accepted for a site whose operator it implements, and a
    #: run reports which sites it actually replaced.
    sites: dict[str, str]
    #: Timed cases. Each carries a provenance naming ``model`` and a
    #: ``<component>_dims`` method, exactly as level-1..3 benchmark cases do.
    benchmark: tuple[Workload, ...]
    #: What the workload does not exercise, stated rather than discovered.
    #: Required: a level-4 task named for a model while training a subset of it
    #: would be worse than the restriction it hides.
    exclusions: str = ""
    #: Pip distribution the factory needs at runtime (e.g. the model
    #: implementation). When missing, the task is reported uncovered — "we did
    #: not run it" and "it has no speedup" are different claims.
    requires: str = ""
    #: Reviewed baseline names patchable into the sites, when wired.
    baselines: tuple[str, ...] = ()
    notes: str = ""
    #: Task hierarchy level. Whole-model is the only shape this type declares.
    level: int = 4

    def __post_init__(self) -> None:
        if self.level != 4:
            raise ValueError(
                f"{self.name}: WorkloadDecl declares whole-model tasks; level "
                f"must be 4, got {self.level!r} (levels 1-3 are OpDecls)"
            )
        if not self.name or not self.family or not self.model:
            raise ValueError("a workload declaration needs name, family, and model")
        if ":" not in self.factory:
            raise ValueError(
                f"{self.name}: factory must be 'module.path:callable', "
                f"got {self.factory!r}"
            )
        if not self.sites:
            raise ValueError(
                f"{self.name}: a level-4 task with no patch sites measures "
                "nothing a kernel can change"
            )
        if not self.benchmark:
            raise ValueError(f"{self.name}: declare at least one benchmark case")
        for workload in self.benchmark:
            if workload.provenance is None:
                raise ValueError(
                    f"{self.name}: every level-4 benchmark case must carry a "
                    "provenance; a whole-model shape with no model behind it "
                    "is a contradiction"
                )
            if workload.provenance.model != self.model:
                raise ValueError(
                    f"{self.name}: benchmark case cites model "
                    f"{workload.provenance.model!r} but the declaration trains "
                    f"{self.model!r}"
                )
        if not self.exclusions:
            raise ValueError(
                f"{self.name}: state what the workload does not exercise "
                "(exclusions), even if that is 'nothing'"
            )

    def resolve_factory(self):
        """Import and return the torch-side workload factory."""
        import importlib

        module_path, _, attribute = self.factory.partition(":")
        module = importlib.import_module(module_path)
        return getattr(module, attribute)


def declare_workload(**kwargs) -> WorkloadDecl:
    """Construct and validate a whole-model workload declaration."""
    return WorkloadDecl(**kwargs)
