"""Binding a model's observed cases onto a reusable primitive contract.

A Level-1 primitive declares mathematics: what an implementation must compute
and how to prove it correct. It does not know how often some model calls it,
at which shapes, in which memory layout, or which fused task it composes into.
Those facts come from one model's captured training step, and they belong to
that model's benchmark package.

This module is the mechanism, and only the mechanism. It knows how to read a
workload's frozen snapshot, rebuild the cases the harvest recorded, and attach
them to a primitive contract -- in the declared order, under the declared suite
name, with the declared coverage mirror. It does not know which models exist,
which primitives any of them binds, or what any of them is called: the
configuration arrives as an argument.

Each model owns its own configuration, in its manifest; a
:class:`ObservedBinding` says what to bind and this module says how.
:mod:`evograd.benchmark.core.registry` is where the two are put together, and
is the one place that names the models there are.

Two consequences worth stating, because they are the point of the seam:

* the primitive is never mutated -- binding returns a new
  :class:`~evograd.opdecl.OpDecl`, so the reusable contract stays reusable, and
  nothing under :mod:`evograd.ops` imports a snapshot, a workload, or this
  module;
* a second harvested architecture binds its own cases to the same primitives by
  writing a table in its own manifest, without editing a primitive package or
  this file.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from evograd.opdecl import OpDecl, Provenance, Workload
from evograd.ops._common import recorded_layout


def observed_workloads(
    workload_name: str,
    task: str,
    *,
    tolerances: dict[str, tuple[float, float]] | None = None,
    dtype: str | None = None,
) -> tuple[Workload, ...]:
    """Every observed configuration of one generic Level-1 task, for one workload.

    Read from that workload's tracked snapshot, which derives them from its
    Level-4 harvest: dims, dtype, roles, frequency and memory layout all come
    from what the canonical step ran. Nothing here is a second hand-written
    copy, and the provenance each workload carries re-derives the same dims from
    the published model configuration, so ``tests/test_provenance`` proves the
    two agree.

    ``workload_name`` selects the snapshot; it is also the provenance's model
    key, so a second harvested architecture reaches this function unchanged.

    A snapshot is a small tracked JSON file with no torch dependency, so this
    stays importable on a machine that has never run the workload.
    """
    from evograd.benchmark.topdown import load_snapshot

    entry = load_snapshot(workload_name)["level1"][task]
    tolerances = tolerances or {}
    workloads = []
    for config in entry["configurations"]:
        case_dtype = dtype or config["dtype"].replace("torch.", "")
        atol, rtol = tolerances.get(case_dtype, (None, None))
        # The layout of the operator's primary activation input. Both inputs a
        # generator branches on -- RoPE's `x`, attention's q/k/v -- are the
        # leading argument at every harvested boundary.
        primary = (config.get("inputs") or [{}])[0]
        workloads.append(
            Workload(
                dims=dict(config["dims"]),
                dtype=case_dtype,
                atol=atol,
                rtol=rtol,
                provenance=Provenance(
                    model=workload_name,
                    component=config["provenance"]["component"],
                    free=dict(config["provenance"]["free"]),
                    source="hf_config",
                    layout=recorded_layout(
                        primary.get("shape", ()), primary.get("stride", ())
                    ),
                ),
            )
        )
    return tuple(workloads)


@dataclasses.dataclass(frozen=True)
class ObservedBinding:
    """One model's observed cases for one primitive.

    ``coverage`` says where the observed cases join the primitive's own untimed
    coverage. The two orders are not interchangeable in a report that lists
    cases in declaration order, so the position each primitive used before the
    cases moved out is recorded here rather than normalised away.
    """

    #: The harvested workload whose snapshot these cases come from. Carried on
    #: the binding rather than passed alongside it, so a configuration is
    #: self-describing and the binder never has to be told which model it is
    #: looking at.
    workload: str
    task: str
    suite: str
    #: Whether the observed cases carry the primitive's own declared tolerances.
    #: A case that names none leaves ``atol``/``rtol`` unset and is gated by the
    #: declaration's defaults; the two primitives that set them here did so
    #: before the cases moved out, and the value is theirs, not this layer's.
    declared_tolerances: bool = False
    coverage: str = "append"  # "append" or "prepend"
    #: A suite that serves the primitive's untimed coverage under a name. It has
    #: to follow the coverage it mirrors, or the same cases would be reported
    #: two different ways. Named explicitly rather than detected by comparing
    #: tuples: after the observed cases move out, an unrelated suite can happen
    #: to equal the shortened coverage, and extending it would invent cases.
    mirrors_coverage: str | None = None

    def workloads(self, primitive: OpDecl) -> tuple[Workload, ...]:
        """The cases this binding contributes, read from the frozen snapshot.

        Overridable, which is the seam a test uses to exercise the binder with
        supplied cases rather than a registered workload.
        """
        return observed_workloads(
            self.workload,
            self.task,
            tolerances=primitive.tolerances if self.declared_tolerances else None,
        )


def index_bindings(bindings: Iterable[ObservedBinding]) -> dict[str, ObservedBinding]:
    """``{primitive: binding}``, refusing two workloads claiming one primitive.

    A primitive can only serve one suite of observed cases under one name, so
    two workloads binding to it is a configuration error rather than something
    to merge -- and it has to be caught where the configurations meet, not in
    whichever one happened to be read second.
    """
    indexed: dict[str, ObservedBinding] = {}
    for binding in bindings:
        existing = indexed.get(binding.task)
        if existing is not None:
            raise ValueError(
                f"workloads {existing.workload!r} and {binding.workload!r} both bind "
                f"observed cases to primitive {binding.task!r}; a suite name must "
                f"identify which model observed them"
            )
        indexed[binding.task] = binding
    return indexed


def bind_suite_cases(primitive: OpDecl) -> OpDecl:
    """Attach the operator suite's performance cases to a primitive contract.

    The primitive declares no timed grid, no untimed benchmark coverage, no
    named suite, no regime split and no case weighting -- those are benchmark
    decisions and live in :mod:`evograd.benchmark.operator_suite.cases`. This
    is where the two meet, and it returns a new declaration rather than
    mutating the one the registry of primitives serves.

    A primitive the suite defines no cases for is returned unchanged.
    """
    from evograd.benchmark.operator_suite.cases import cases_for

    suite = cases_for(primitive.name)
    if suite is None:
        return primitive
    for field, declared in (("benchmark", primitive.benchmark),
                            ("coverage", primitive.coverage),
                            ("benchmark_suites", primitive.benchmark_suites)):
        if declared:
            raise ValueError(
                f"{primitive.name}: the primitive declares {field}, and the "
                f"operator suite also defines cases for it. A case collection "
                f"has one owner."
            )
    task = dataclasses.replace(
        primitive,
        benchmark=tuple(suite.benchmark),
        coverage=tuple(suite.coverage),
        benchmark_suites={k: tuple(v) for k, v in suite.suites.items()},
        regime_feature=suite.regime_feature if suite.regime_feature is not None
        else primitive.regime_feature,
        regime_split=suite.regime_split if suite.regime_split is not None
        else primitive.regime_split,
        case_weight=suite.case_weight if suite.case_weight is not None
        else primitive.case_weight,
    )
    task.validate()
    return task


def bind_benchmark_cases(
    primitive: OpDecl, observed: Iterable[ObservedBinding] = ()
) -> OpDecl:
    """The executable task for one primitive: suite cases, then observed ones.

    Order matters. The suite's cases establish the coverage a model's observed
    cases then join, and the suite that mirrors that coverage has to follow the
    result rather than the intermediate.

    ``observed`` is supplied by the caller. This module knows how to bind a
    configuration; which primitives a model binds, under which suite name and
    in which order, belongs to that model.
    """
    return bind_observed_cases(bind_suite_cases(primitive), observed)


def bind_observed_cases(
    primitive: OpDecl, bindings: Iterable[ObservedBinding] = ()
) -> OpDecl:
    """Return the executable task for one primitive, with its observed cases.

    A primitive none of ``bindings`` names is returned unchanged, which is what
    makes this safe to apply across the whole registry.
    """
    binding = index_bindings(bindings).get(primitive.name)
    if binding is None:
        return primitive
    observed = binding.workloads(primitive)
    if not observed:
        return primitive
    coverage = (
        observed + primitive.coverage
        if binding.coverage == "prepend"
        else primitive.coverage + observed
    )
    suites = dict(primitive.benchmark_suites)
    if binding.suite in suites:
        raise ValueError(
            f"{primitive.name}: suite {binding.suite!r} is already declared on the "
            f"primitive; observed cases must be bound in exactly one place"
        )
    suites[binding.suite] = observed
    if binding.mirrors_coverage is not None:
        if binding.mirrors_coverage not in suites:
            raise ValueError(
                f"{primitive.name}: no suite named {binding.mirrors_coverage!r} to "
                f"follow the coverage it mirrors"
            )
        suites[binding.mirrors_coverage] = coverage
    task = dataclasses.replace(primitive, coverage=coverage, benchmark_suites=suites)
    task.validate()
    return task


__all__ = [
    "ObservedBinding",
    "bind_benchmark_cases",
    "bind_observed_cases",
    "bind_suite_cases",
    "index_bindings",
    "observed_workloads",
]
