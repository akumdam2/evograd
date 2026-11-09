"""Binding a model's observed cases onto a reusable primitive contract.

A Level-1 primitive declares mathematics: what an implementation must compute.
It does not know how often some model calls it,
at which shapes, in which memory layout, or which fused task it composes into.
Those facts come from one model's captured training step, and they belong to
that model's benchmark package.

This module is the mechanism, and only the mechanism. It knows how to read a
workload's frozen snapshot, rebuild the cases the harvest recorded, and attach
them to a primitive contract -- in the declared order, under the declared suite
name. It does not know which models exist,
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

from evograd.opdecl import OpDecl, Provenance, Workload, bind_shape
from evograd.ops._common import contiguous_stride, recorded_layout, view_stride


def recorded_strides(op: OpDecl | None, dims: dict[str, int], entries, layout: str):
    """The exact strides a harvested call gave the primitive's tensor arguments.

    The harvested boundary's tensors are paired with the primitive's tensor
    arguments by position. A pair counts only when the recorded tensor *is* the
    argument at this workload's shape -- the same shape, or a higher-rank view
    the rank adapter would flatten into it without a copy (``view_stride``);
    anything else (a boundary whose argument list is not the primitive's) is
    skipped rather than guessed. Contiguous tensors are left out, and so are
    head-major views when the primary input is one, because the generators
    already reproduce those; what remains is every layout a generator would
    otherwise get wrong.
    """
    if op is None:
        return ()
    tensor_args = [arg for arg in op.args if getattr(arg, "shape", None) not in (None, "unspecified")]
    found = []
    for arg, entry in zip(tensor_args, entries or ()):
        try:
            target = bind_shape(arg.shape, dims)
        except ValueError:
            continue
        stride = view_stride(entry.get("shape", ()), entry.get("stride", ()), target)
        if stride is None or stride == contiguous_stride(target):
            continue
        if (layout == "head_major_view"
                and recorded_layout(target, stride) == "head_major_view"):
            continue
        found.append((arg.name, tuple(stride)))
    return tuple(found)


def observed_workloads(
    workload_name: str,
    task: str,
    *,
    dtype: str | None = None,
    op: OpDecl | None = None,
) -> tuple[Workload, ...]:
    """Every observed configuration of one generic Level-1 task, for one workload.

    Read from that workload's tracked snapshot, which derives them from its
    Level-3 harvest: dims, dtype, roles, frequency and memory layout all come
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
    workloads = []
    for config in entry["configurations"]:
        case_dtype = dtype or config["dtype"].replace("torch.", "")
        # The layout of the operator's primary activation input. Both inputs a
        # generator branches on -- RoPE's `x`, attention's q/k/v -- are the
        # leading argument at every harvested boundary.
        primary = (config.get("inputs") or [{}])[0]
        layout = recorded_layout(primary.get("shape", ()), primary.get("stride", ()))
        # Exact strides for every argument a generator would otherwise lay out
        # wrong (``op`` names the arguments; without it, none are recorded).
        strides = recorded_strides(op, dict(config["dims"]), config.get("inputs"), layout)
        first = next((a.name for a in (op.args if op is not None else ())
                      if getattr(a, "shape", None) not in (None, "unspecified")), None)
        if any(name == first for name, _ in strides):
            layout = "strided"
        workloads.append(
            Workload(
                dims=dict(config["dims"]),
                dtype=case_dtype,
                provenance=Provenance(
                    model=workload_name,
                    component=config["provenance"]["component"],
                    free=dict(config["provenance"]["free"]),
                    source=config["provenance"].get("source", "hf_config"),
                    layout=layout,
                    strides=strides,
                ),
            )
        )
    return tuple(workloads)


def observed_workloads_if_harvested(
    workload_name: str,
    task: str,
    **kwargs,
) -> tuple[Workload, ...]:
    """:func:`observed_workloads`, or ``()`` when that workload has no snapshot.

    A workload package can exist long before anyone has run its harvest -- the
    snapshot is *derived* from a GPU run, not authored -- and a registry must
    stay importable in the meantime. This is the form a second architecture
    uses: its suite appears the moment the snapshot is tracked, and no
    declaration needs editing to make that happen.

    Distinct from a bare ``try/except`` at each call site, which would also
    swallow a *corrupt* snapshot. Only "not harvested yet" is tolerated here; a
    snapshot that exists and fails its hash check still raises, because a
    silently-empty suite and a broken one are different problems.
    """
    from evograd.benchmark.topdown import has_snapshot

    if not has_snapshot(workload_name):
        return ()
    return observed_workloads(workload_name, task, **kwargs)


@dataclasses.dataclass(frozen=True)
class ObservedBinding:
    """One model's observed cases for one primitive.

    ``coverage`` says where the observed cases join the coverage the
    primitive has collected so far (earlier models' observed cases). The two
    orders are not interchangeable in a report that lists cases in declaration
    order, so the position each binding used is recorded here rather than
    normalised away.
    """

    #: The harvested workload whose snapshot these cases come from. Carried on
    #: the binding rather than passed alongside it, so a configuration is
    #: self-describing and the binder never has to be told which model it is
    #: looking at.
    workload: str
    task: str
    suite: str
    coverage: str = "append"  # "append" or "prepend"

    def workloads(self, primitive: OpDecl) -> tuple[Workload, ...]:
        """The cases this binding contributes, read from the frozen snapshot.

        Empty when that workload has not been harvested yet, so a model package
        can exist before its snapshot does. A snapshot that exists and fails its
        hash check still raises.

        Overridable, which is the seam a test uses to exercise the binder with
        supplied cases rather than a registered workload.
        """
        return observed_workloads_if_harvested(self.workload, self.task, op=primitive)


def index_bindings(
    bindings: Iterable[ObservedBinding],
) -> dict[str, tuple[ObservedBinding, ...]]:
    """``{primitive: bindings}``, in the order the configurations were supplied.

    Two models binding the same primitive is the ordinary case, not a
    collision: two architectures may both run the same mathematics, and each
    contributes its own observed suite under its own name. What is refused is a
    real collision --
    two bindings claiming the same primitive *and* the same suite name, which
    would make the suite ambiguous about whose widths it holds.

    Ordering is by supply order, so a caller that always passes the same
    configurations in the same order always gets the same suites in the same
    place. The registry supplies them in a fixed order for exactly that reason.
    """
    indexed: dict[str, list[ObservedBinding]] = {}
    seen: dict[tuple[str, str], ObservedBinding] = {}
    for binding in bindings:
        identity = (binding.task, binding.suite)
        clash = seen.get(identity)
        if clash is not None:
            raise ValueError(
                f"workloads {clash.workload!r} and {binding.workload!r} both bind "
                f"observed cases to primitive {binding.task!r} under the suite name "
                f"{binding.suite!r}; a suite name must identify which model observed "
                f"the cases in it"
            )
        seen[identity] = binding
        indexed.setdefault(binding.task, []).append(binding)
    return {task: tuple(items) for task, items in indexed.items()}


def bind_observed_cases(
    primitive: OpDecl, bindings: Iterable[ObservedBinding] = ()
) -> OpDecl:
    """Return the executable task for one primitive, with its observed cases.

    Every binding that names this primitive is applied, in supply order, so two
    architectures can each contribute a suite to the same contract. A primitive
    none of ``bindings`` names is returned unchanged, which is what makes this
    safe to apply across the whole registry.

    A binding whose workload has no tracked snapshot contributes nothing and is
    skipped: a workload package exists before its harvest has been run, and the
    suite appears the moment the snapshot is tracked. That is not the same as
    tolerating a broken snapshot -- one that exists and fails its hash check
    still raises, from :func:`observed_workloads_if_harvested`.
    """
    applicable = index_bindings(bindings).get(primitive.name)
    if not applicable:
        return primitive

    coverage = primitive.coverage
    suites = dict(primitive.benchmark_suites)
    contributed = False

    for binding in applicable:
        observed = binding.workloads(primitive)
        if not observed:
            continue
        contributed = True
        coverage = (
            observed + coverage
            if binding.coverage == "prepend"
            else coverage + observed
        )
        if binding.suite in suites:
            raise ValueError(
                f"{primitive.name}: suite {binding.suite!r} is already declared on the "
                f"primitive; observed cases must be bound in exactly one place"
            )
        suites[binding.suite] = observed

    if not contributed:
        return primitive

    task = dataclasses.replace(primitive, coverage=coverage, benchmark_suites=suites)
    task.validate()
    return task


__all__ = [
    "ObservedBinding",
    "bind_observed_cases",
    "index_bindings",
    "observed_workloads",
    "observed_workloads_if_harvested",
]
