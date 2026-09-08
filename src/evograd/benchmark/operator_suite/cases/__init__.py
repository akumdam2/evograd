"""The operator suite's performance cases, one module per primitive.

A Level-1 primitive declares mathematics: what an implementation must compute,
and the generic cases that prove it correct. It does not decide which shapes
are worth *timing*. That is a benchmark question -- the grid comes from a
published model configuration, the regime split says where small stops and
large begins, the weighting says how much each case counts, and the named
suites are what a report and a CLI select by. All of it belongs here.

Every module in this package exposes ``cases = SuiteCases(...)`` for exactly
one primitive, named after it. :mod:`evograd.benchmark.cases` binds those onto
the primitive contract when the task registry is built, so the primitive itself
is never mutated and never imports a model configuration.

Two things this package is *not*:

* it is not a second declaration -- a module here carries cases, never a
  contract, a reference, or a tolerance policy;
* it is not top-down coverage. A grid whose dimensions were computed from a
  published Llama-3 or AlphaFold3 configuration is an operator-suite grid, and
  saying so is the point of it living here rather than under
  :mod:`evograd.benchmark.topdown`. A case observed in a real captured run is a
  different kind of evidence and is bound from that model's manifest.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from typing import Any, Callable

from evograd.opdecl import Workload


@dataclasses.dataclass(frozen=True)
class SuiteCases:
    """One primitive's performance cases, as the operator suite defines them.

    Every field mirrors the ``declare_op`` argument it is bound to, so a reader
    comparing this against the declaration it came from sees the same names.
    A field left at its default means the primitive declared none.
    """

    #: The timed grid.
    benchmark: tuple[Workload, ...] = ()
    #: Untimed benchmark coverage. Distinct from the primitive's ``correctness``
    #: cases, which prove an implementation right and stay with the mathematics.
    coverage: tuple[Workload, ...] = ()
    #: Named selections a report or a CLI can ask for, in declaration order.
    suites: dict[str, tuple[Workload, ...]] = dataclasses.field(default_factory=dict)
    #: Which dimension separates the shape regimes, and where they split.
    #: The two are one setting: a feature without a split describes nothing,
    #: and ``OpDecl`` rejects either alone.
    regime_feature: Callable[[Workload], float] | None = None
    regime_split: int | None = None
    #: How much each case counts when results are pooled.
    case_weight: Callable[[Workload], float] | None = None


def _module_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            info.name
            for info in pkgutil.iter_modules(__path__)
            if not info.name.startswith("_")
        )
    )


def cases_for(task: str) -> SuiteCases | None:
    """The suite's performance cases for one primitive, or ``None``."""
    if task not in _module_names():
        return None
    module = importlib.import_module(f"{__name__}.{task}")
    found: Any = getattr(module, "cases", None)
    if not isinstance(found, SuiteCases):
        raise TypeError(f"{module.__name__}.cases must be SuiteCases")
    return found


def all_cases() -> dict[str, SuiteCases]:
    """Every primitive the suite defines performance cases for."""
    return {name: cases_for(name) for name in _module_names()}


__all__ = ["SuiteCases", "all_cases", "cases_for"]
