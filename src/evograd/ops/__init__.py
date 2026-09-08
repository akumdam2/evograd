"""Registry of reusable Level-1 mathematical primitives.

A primitive is registered by adding an ``evograd/ops/level1/<name>/`` package
whose ``__init__.py`` exposes ``op = declare_op(...)``. Its forward reference,
input generator and reviewed baselines live in that same package. No central
list is edited when a primitive is added.

**This package owns mathematics, not benchmarks.** A primitive declares the
contract an implementation must satisfy, a reference that defines it, the
generic cases that prove an implementation correct, and how to build inputs for
them. It declares no timed grid, no benchmark coverage, no named suite, no
shape-regime split and no case weighting: those say which shapes are worth
*measuring* and how much each one counts, which is a benchmark decision even
when the shapes are this primitive's own.

:mod:`evograd.benchmark` attaches them when it builds the task registry --
the suite's performance grids from
:mod:`evograd.benchmark.operator_suite.cases`, a model's observed cases from
its top-down manifest -- returning a new declaration rather than editing the
one served here. That is why nothing in this package imports a workload, a
frozen snapshot, an evaluation policy or the benchmark itself, why importing it
in a fresh interpreter loads none of them, and why the executable task registry
is :data:`evograd.benchmark.TASKS` rather than this one.

Reviewed pair baselines (``liger.py``, ``cublas.py``) stay in the primitive
package. An adapter implements *this* contract and is discovered through the
declaration that names it, so co-locating it is the arrangement, not an
oversight.

Levels above 1 are therefore not found here. A fused task that belongs to no
single model lives in :mod:`evograd.benchmark.operator_suite.tasks`; one whose
identity comes from a captured model lives under that model's package in
:mod:`evograd.benchmark.topdown`.
"""

from __future__ import annotations

import importlib
import os
import pkgutil

from evograd.opdecl import OpDecl
from evograd.opdecl.loader import load_declaration


def _collect(search_path, prefix: str, discovered: dict[str, OpDecl], *, depth: int) -> None:
    for module_info in pkgutil.iter_modules(search_path, prefix):
        if not module_info.ispkg:
            continue
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        op = getattr(module, "op", None)
        if op is None:
            # A grouping package rather than a primitive. Recurse once; deeper
            # nesting is not a layout we use, and allowing it would let a stray
            # package anywhere silently register a primitive.
            if depth > 0:
                _collect(module.__path__, f"{module_info.name}.", discovered, depth=depth - 1)
            continue
        if not isinstance(op, OpDecl):
            raise TypeError(f"{module_info.name}.op must be OpDecl, got {type(op).__name__}")
        if op.name != short_name:
            raise ValueError(
                f"{module_info.name}: declaration name {op.name!r} must match module {short_name!r}"
            )
        if op.level != 1:
            # The registry is the enforcement point, not a comment: a level-2
            # task placed here would be discovered as a primitive and inherit
            # a reusability claim it cannot honour.
            raise ValueError(
                f"{module_info.name}: evograd.ops holds Level-1 primitives only, but "
                f"{op.name!r} declares level {op.level!r}. A fused task belongs to "
                f"evograd.benchmark.operator_suite.tasks or to a model package under "
                f"evograd.benchmark.topdown."
            )
        if op.name in discovered:
            raise ValueError(f"duplicate primitive declaration {op.name!r}")
        discovered[op.name] = op


def _discover() -> dict[str, OpDecl]:
    discovered: dict[str, OpDecl] = {}
    _collect(__path__, f"{__name__}.", discovered, depth=1)
    return dict(sorted(discovered.items()))


#: Every reusable Level-1 primitive contract, by name.
PRIMITIVES: dict[str, OpDecl] = _discover()


def get_primitive(name: str) -> OpDecl:
    """One primitive contract by name.

    ``EVOGRAD_DECLARATION`` still resolves an external declaration, so a
    scaffolded operator can be driven through the same code paths as a
    reviewed one without being registered.
    """
    try:
        return PRIMITIVES[name]
    except KeyError:
        reference = os.environ.get("EVOGRAD_DECLARATION")
        if reference:
            op = load_declaration(reference)
            if op.name == name:
                return op
        raise KeyError(
            f"Unknown primitive {name!r}; available: {sorted(PRIMITIVES)}. "
            f"Executable benchmark tasks resolve through evograd.benchmark.get_task."
        ) from None


__all__ = ["PRIMITIVES", "get_primitive"]
