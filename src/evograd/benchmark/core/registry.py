"""The unified registry of executable benchmark tasks.

One name resolves one contract. Three kinds of contract reach this table and
they are kept distinguishable rather than merged:

* **Level-1 primitives** come from :mod:`evograd.ops`, which owns the
  mathematics and nothing about any model. Their model-observed cases are
  attached here by :mod:`evograd.benchmark.cases`, so the primitive package
  stays reusable and the cases stay owned by the workload that observed them.
* **Generic Level-2 tasks** come from
  :mod:`evograd.benchmark.operator_suite.tasks`. They are fusions any
  transformer-shaped run performs, so no model owns their identity.
* **Model-specific tasks** come from each workload package under
  :mod:`evograd.benchmark.topdown`. Their shapes, frequencies and provenance
  are one captured step's, and their identity belongs to that model even where
  the mathematics resembles another's.

Aggregation is where a name collision would first do damage -- two packages
could each declare ``rmsnorm`` and the loser would vanish silently -- so the
merge refuses duplicates and names both owners.
"""

from __future__ import annotations

import importlib
import os
import pkgutil

from evograd.opdecl import OpDecl
from evograd.opdecl.loader import load_declaration


class DuplicateTask(ValueError):
    """Two packages claim the same task name."""


def _discover_package(package_name: str, discovered: dict[str, tuple[str, OpDecl]],
                      *, depth: int, require_matching_name: bool) -> None:
    """Register every ``op`` found in ``package_name``, recursing into groups."""
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, f"{package_name}."):
        if not module_info.ispkg:
            continue
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        op = getattr(module, "op", None)
        if op is None:
            if depth > 0:
                _discover_package(
                    module_info.name, discovered,
                    depth=depth - 1, require_matching_name=require_matching_name,
                )
            continue
        if not isinstance(op, OpDecl):
            raise TypeError(f"{module_info.name}.op must be OpDecl, got {type(op).__name__}")
        if require_matching_name and op.name != short_name:
            raise ValueError(
                f"{module_info.name}: declaration name {op.name!r} must match "
                f"module {short_name!r}"
            )
        _register(discovered, op, module_info.name)


def _register(discovered: dict[str, tuple[str, OpDecl]], op: OpDecl, owner: str) -> None:
    existing = discovered.get(op.name)
    if existing is not None:
        raise DuplicateTask(
            f"task {op.name!r} is declared twice: by {existing[0]} and by {owner}. "
            f"A task name is the key a report, a candidate and a CLI all use, so "
            f"the two cannot coexist -- give the model-specific one its own key."
        )
    discovered[op.name] = (owner, op)


def _observed_bindings() -> tuple:
    """Every model's observed-case configuration, from its own manifest.

    Two models may bind the same primitive; each contributes its own named
    suite and the binder keeps both.

    This is the assembly point, and the concrete-model import belongs here
    rather than inside the binder: :mod:`evograd.benchmark.cases` knows how to
    apply a configuration, each model owns which primitives it binds, and this
    function is the one place that says which models there are.
    """
    from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
        OBSERVED_BINDINGS as LLAMA_3_2_1B,
    )
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import (
        OBSERVED_BINDINGS as QWEN3_0_6B,
    )

    # Order is fixed and part of the contract: a primitive both models bind
    # receives Qwen3's suite first and Llama-3's second, so the suites a task
    # serves are in the same place on every import.
    return QWEN3_0_6B + LLAMA_3_2_1B


def _discover() -> dict[str, OpDecl]:
    from evograd.benchmark.cases import bind_benchmark_cases
    from evograd.ops import PRIMITIVES

    discovered: dict[str, tuple[str, OpDecl]] = {}
    observed = _observed_bindings()

    # 1. Reusable primitives, with their benchmark cases bound on: the
    #    operator suite's performance grid, then any model's observed cases.
    for name, primitive in PRIMITIVES.items():
        _register(
            discovered,
            bind_benchmark_cases(primitive, observed),
            f"evograd.ops.level1.{name}",
        )

    # 2. Fusions that belong to no single model.
    _discover_package(
        "evograd.benchmark.operator_suite.tasks", discovered,
        depth=1, require_matching_name=True,
    )

    # 3. Model-specific tasks. A site package is named for the place in the
    #    model, not for the task key it serves, so the two are allowed to differ
    #    here; the workload's manifest is what ties them together.
    for workload in _model_task_packages():
        _discover_package(workload, discovered, depth=1, require_matching_name=False)

    return {name: op for name, (_owner, op) in sorted(discovered.items())}


def _model_task_packages() -> tuple[str, ...]:
    """Packages under ``topdown`` that declare executable tasks.

    Listed explicitly: a workload owns whole-model and captured-layer material
    that is not an executable pair contract, so "everything under topdown" would
    be the wrong rule.
    """
    return (
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2",
        "evograd.benchmark.topdown.llama3_2_1b.levels.level2",
    )


#: Every executable task, by the name a report, a candidate and a CLI all use.
TASKS: dict[str, OpDecl] = _discover()


def get_task(name: str) -> OpDecl:
    """One executable task by name.

    ``EVOGRAD_DECLARATION`` still resolves an external declaration, so a
    scaffolded task runs through the same paths as a reviewed one without being
    registered anywhere.
    """
    try:
        return TASKS[name]
    except KeyError:
        reference = os.environ.get("EVOGRAD_DECLARATION")
        if reference:
            op = load_declaration(reference)
            if op.name == name:
                return op
        raise KeyError(f"Unknown task {name!r}; available: {sorted(TASKS)}") from None


def load_task(reference: str) -> OpDecl:
    """Load an external ``path.py:attribute`` task declaration."""
    return load_declaration(reference)


def tasks_at_level(level: int) -> dict[str, OpDecl]:
    """Every registered task declaring one benchmark level."""
    return {name: op for name, op in TASKS.items() if op.level == level}


__all__ = [
    "DuplicateTask",
    "TASKS",
    "get_task",
    "load_task",
    "tasks_at_level",
]
