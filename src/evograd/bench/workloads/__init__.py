"""Level-4 workloads: whole models, executed the way training executes them.

Levels 1-3 are declared operators -- a forward reference, a pair contract, a
shape suite -- because that is what an evolved kernel replaces. Level 4 is the
other end of the telescope: one real model, one real training step, run through
the framework a user would actually run. Nothing here is declared through
``OpDecl``; a model is not an operator and forcing it into that shape would
distort both.

What Level 4 exists to provide is a *reference execution* that later stages can
observe. The operator suites answer "is this kernel faster than that kernel";
only a real training step can answer "does this kernel appear in the model at
all, at which shapes, and how often".

One package per workload, each owning only what is specific to its model --
architecture, boundary classes, adapters, and the snapshot its own harvest
produced. Everything those packages share lives in :mod:`.common`.

This module is the registry. It resolves a workload *name* to its tracked
snapshot, which is what lets ``evograd.ops`` state "these are the shapes the
model ran" without importing any particular workload. It holds no torch and no
Transformers import, because every declaration in ``ops/`` reaches it at import
time on machines that have neither.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Workload name -> the package directory holding its ``harvest/snapshot.json``.
#: The name is also the ``Provenance.model`` key and the ``--model`` argument, so
#: adding a workload is one entry here plus the package it points at.
#:
#: A name here means the package exists, not that it has been harvested: a
#: snapshot is *derived* from a run, so a newly added workload has a package and
#: no ``snapshot.json`` until someone executes its harvest. :func:`has_snapshot`
#: is the question to ask; :func:`load_snapshot` refuses with the command to run.
WORKLOADS: dict[str, str] = {
    "qwen3_0_6b": "qwen3",
    "llama_3_8b": "llama3",
}


class UnknownWorkload(KeyError):
    """A workload name with no package behind it."""


class UnharvestedWorkload(RuntimeError):
    """The package exists; nobody has run its harvest yet."""


def snapshot_path(name: str) -> Path:
    """Where one workload's tracked snapshot lives."""
    try:
        package = WORKLOADS[name]
    except KeyError:
        raise UnknownWorkload(
            f"no workload named {name!r}; this repository carries "
            f"{sorted(WORKLOADS)}"
        ) from None
    return Path(__file__).parent / package / "harvest" / "snapshot.json"


def has_snapshot(name: str) -> bool:
    """Has this workload been harvested on some machine and the result tracked?"""
    return snapshot_path(name).is_file()


def load_snapshot(name: str) -> dict[str, Any]:
    """One workload's frozen snapshot, with its hash verified.

    The verification lives in :mod:`.common.snapshot`, which imports only the
    standard library -- the whole point of the snapshot is that a declaration
    can read it without torch, Transformers, or a GPU.

    A workload whose harvest has never been run is refused with the command that
    would produce one, rather than with a missing-file traceback: nothing here
    can synthesise a snapshot, and a hand-written one would defeat its purpose.
    """
    from .common.snapshot import load

    path = snapshot_path(name)
    if not path.is_file():
        raise UnharvestedWorkload(
            f"{name} has no tracked snapshot at {path}. A snapshot is derived "
            f"from a harvest, not authored, so run one on a machine with a GPU:\n"
            f"    python -m evograd.bench.workloads.{WORKLOADS[name]}.harvest.harvest "
            f"--out results/{WORKLOADS[name]}-level4/harvest.json\n"
            f"    python -m evograd.bench.workloads.{WORKLOADS[name]}.harvest.snapshot "
            f"--harvest results/{WORKLOADS[name]}-level4/harvest.json --write"
        )
    return load(path)


# ── tier-3 workload adapters ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Tier3Adapter:
    """How the tier-3 CLI reaches one workload without knowing what it is.

    The runner has always been model-agnostic; the CLI was not, because a
    ``--model`` string had to be turned into a workload and into whatever extra
    providers that workload offers. Both were ``if args.model == ...`` branches,
    which is the shape that makes a third workload an edit in three places.

    ``build`` takes the parsed arguments and returns a ``TrainingWorkload``.
    ``providers`` optionally contributes named ``KernelSet`` s beyond the ones
    every workload has -- Qwen3's structural-identity control, for instance.

    ``options`` is what turns "qwen3 only" in a help string into something the
    parser can enforce: a flag named here is accepted for this workload and
    refused, by name, for one that does not declare it.
    """

    name: str
    #: ``(args) -> TrainingWorkload``
    build: Callable[[Any], Any]
    #: ``(args, site_registry) -> dict[str, KernelSet]``; extra providers.
    providers: Callable[[Any, Any], dict[str, Any]] | None = None
    #: Optional CLI flags this workload understands, by ``dest`` name.
    options: frozenset[str] = field(default_factory=frozenset)
    #: One line for ``--help``, so the model list documents itself.
    summary: str = ""


#: Workload name -> ``"module.path:attribute"`` naming its :class:`Tier3Adapter`.
#:
#: Resolved lazily, because an adapter reaches torch and possibly Transformers
#: while this module is imported by every operator declaration at ``ops`` import
#: time. Nothing here is imported until a name is actually looked up.
#: A workload appears here once it has tier-3 *sites* -- adapters that swap a
#: kernel into the live model. Being in :data:`WORKLOADS` is not enough: a
#: harvested workload can describe its shapes long before anything can patch it.
TIER3_ADAPTERS: dict[str, str] = {
    "qwen3_0_6b": "evograd.bench.workloads.qwen3.evaluation.tier3.adapter:ADAPTER",
    "alphafold3_2l": "evograd.ops.level4.alphafold3.adapter:ADAPTER_2L",
    "alphafold3": "evograd.ops.level4.alphafold3.adapter:ADAPTER",
}


def tier3_model_names() -> tuple[str, ...]:
    """Every ``--model`` the tier-3 CLI accepts, in registration order."""
    return tuple(TIER3_ADAPTERS)


def tier3_adapter(name: str) -> Tier3Adapter:
    """The adapter for one workload name, imported on demand."""
    try:
        target = TIER3_ADAPTERS[name]
    except KeyError:
        raise UnknownWorkload(
            f"no tier-3 workload named {name!r}; this repository carries "
            f"{sorted(TIER3_ADAPTERS)}"
        ) from None
    module_path, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_path), attribute)


def load_snapshot_task(name: str, task: str) -> dict[str, Any]:
    """One Level-2 task entry from a workload's snapshot, by workload name.

    The door ``evograd.ops`` uses. A declaration for a model-specific operator is
    still a declaration: it should name the workload it was harvested from, not
    the import path that workload's package happens to have today.
    """
    from .common.snapshot import task as _task

    return _task(task, snapshot_path(name))
