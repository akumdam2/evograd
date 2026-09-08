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

from pathlib import Path
from typing import Any

#: Workload name -> the package directory holding its ``harvest/snapshot.json``.
#: The name is also the ``Provenance.model`` key and the ``--model`` argument, so
#: adding a workload is one entry here plus the package it points at.
#:
#: A name here means the package exists, not that it has been harvested: a
#: snapshot is *derived* from a run, so a newly added workload has a package and
#: no ``snapshot.json`` until someone executes its harvest. :func:`has_snapshot`
#: is the question to ask; :func:`load_snapshot` refuses with the command to run.
TOPDOWN_WORKLOADS: dict[str, str] = {
    "qwen3_0_6b": "qwen3_0_6b",
    "llama_3_8b": "llama3_8b",
}


class UnknownWorkload(KeyError):
    """A workload name with no package behind it."""


class UnharvestedWorkload(RuntimeError):
    """The package exists; nobody has run its harvest yet."""


def snapshot_path(name: str) -> Path:
    """Where one workload's tracked snapshot lives."""
    try:
        package = TOPDOWN_WORKLOADS[name]
    except KeyError:
        raise UnknownWorkload(
            f"no workload named {name!r}; this repository carries "
            f"{sorted(TOPDOWN_WORKLOADS)}"
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
            f"    python -m evograd.benchmark.topdown.{TOPDOWN_WORKLOADS[name]}.harvest.harvest "
            f"--out results/benchmark/topdown/{TOPDOWN_WORKLOADS[name]}/harvest.json\n"
            f"    python -m evograd.benchmark.topdown.{TOPDOWN_WORKLOADS[name]}.harvest.snapshot "
            f"--harvest results/benchmark/topdown/{TOPDOWN_WORKLOADS[name]}/harvest.json --write"
        )
    return load(path)


def load_snapshot_task(name: str, task: str) -> dict[str, Any]:
    """One Level-2 task entry from a workload's snapshot, by workload name.

    The door ``evograd.ops`` uses. A declaration for a model-specific operator is
    still a declaration: it should name the workload it was harvested from, not
    the import path that workload's package happens to have today.
    """
    from .common.snapshot import task as _task

    return _task(task, snapshot_path(name))


__all__ = [
    "TOPDOWN_WORKLOADS",
    "UnknownWorkload",
    "UnharvestedWorkload",
    "has_snapshot",
    "load_snapshot",
    "load_snapshot_task",
    "snapshot_path",
]
