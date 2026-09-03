"""Automatic registry for declaration modules in :mod:`evograd.ops`.

An operator is registered by adding an ``evograd/ops/level<N>/<name>/`` package
whose ``__init__.py`` exposes ``op = declare_op(...)``. Its forward reference,
input-generation helpers, baselines, and related implementation files live in
that same package. No central list is edited when operator #26 is added.

Level-4 tasks are whole-model training steps rather than operator pairs, so a
``level4/<name>/`` package exposes ``workload = declare_workload(...)`` instead
and registers into ``WORKLOADS``. Same discovery, different registry: an
``OpDecl`` is something the search can evolve, a ``WorkloadDecl`` is something
evolved kernels are patched into and measured on.

Operators are grouped into ``level1/`` through ``level4/`` directories matching
the benchmark hierarchy. The grouping is presentation only — the authority on a
task's level remains the declaration, and ``tests/test_ops_layout.py`` asserts
the directory agrees with it so the two cannot drift apart. Discovery therefore
recurses one level into those group packages, and a package that declares
neither an ``op`` nor a ``workload`` is treated as a group rather than an error.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import uuid
from pathlib import Path

from evograd.opdecl import OpDecl
from evograd.opdecl.workloads import WorkloadDecl


def _collect_ops(
    search_path,
    prefix: str,
    discovered: dict[str, OpDecl],
    workloads: dict[str, WorkloadDecl],
    *,
    depth: int,
) -> None:
    for module_info in pkgutil.iter_modules(search_path, prefix):
        if not module_info.ispkg:
            continue
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        op = getattr(module, "op", None)
        workload = getattr(module, "workload", None)
        if op is None and workload is None:
            # A grouping package (level1/ .. level4/) rather than a task.
            # Recurse once; deeper nesting is not a layout we use, and
            # allowing it would make a stray package anywhere silently register
            # tasks.
            if depth > 0:
                _collect_ops(
                    module.__path__,
                    f"{module_info.name}.",
                    discovered,
                    workloads,
                    depth=depth - 1,
                )
            continue
        if op is not None and workload is not None:
            raise ValueError(
                f"{module_info.name} declares both an op and a workload; "
                "a task is one or the other"
            )
        if op is not None:
            if not isinstance(op, OpDecl):
                raise TypeError(f"{module_info.name}.op must be OpDecl, got {type(op).__name__}")
            if op.name != short_name:
                raise ValueError(
                    f"{module_info.name}: declaration name {op.name!r} must match module {short_name!r}"
                )
            if op.name in discovered:
                raise ValueError(f"duplicate operator declaration {op.name!r}")
            discovered[op.name] = op
            continue
        if not isinstance(workload, WorkloadDecl):
            raise TypeError(
                f"{module_info.name}.workload must be WorkloadDecl, got {type(workload).__name__}"
            )
        if workload.name != short_name:
            raise ValueError(
                f"{module_info.name}: declaration name {workload.name!r} must match module {short_name!r}"
            )
        if workload.name in workloads:
            raise ValueError(f"duplicate workload declaration {workload.name!r}")
        workloads[workload.name] = workload


def _discover() -> tuple[dict[str, OpDecl], dict[str, WorkloadDecl]]:
    discovered: dict[str, OpDecl] = {}
    workloads: dict[str, WorkloadDecl] = {}
    _collect_ops(__path__, f"{__name__}.", discovered, workloads, depth=1)
    return dict(sorted(discovered.items())), dict(sorted(workloads.items()))


OPS: dict[str, OpDecl]
WORKLOADS: dict[str, WorkloadDecl]
OPS, WORKLOADS = _discover()


def load_op(reference: str) -> OpDecl:
    """Load an external ``path.py:attribute`` declaration."""
    path_text, separator, attribute = reference.partition(":")
    if not separator:
        raise ValueError(
            f"external declaration must be 'path.py:attribute', got {reference!r}"
        )
    path = Path(path_text)
    spec = importlib.util.spec_from_file_location(
        f"evograd_external_op_{uuid.uuid4().hex}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load declaration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    op = getattr(module, attribute)
    if not isinstance(op, OpDecl):
        raise TypeError(f"{reference} is not an OpDecl")
    op.validate()
    return op


def get_op(name: str) -> OpDecl:
    try:
        return OPS[name]
    except KeyError:
        reference = os.environ.get("EVOGRAD_DECLARATION")
        if reference:
            op = load_op(reference)
            if op.name == name:
                return op
        raise KeyError(f"Unknown operator {name!r}; available: {sorted(OPS)}") from None


def get_workload(name: str) -> WorkloadDecl:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise KeyError(
            f"Unknown workload {name!r}; available: {sorted(WORKLOADS)}"
        ) from None


__all__ = ["OPS", "WORKLOADS", "get_op", "get_workload", "load_op"]
