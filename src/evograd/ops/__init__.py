"""Automatic registry for declaration modules in :mod:`evograd.ops`.

An operator is registered by adding an ``evograd/ops/<name>/`` package whose
``__init__.py`` exposes ``op = declare_op(...)``. Its forward reference,
input-generation helpers, baselines, and related implementation files live in
that same package. No central list is edited when operator #7 is added.
"""

from __future__ import annotations

import importlib
import pkgutil

from evograd.opdecl import OpDecl


def _discover_ops() -> dict[str, OpDecl]:
    discovered: dict[str, OpDecl] = {}
    prefix = f"{__name__}."
    for module_info in pkgutil.iter_modules(__path__, prefix):
        if not module_info.ispkg:
            continue
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        op = getattr(module, "op", None)
        if op is None:
            continue
        if not isinstance(op, OpDecl):
            raise TypeError(f"{module_info.name}.op must be OpDecl, got {type(op).__name__}")
        if op.name != short_name:
            raise ValueError(
                f"{module_info.name}: declaration name {op.name!r} must match module {short_name!r}"
            )
        if op.name in discovered:
            raise ValueError(f"duplicate operator declaration {op.name!r}")
        discovered[op.name] = op
    return dict(sorted(discovered.items()))


OPS: dict[str, OpDecl] = _discover_ops()


def get_op(name: str) -> OpDecl:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"Unknown operator {name!r}; available: {sorted(OPS)}") from None


__all__ = ["OPS", "get_op"]
