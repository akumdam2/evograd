"""Import helpers shared by declarations, extraction, and synthesis.

Forward references may be regular ``package.module:function`` specs or
``/path/to/file.py:function`` specs. The latter is the stable boundary used by
the high-level callable/file interface. :func:`reference_closure` writes a
forward spec out as one standalone source file, for prompts and task packs.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import inspect
import sys
import textwrap
import types
import uuid
from pathlib import Path
from typing import Any


def resolve_callable(spec: str):
    module_or_path, separator, function_name = spec.partition(":")
    if not separator:
        module_or_path, function_name = spec.rsplit(".", 1)

    path = Path(module_or_path)
    if path.suffix == ".py" or "/" in module_or_path:
        if not path.is_file():
            raise FileNotFoundError(f"forward file not found: {path}")
        module_spec = importlib.util.spec_from_file_location(
            f"evograd_forward_{uuid.uuid4().hex}", path
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not import forward file: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_or_path)

    value = getattr(module, function_name)
    if not callable(value):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return value


# ── the forward reference, as a standalone module ────────────────────────────


def _imports_for(name: str, value: Any) -> str | None:
    """An import line binding ``name`` to ``value``, when one exists."""
    if isinstance(value, types.ModuleType):
        module = value.__name__
        if name == module:
            return f"import {module}"
        if "." not in module and name != module:
            return f"import {module} as {name}"
        parent, _, leaf = module.rpartition(".")
        if leaf == name:
            return f"from {parent} import {leaf}"
        return f"import {module} as {name}"
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname and "." not in qualname and not module.startswith("evograd"):
        alias = "" if qualname == name else f" as {name}"
        return f"from {module} import {qualname}{alias}"
    return None


def _global_names(node: ast.AST) -> list[str]:
    """Free names a definition reads, in first-use order.

    Conservative about scope: a name bound anywhere inside the definition
    (argument, assignment, nested def, import, handler) counts as local.
    """
    bound = set()
    seen: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and child is not node:
            bound.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in bound and child.id not in seen:
                seen.append(child.id)
    return seen


def reference_closure(forward_spec: str) -> tuple[str, str]:
    """Source of the forward reference and everything it reads from evograd.

    Returns ``(source, entry_name)``. Library names (``torch``, ``F``,
    ``math``...) become imports; functions and literal constants defined in an
    evograd module are copied, recursively, in dependency order. This is the
    information the frontend reaches by tracing the callable, written out so a
    text-only agent can read the same computation.
    """
    entry = resolve_callable(forward_spec)
    imports: dict[str, str] = {}
    definitions: dict[str, str] = {}
    unresolved: list[str] = []
    order: list[str] = []

    def visit(name: str, value: Any, namespace: dict[str, Any]) -> None:
        if name in definitions or name in imports:
            return
        line = _imports_for(name, value)
        if line is not None and not (
            callable(value) and getattr(value, "__module__", "").startswith("evograd")
        ):
            imports[name] = line
            return
        if inspect.isfunction(value) or inspect.isclass(value):
            source = textwrap.dedent(inspect.getsource(value))
            tree = ast.parse(source)
            if value.__name__ != name:
                source += f"\n{name} = {value.__name__}\n"
            definitions[name] = source
            inner = getattr(value, "__globals__", None) or vars(sys.modules[value.__module__])
            for free in _global_names(tree.body[0]):
                if free in inner:
                    visit(free, inner[free], inner)
                elif not hasattr(builtins, free):
                    unresolved.append(free)
            order.append(name)
            return
        if isinstance(value, (int, float, str, bool, tuple, frozenset, type(None))):
            definitions[name] = f"{name} = {value!r}\n"
            order.append(name)
            return
        unresolved.append(name)

    name = entry.__name__
    visit(name, entry, entry.__globals__)
    if unresolved:
        raise ValueError(
            f"{forward_spec}: cannot write a standalone reference; unresolved names "
            f"{sorted(set(unresolved))}"
        )
    body = "\n\n".join(definitions[n].rstrip() + "\n" for n in order)
    header = (
        '"""The PyTorch forward reference this task is defined by.\n\n'
        f"Source: {forward_spec}\n"
        "The oracle differentiates this function with torch.autograd; candidates\n"
        "must not call it.\n"
        '"""\n'
    )
    import_block = "\n".join(sorted(set(imports.values())))
    source = f"{header}\n{import_block}\n\n\n{body}\nreference_forward = {name}\n"
    compile(source, f"<reference {forward_spec}>", "exec")
    return source, name
