"""Loading a declaration that lives outside the repository.

A scaffolded or user-supplied declaration is a ``path.py:attribute`` reference
rather than a package the registry discovers. Loading one is declaration
infrastructure, not registry membership: the result is an :class:`OpDecl` that
has never been registered anywhere and does not become a benchmark task by
being read. That is why this lives here rather than in a registry module --
:mod:`evograd.ops` holds reusable primitives and :mod:`evograd.benchmark` holds
what is benchmarked, and an external declaration is neither.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from evograd.opdecl.activity import OpDecl


def load_declaration(reference: str) -> OpDecl:
    """Load and validate an external ``path.py:attribute`` declaration."""
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


__all__ = ["load_declaration"]
