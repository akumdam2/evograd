"""Typed operator declarations: the contract every evograd component derives from.

Torch-facing derivations (oracle, bind, verify, make_case_inputs) are exported
lazily so declarations stay importable on machines
without torch (dev boxes have no CUDA; only GPU nodes run the real stack).
"""

import importlib

from evograd.opdecl.activity import (
    Arg,
    Active,
    Inactive,
    OpDecl,
    Workload,
    bind_shape,
    declare_op,
    format_default,
)

_LAZY = {
    "oracle": "evograd.opdecl.oracle",
    "resolve_forward": "evograd.opdecl.oracle",
    "bind": "evograd.opdecl.bind",
    "verify": "evograd.opdecl.verify",
    "VerifyReport": "evograd.opdecl.verify",
    "make_case_inputs": "evograd.opdecl.inputs",
}


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        # Importing e.g. evograd.opdecl.oracle pins the *submodule* onto this
        # package under the same name, shadowing the lazy export on the next
        # lookup. Cache the resolved object afterwards so it wins.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Arg",
    "Active",
    "Inactive",
    "OpDecl",
    "VerifyReport",
    "Workload",
    "bind",
    "bind_shape",
    "declare_op",
    "format_default",
    "make_case_inputs",
    "oracle",
    "resolve_forward",
    "verify",
]
