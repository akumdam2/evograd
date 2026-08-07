"""Typed operator declarations: the contract every evograd component derives from.

Torch-facing derivations (oracle, bind, verify, make_case_inputs) are exported
lazily so declarations stay importable on machines
without torch (dev boxes have no CUDA; only GPU nodes run the real stack).
"""

import importlib
import sys
import types

from evograd.opdecl.activity import (
    Arg,
    Active,
    Inactive,
    OpDecl,
    Provenance,
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
        globals()[name] = value  # direct dict write, not routed through __setattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _Module(types.ModuleType):
    """Keeps the lazy function exports authoritative.

    ``oracle``, ``bind`` and ``verify`` name both a submodule and its main
    function. After ``import evograd.opdecl.oracle`` the import system pins the
    *submodule* onto this package via setattr, which would shadow the function
    on every later ``from evograd.opdecl import oracle`` — in whichever order
    the two imports happen. Dropping the pin keeps ``__getattr__`` (and its
    cache) the source of truth; submodule-path imports are unaffected.
    """

    def __setattr__(self, name: str, value) -> None:
        if name in _LAZY and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Module


__all__ = [
    "Arg",
    "Active",
    "Inactive",
    "OpDecl",
    "Provenance",
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
