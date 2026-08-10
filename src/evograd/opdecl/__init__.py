"""Typed operator declarations: the contract every evograd component derives from.

Declarations stay importable on machines without torch.  When torch is
available, torch-facing derivations are bound eagerly so their public function
names cannot be replaced by Python's same-named submodule attributes.
"""

import importlib
import importlib.util

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

# ``from evograd.opdecl import oracle`` normally prefers the existing
# ``evograd.opdecl.oracle`` submodule over ``__getattr__`` and therefore returns
# a module instead of the public function.  Bind these exports after importing
# their modules whenever torch is installed.  On torch-free machines, retain
# lazy lookup so the declaration-only API above remains usable.
if importlib.util.find_spec("torch") is not None:
    from evograd.opdecl.bind import bind
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import oracle, resolve_forward
    from evograd.opdecl.verify import VerifyReport, verify


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
