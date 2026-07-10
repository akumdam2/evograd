"""Typed operator declarations: the contract every evograd component derives from."""

from evograd.opdecl.activity import Arg, Const, Duplicated, OpDecl, Workload, declare_op
from evograd.opdecl.compat import OperatorSpec, to_operator_spec, to_spec_dict

__all__ = [
    "Arg",
    "Const",
    "Duplicated",
    "OpDecl",
    "OperatorSpec",
    "Workload",
    "declare_op",
    "to_operator_spec",
    "to_spec_dict",
]
