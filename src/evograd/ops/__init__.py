"""Operator registry: one declaration per operator, keyed by name.

Adding operator #N+1 to evograd means adding exactly two things here:
a ``<name>_forward_ref.py`` (or module path to an external forward) and a
``<name>.py`` containing its ``declare_op(...)``. Everything else — prompts,
wrapper codegen, oracle, verifier, evaluator — derives from the declaration.
"""

from evograd.opdecl import OpDecl

from evograd.ops import (
    evoattention,
    layernorm,
    layernorm_linear,
    linear,
    matmul,
    rmsnorm,
)

_MODULES = (evoattention, layernorm, layernorm_linear, linear, matmul, rmsnorm)

OPS: dict[str, OpDecl] = {module.op.name: module.op for module in _MODULES}


def get_op(name: str) -> OpDecl:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"Unknown operator {name!r}; available: {sorted(OPS)}") from None
