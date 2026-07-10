"""Bridge to the legacy stringly-typed ``OperatorSpec`` contract.

The old repo's three pipelines share ``OperatorSpec`` (see
``pipeline/autograd_pair_fusion_agent/prompts.py``), loaded from
``<op>_spec.json`` files. This module derives that exact contract from an
:class:`~evograd.opdecl.activity.OpDecl`, so pipelines can be ported one at a
time: point them at ``to_operator_spec(op)`` first, then switch them to
consuming the ``OpDecl`` natively and delete this bridge.

Fidelity is enforced by ``tests/test_compat.py``, which diffs the derived
spec against snapshots of the original JSON files.
"""

from __future__ import annotations

from dataclasses import dataclass

from evograd.opdecl.activity import Const, OpDecl


@dataclass(frozen=True)
class OperatorSpec:
    """Mirror of the legacy pipeline contract (field-for-field)."""

    forward_fn_name: str
    forward_args: str
    backward_fn_name: str
    backward_args: str
    backward_returns: str
    forward_semantics: str
    backward_semantics: str
    no_grad_inputs: tuple[str, ...] = ()
    extra_constraints: str = ""


def _fmt_default(value: float | int) -> str:
    """Format a scalar default the way the legacy specs spell it (1e-5, not 1e-05)."""
    if isinstance(value, float):
        text = f"{value:g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            sign = "-" if exponent.startswith("-") else "+" if exponent.startswith("+") else ""
            digits = exponent.lstrip("+-").lstrip("0") or "0"
            text = f"{mantissa}e{sign}{digits}"
        return text
    return str(value)


def _forward_arg_decl(arg) -> str:
    if isinstance(arg, Const) and not arg.is_tensor and arg.default is not None:
        return f"{arg.name}={_fmt_default(arg.default)}"
    return arg.name


def to_operator_spec(op: OpDecl) -> OperatorSpec:
    forward_args = ", ".join(_forward_arg_decl(a) for a in op.args)

    backward_parts = [op.upstream_grad_name, "saved_tensors"]
    for const in op.scalar_const_args():
        if const.default is not None:
            backward_parts.append(f"{const.name}={_fmt_default(const.default)}")
    backward_args = ", ".join(backward_parts)

    return OperatorSpec(
        forward_fn_name=op.forward_fn_name,
        forward_args=forward_args,
        backward_fn_name=op.backward_fn_name,
        backward_args=backward_args,
        backward_returns=", ".join(op.grad_names()),
        forward_semantics=op.forward_semantics,
        backward_semantics=op.backward_semantics,
        no_grad_inputs=tuple(c.name for c in op.tensor_const_args()),
        extra_constraints=op.extra_constraints,
    )


def to_spec_dict(op: OpDecl) -> dict:
    """JSON-shaped view matching the legacy ``<op>_spec.json`` files exactly."""
    spec = to_operator_spec(op)
    return {
        "forward_fn_name": spec.forward_fn_name,
        "forward_args": spec.forward_args,
        "backward_fn_name": spec.backward_fn_name,
        "backward_args": spec.backward_args,
        "backward_returns": spec.backward_returns,
        "forward_semantics": spec.forward_semantics,
        "backward_semantics": spec.backward_semantics,
        "no_grad_inputs": list(spec.no_grad_inputs),
        "extra_constraints": spec.extra_constraints,
    }
