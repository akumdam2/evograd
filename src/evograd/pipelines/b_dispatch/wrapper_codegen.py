"""Autograd-pair seed wrapper rendering for Pipeline B — declaration-native.

Replaces the old ``render_dispatch_autograd_pair_wrapper`` and its string
conventions (``"d"+name`` gradient matching, the ``eps``-only-kwarg rule,
``grad_reorder()``). Here everything is read off the :class:`OpDecl`:

* placeholder order   = tensor args in declaration order
* gradient selection  = positions of the ``Active`` args (``Inactive``
  tensor gradients such as ``d_res_mask`` are dropped by construction)
* return order        = ``op.grad_names()`` (honors ``grad_order`` overrides)
* scalar inactive args = re-passed as keyword args with their declared defaults

Pure string construction (no torch/triton), so it is unit-testable without
a GPU.
"""

from __future__ import annotations

from evograd.opdecl.activity import Active, Inactive, OpDecl, format_default
from evograd.pipelines.shared.artifact import render_deployment_layer


def _tensor_args(op: OpDecl) -> list[str]:
    """Forward tensor inputs in declared order = AtenIR placeholders[1:]."""
    return [a.name for a in op.args if getattr(a, "shape", None) is not None]


def _scalar_inactive(op: OpDecl) -> list[Inactive]:
    return list(op.scalar_inactive_args())


def grad_indices(op: OpDecl) -> list[int]:
    """For each backward return, its index among differentiable graph outputs.

    ``run_graph_program`` returns gradients for floating/complex tensor
    placeholders in declaration order. A floating Inactive tensor can still
    appear in the extracted backward graph (e.g. an attention mask), whereas
    integer/bool metadata never has a gradient slot.
    """
    non_differentiable_dtypes = {"int64", "int32", "bool"}
    graph_grad_names = [
        arg.name
        for arg in op.args
        if getattr(arg, "shape", None) is not None
        and getattr(arg, "dtype", None) not in non_differentiable_dtypes
    ]
    by_grad = {a.grad_name: a.name for a in op.active_args()}
    return [graph_grad_names.index(by_grad[g]) for g in op.grad_names()]


def render_autograd_pair_wrapper(forward: str, op: OpDecl) -> str:
    tensor_names = _tensor_args(op)
    scalar_inactive = _scalar_inactive(op)
    indices = grad_indices(op)

    scalar_sig = "".join(
        f", {c.name}" + (f"={format_default(c.default)}" if c.default is not None else "")
        for c in scalar_inactive
    )
    forward_sig = ", ".join(tensor_names) + scalar_sig
    fwd_call = ", ".join(tensor_names + [c.name for c in scalar_inactive])
    # Structured outputs are the declaration's business, not the pipeline's:
    # a multi-output op arrives with one upstream gradient per output, and the
    # graph program returns the same input gradients either way.
    upstream = tuple(op.upstream_grad_names)
    grads_param = op.upstream_grad_parameter
    backward_sig = f"{grads_param}, saved_tensors{scalar_sig}"
    outs = tuple(op.output_names)
    forward_capture = f"({', '.join(outs)})" if len(outs) > 1 else outs[0]

    saved = ", ".join(f"{n}.contiguous()" for n in tensor_names)
    saved_tuple = f"({saved},)" if len(tensor_names) == 1 else f"({saved})"
    # Trailing comma so a single name still tuple-unpacks (plain `x = t[:1]`
    # would bind the one-element slice itself, not the tensor).
    unpack_targets = ", ".join(tensor_names) + ("," if len(tensor_names) == 1 else "")
    unpack = f"{unpack_targets} = saved_tensors[:{len(tensor_names)}]"
    unpack_grads = f"    {', '.join(upstream)} = {grads_param}\n"
    run_args = ",\n        ".join(
        [f"{name}.contiguous()" for name in upstream]
        + [f"{n}.contiguous()" for n in tensor_names]
    )
    unused_scalars = "".join(f"    _ = {c.name}\n" for c in scalar_inactive)

    # Select grads by index and cast each to its source input's dtype — the
    # autograd contract. Dtype-generic graph programs bake extraction-time
    # dtypes into materialized constants (aten.full/scalar_tensor), so a
    # low-precision run can drift to fp32 internally; the cast is a no-op
    # when dtypes already match. `indices` live in graph-gradient space
    # (differentiable placeholders only), so the cast source is looked up by
    # the grad's own arg name, never by tensor-arg position.
    by_grad = {a.grad_name: a.name for a in op.active_args()}
    grad_sources = [by_grad[g] for g in op.grad_names()]
    ret = ", ".join(
        f"_grads[{i}].to({src}.dtype)" for i, src in zip(indices, grad_sources)
    )
    ret += "," if len(indices) == 1 else ""
    backward_body = (
        f"{unused_scalars}"
        f"{unpack_grads}"
        f"    {unpack}\n"
        f"    _grads = run_graph_program(\n        {run_args},\n    )\n"
        f"    return ({ret})"
    )

    body = f'''

_FORWARD_SPEC = {forward!r}


def _load_forward_callable():
    from evograd.opdecl.importing import resolve_callable

    return resolve_callable(_FORWARD_SPEC)


def {op.forward_fn_name}({forward_sig}):
    """Layer 2: this *is* the implementation, not a forwarder to a private twin.

    Conservative seed -- only the original forward inputs are saved. Evolution
    may replace the launch strategy and the saved state together.
    """
    {forward_capture} = _load_forward_callable()({fwd_call})
    return {forward_capture}, {saved_tuple}


def {op.backward_fn_name}({backward_sig}):
{backward_body}
'''
    return body + render_deployment_layer(op)
