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


def _tensor_args(op: OpDecl) -> list[str]:
    """Forward tensor inputs in declared order = AtenIR placeholders[1:]."""
    return [a.name for a in op.args if getattr(a, "shape", None) is not None]


def _scalar_inactive(op: OpDecl) -> list[Inactive]:
    return list(op.scalar_inactive_args())


def grad_indices(op: OpDecl) -> list[int]:
    """For each backward return, the index of its input among the tensor args.

    ``run_graph_program`` (emitted by the dispatch codegen) returns one
    gradient per forward tensor input, in input order. The wrapper picks the
    ``Active`` ones and emits them in contract order — no name matching.
    """
    tensor_names = _tensor_args(op)
    by_grad = {a.grad_name: a.name for a in op.active_args()}
    return [tensor_names.index(by_grad[g]) for g in op.grad_names()]


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
    backward_sig = f"{op.upstream_grad_name}, saved_tensors{scalar_sig}"
    backward_call = ", ".join(
        [op.upstream_grad_name, "saved_tensors"] + [c.name for c in scalar_inactive]
    )

    saved = ", ".join(f"{n}.contiguous()" for n in tensor_names)
    saved_tuple = f"({saved},)" if len(tensor_names) == 1 else f"({saved})"
    unpack = f"{', '.join(tensor_names)} = saved_tensors[:{len(tensor_names)}]"
    run_args = ",\n        ".join(
        [f"{op.upstream_grad_name}.contiguous()"] + [f"{n}.contiguous()" for n in tensor_names]
    )
    unused_scalars = "".join(f"    _ = {c.name}\n" for c in scalar_inactive)

    # An Active-only op with declaration-order grads can return the graph
    # outputs directly; anything else (Inactive tensors to drop, grad_order
    # overrides) selects by index.
    if indices == list(range(len(tensor_names))):
        backward_body = (
            f"{unused_scalars}"
            f"    {unpack}\n"
            f"    return run_graph_program(\n        {run_args},\n    )"
        )
    else:
        ret = ", ".join(f"_grads[{i}]" for i in indices)
        backward_body = (
            f"{unused_scalars}"
            f"    {unpack}\n"
            f"    _grads = run_graph_program(\n        {run_args},\n    )\n"
            f"    return ({ret})"
        )

    return f'''

_FORWARD_SPEC = {forward!r}


def _load_forward_callable():
    module_name, fn_name = (
        _FORWARD_SPEC.split(":", 1)
        if ":" in _FORWARD_SPEC
        else _FORWARD_SPEC.rsplit(".", 1)
    )
    module = __import__(module_name, fromlist=[fn_name])
    return getattr(module, fn_name)


def _forward_with_saved_impl({forward_sig}):
    # Conservative seed: only save original forward inputs. OpenEvolve may
    # replace this with saved intermediates.
    y = _load_forward_callable()({fwd_call})
    return y, {saved_tuple}


def _backward_from_saved_impl({backward_sig}):
{backward_body}


def {op.forward_fn_name}({forward_sig}):
    return _forward_with_saved_impl({fwd_call})


def {op.backward_fn_name}({backward_sig}):
    return _backward_from_saved_impl({backward_call})
'''
