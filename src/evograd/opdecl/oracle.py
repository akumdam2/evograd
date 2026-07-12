"""Ground-truth backward, derived from the declaration.

This is the activity annotations doing real work: ``requires_grad`` is set on
exactly the ``Active`` args, the declared forward reference runs under
autograd, and ``torch.autograd.grad`` with the output shadow as
``grad_outputs`` yields the reference gradients. No per-operator
``backward_ref.py`` is ever written.
"""

from __future__ import annotations

import importlib

import torch

from evograd.opdecl.activity import Active, OpDecl


def resolve_forward(op: OpDecl):
    module_name, _, fn_name = op.forward.partition(":")
    return getattr(importlib.import_module(module_name), fn_name)


def oracle(op: OpDecl, inputs: dict, dout: torch.Tensor | None = None):
    """Run the forward reference under autograd and return ``(y, grads)``.

    ``inputs`` maps declared arg names to values (scalar ``Inactive`` args may be
    omitted; their declared default is used). ``grads`` maps gradient names to
    tensors, ordered per ``op.grad_names()``. ``dout`` defaults to
    ``inputs[op.upstream_grad_name]``.
    """
    fwd = resolve_forward(op)

    positional = []
    leaves: list[tuple[str, torch.Tensor]] = []
    for arg in op.args:
        value = inputs.get(arg.name, getattr(arg, "default", None))
        if value is None:
            raise ValueError(f"{op.name}: missing input {arg.name!r} and no default")
        if isinstance(arg, Active):
            value = value.detach().clone().requires_grad_(True)
            leaves.append((arg.grad_name, value))
        positional.append(value)

    y = fwd(*positional)

    if dout is None:
        dout = inputs[op.upstream_grad_name]
    grad_tensors = torch.autograd.grad(y, [t for _, t in leaves], grad_outputs=dout)

    by_name = {name: g for (name, _), g in zip(leaves, grad_tensors)}
    return y, {name: by_name[name] for name in op.grad_names()}
