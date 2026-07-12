"""Build a training-ready ``torch.autograd.Function`` from a candidate seed.

Replaces the per-bench hand-written ``autograd_wrapper.py``: the declaration
drives argument routing, so the error-prone parts — placing ``None`` in the
grad slots of ``Const`` args and ordering the returned gradients — are
mechanical and cannot be misaligned.

A seed module must expose (either generic or op-prefixed names):

    forward_with_saved(*declared args)      -> (y, saved_tensors)
    backward_from_saved(dout, saved_tensors, **scalar consts it accepts)
                                            -> grads per op.grad_names() order

``saved_tensors`` may be any tuple mixing tensors and plain values — the
saved-tensor contract stays inside the EVOLVE-BLOCK, free for OpenEvolve to
change.
"""

from __future__ import annotations

import inspect

import torch

from evograd.opdecl.activity import Duplicated, OpDecl


def lookup_pair(op: OpDecl, module):
    """Find the seed's forward/backward, accepting generic or op-prefixed names."""

    def _lookup(generic: str, prefixed: str):
        fn = getattr(module, generic, None)
        if fn is None:
            fn = getattr(module, prefixed, None)
        if fn is None:
            raise AttributeError(
                f"seed module {getattr(module, '__name__', module)!r} defines neither "
                f"{generic!r} nor {prefixed!r}"
            )
        return fn

    return (
        _lookup("forward_with_saved", op.forward_fn_name),
        _lookup("backward_from_saved", op.backward_fn_name),
    )


def backward_const_kwargs(op: OpDecl, bwd, values: dict) -> dict:
    """Scalar Const args are re-passed to the backward — but only those its
    signature accepts (legacy seeds take eps, evoattention seeds take none)."""
    params = inspect.signature(bwd).parameters
    return {
        c.name: values.get(c.name, c.default)
        for c in op.scalar_const_args()
        if c.name in params
    }


def bind(op: OpDecl, module):
    """Return a callable ``fn(*declared args) -> y`` with autograd wired up."""
    fwd, bwd = lookup_pair(op, module)
    arg_names = [a.name for a in op.args]
    slot_by_grad = {
        a.grad_name: i for i, a in enumerate(op.args) if isinstance(a, Duplicated)
    }
    return_names = op.grad_names()

    class _Bound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            y, saved = fwd(*args)
            saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
            # save_for_backward only handles tensors; keep plain values (block
            # sizes, shapes...) in a layout list so any saved contract works.
            tensors, layout = [], []
            for item in saved:
                if torch.is_tensor(item):
                    layout.append(("tensor", len(tensors)))
                    tensors.append(item)
                else:
                    layout.append(("value", item))
            ctx.save_for_backward(*tensors)
            ctx.saved_layout = layout
            ctx.const_values = dict(
                zip(arg_names, args)
            )  # only consts are read back; grads flow via saved tensors
            return y

        @staticmethod
        def backward(ctx, dout):
            saved = tuple(
                ctx.saved_tensors[payload] if kind == "tensor" else payload
                for kind, payload in ctx.saved_layout
            )
            kwargs = backward_const_kwargs(op, bwd, ctx.const_values)
            grads = bwd(dout, saved, **kwargs)
            grads = (grads,) if torch.is_tensor(grads) else tuple(grads)
            if len(grads) != len(return_names):
                raise ValueError(
                    f"{op.name}: backward returned {len(grads)} gradients, "
                    f"contract requires {len(return_names)}: {return_names}"
                )
            slots: list = [None] * len(arg_names)
            for name, grad in zip(return_names, grads):
                slots[slot_by_grad[name]] = grad
            return tuple(slots)

    def call(*args, **kwargs):
        named = dict(zip(arg_names, args))
        named.update(kwargs)
        full = [named.get(a.name, getattr(a, "default", None)) for a in op.args]
        return _Bound.apply(*full)

    call.__name__ = f"{op.name}_autograd"
    call.__qualname__ = call.__name__
    return call
