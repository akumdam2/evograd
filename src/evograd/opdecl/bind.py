"""Build a training-ready ``torch.autograd.Function`` from a candidate seed.

Replaces the per-bench hand-written ``autograd_wrapper.py``: the declaration
drives argument routing, so the error-prone parts — placing ``None`` in the
grad slots of ``Inactive`` args and ordering the returned gradients — are
mechanical and cannot be misaligned.

A seed module must expose (either generic or op-prefixed names):

    forward_with_saved(*declared args)      -> (y, saved_tensors)
    backward_from_saved(dout, saved_tensors, **scalar inactive args it accepts)
                                                -> grads per op.grad_names() order

``saved_tensors`` may be any tuple mixing tensors and plain values — the
saved-tensor contract stays inside the EVOLVE-BLOCK, free for OpenEvolve to
change.
"""

from __future__ import annotations

import inspect

import torch

from evograd.opdecl.activity import Active, OpDecl


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
        if not callable(fn):
            raise TypeError(f"seed attribute {generic!r}/{prefixed!r} is not callable")
        return fn

    return (
        _lookup("forward_with_saved", op.forward_fn_name),
        _lookup("backward_from_saved", op.backward_fn_name),
    )


def backward_inactive_kwargs(op: OpDecl, bwd, values: dict) -> dict:
    """Scalar Inactive args are re-passed to the backward — but only those its
    signature accepts (legacy seeds take eps, evoattention seeds take none)."""
    params = inspect.signature(bwd).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return {
        c.name: values.get(c.name, c.default)
        for c in op.scalar_inactive_args()
        if c.name in params or accepts_kwargs
    }


def bind(op: OpDecl, module):
    """Return a callable ``fn(*declared args) -> y`` with autograd wired up."""
    fwd, bwd = lookup_pair(op, module)
    arg_names = [a.name for a in op.args]
    slot_by_grad = {
        a.grad_name: i for i, a in enumerate(op.args) if isinstance(a, Active)
    }
    return_names = op.grad_names()

    class _Bound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            y, saved = fwd(*args)
            ctx.multi_output = isinstance(y, (tuple, list))
            if ctx.multi_output:
                y = tuple(y)
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
            # Do not retain tensor inputs outside the candidate's explicit saved
            # state. Only scalar Inactive values are needed again by backward.
            ctx.inactive_values = {
                c.name: args[arg_names.index(c.name)] for c in op.scalar_inactive_args()
            }
            return y

        @staticmethod
        def backward(ctx, *douts):
            # One incoming gradient per declared output; the pair's backward
            # receives them as a tuple when there are several.
            dout = tuple(douts) if ctx.multi_output else douts[0]
            saved = tuple(
                ctx.saved_tensors[payload] if kind == "tensor" else payload
                for kind, payload in ctx.saved_layout
            )
            kwargs = backward_inactive_kwargs(op, bwd, ctx.inactive_values)
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
        if len(args) > len(arg_names):
            raise TypeError(
                f"{op.name}: expected at most {len(arg_names)} positional args, got {len(args)}"
            )
        named = dict(zip(arg_names, args))
        unknown = set(kwargs) - set(arg_names)
        if unknown:
            raise TypeError(f"{op.name}: unexpected keyword arguments {sorted(unknown)}")
        duplicates = set(kwargs) & set(arg_names[: len(args)])
        if duplicates:
            raise TypeError(f"{op.name}: multiple values for arguments {sorted(duplicates)}")
        named.update(kwargs)
        full = []
        for arg in op.args:
            if arg.name in named:
                full.append(named[arg.name])
            elif isinstance(arg, Active) or getattr(arg, "is_tensor", False):
                raise TypeError(f"{op.name}: missing required argument {arg.name!r}")
            elif arg.default is not None:
                full.append(arg.default)
            else:
                raise TypeError(f"{op.name}: missing required argument {arg.name!r}")
        return _Bound.apply(*full)

    call.__name__ = f"{op.name}_autograd"
    call.__qualname__ = call.__name__
    return call
