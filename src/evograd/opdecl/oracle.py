"""Ground-truth backward, derived from the declaration.

This is the activity annotations doing real work: ``requires_grad`` is set on
exactly the ``Active`` args, the declared forward reference runs under
autograd, and ``torch.autograd.grad`` with the output shadow as
``grad_outputs`` yields the reference gradients. No per-operator
``backward_ref.py`` is ever written.
"""

from __future__ import annotations

import torch

from evograd.opdecl.activity import Active, OpDecl
from evograd.opdecl.importing import resolve_callable


def resolve_forward(op: OpDecl):
    return resolve_callable(op.forward)


def _promote(value, reference_dtype: torch.dtype):
    """Widen a floating tensor to the reference dtype; leave everything else."""
    if torch.is_tensor(value) and value.is_floating_point():
        return value.to(reference_dtype)
    return value


def oracle(
    op: OpDecl,
    inputs: dict,
    dout: torch.Tensor | None = None,
    *,
    use_reference_dtype: bool = True,
):
    """Run the forward reference under autograd and return ``(y, grads)``.

    ``inputs`` maps declared arg names to values (scalar ``Inactive`` args may be
    omitted; their declared default is used). ``grads`` maps gradient names to
    tensors, ordered per ``op.grad_names()``. ``dout`` defaults to
    ``inputs[op.upstream_grad_name]``.

    When the declaration sets ``reference_dtype``, every floating input is
    widened to it, the reference runs at that precision, and the results are
    cast back to the dtype the candidate is required to return. The comparison
    then measures the candidate's own error rather than the difference between
    two equally-rounded computations — which is the only way a tolerance stays
    interpretable once a task composes many operators at low precision.

    ``use_reference_dtype=False`` suppresses that promotion. The eager-PyTorch
    *performance* baseline runs through this same function, and timing it at
    float32 while the candidate runs at bfloat16 would not be a baseline at all;
    only the correctness path wants the more accurate reference.
    """
    fwd = resolve_forward(op)
    reference_dtype = (
        getattr(torch, op.reference_dtype)
        if (op.reference_dtype and use_reference_dtype)
        else None
    )

    positional = []
    leaves: list[tuple[str, torch.Tensor]] = []
    output_dtypes: dict[str, torch.dtype] = {}
    for arg in op.args:
        value = inputs.get(arg.name, getattr(arg, "default", None))
        if value is None:
            raise ValueError(f"{op.name}: missing input {arg.name!r} and no default")
        if isinstance(arg, Active):
            # The gradient must come back in the input's dtype, so remember it
            # before promotion rather than reading it off the promoted leaf.
            output_dtypes[arg.grad_name] = value.dtype
            if reference_dtype is not None:
                value = _promote(value, reference_dtype)
            value = value.detach().clone().requires_grad_(True)
            leaves.append((arg.grad_name, value))
        elif reference_dtype is not None:
            value = _promote(value, reference_dtype)
        positional.append(value)

    y = fwd(*positional)

    if dout is None:
        dout = inputs[op.upstream_grad_name]
    y_dtype = dout.dtype
    if reference_dtype is not None:
        dout = _promote(dout, reference_dtype)
    grad_tensors = torch.autograd.grad(y, [t for _, t in leaves], grad_outputs=dout)

    by_name = {name: g for (name, _), g in zip(leaves, grad_tensors)}
    if reference_dtype is not None:
        y = y.to(y_dtype)
        by_name = {
            name: grad.to(output_dtypes[name]) for name, grad in by_name.items()
        }
    return y, {name: by_name[name] for name in op.grad_names()}
