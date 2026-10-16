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


def resolve_runtime_forward(op: OpDecl):
    """The forward to *time* the eager baseline through.

    ``op.forward`` states the mathematics in primitives, which is what the
    oracle must differentiate and what AtenIR lowers into an unfused seed. It is
    not what a production PyTorch user would run, and timing against it turns
    every speedup into a comparison with a strawman. Declarations that have a
    fused equivalent name it in ``runtime_forward``; the rest fall back, because
    for them the primitive spelling really is the best PyTorch available.

    A model-reference declaration (``op.reference_contract``) has a single
    spelling -- the model's own computation -- so this resolves the same
    callable the oracle differentiates.
    """
    if op.runtime_forward:
        return resolve_callable(op.runtime_forward)
    return resolve_callable(op.forward)


#: ``reference_identity(op)["kind"]`` for the two contracts a declaration can have.
DECLARED_MATHEMATICS = "declared_forward_oracle"
MODEL_REFERENCE = "model_reference"


def reference_identity(op: OpDecl) -> dict:
    """What a report or a policy records about the reference ``op`` is judged against.

    ``kind`` distinguishes a model-reference contract (``op.reference_contract``
    set: ``forward`` is a pinned model's own computation) from the historical
    declared-mathematics oracle. For a model reference, ``source`` is what the
    reference module reports about its pinned source (``source_identity()``),
    so a verdict or a frozen threshold names the exact computation behind it.
    """
    identity = {
        "kind": MODEL_REFERENCE if op.reference_contract else DECLARED_MATHEMATICS,
        "forward": op.forward,
        "runtime_forward": op.runtime_forward,
        "reference_dtype": op.reference_dtype,
        "reference_contract": op.reference_contract,
    }
    if op.reference_contract:
        import importlib

        module = importlib.import_module(op.forward.split(":", 1)[0])
        describe = getattr(module, "source_identity", None)
        identity["source"] = describe() if callable(describe) else None
    return identity


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
    the declaration's upstream gradients from ``inputs``. For a multi-output
    declaration ``y`` is an ordered tuple and ``dout`` may be a matching tuple;
    a single-output declaration returns a bare Tensor, exactly as before.

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

    A model-reference declaration (``op.reference_contract``) is differentiated
    at the operands' own dtypes, through exactly the casts the model performs.
    ``reference_dtype`` on such a declaration is refused here as well as in
    ``OpDecl.validate``, so a declaration rebuilt with ``dataclasses.replace``
    cannot widen the model's computation into a different target.
    """
    if op.reference_contract and op.reference_dtype:
        raise ValueError(
            f"{op.name}: reference_dtype={op.reference_dtype!r} on a model reference "
            f"contract ({op.reference_contract}) would change its numerical target")
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

    from evograd.opdecl.inputs import as_output_tuple, upstream_grad_values

    result = fwd(*positional)
    ys = as_output_tuple(op, result)

    if dout is None:
        dout = upstream_grad_values(op, inputs)
    douts = dout if isinstance(dout, (tuple, list)) else (dout,)
    if len(douts) != len(ys):
        raise ValueError(
            f"{op.name}: got {len(douts)} upstream gradients for {len(ys)} outputs"
        )
    y_dtypes = [d.dtype for d in douts]
    if reference_dtype is not None:
        douts = tuple(_promote(d, reference_dtype) for d in douts)
    grad_tensors = torch.autograd.grad(
        ys, [t for _, t in leaves], grad_outputs=tuple(douts)
    )

    by_name = {name: g for (name, _), g in zip(leaves, grad_tensors)}
    if reference_dtype is not None:
        ys = tuple(y.to(dtype) for y, dtype in zip(ys, y_dtypes))
        by_name = {
            name: grad.to(output_dtypes[name]) for name, grad in by_name.items()
        }
    # A single-output declaration still returns a bare Tensor, so every existing
    # caller is untouched.
    out = ys if op.is_multi_output else ys[0]
    return out, {name: by_name[name] for name in op.grad_names()}
