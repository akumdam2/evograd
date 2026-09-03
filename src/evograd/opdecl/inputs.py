"""Concrete input construction for a declared workload.

Builds every primal plus the upstream gradient from the declaration alone,
deterministically seeded by the workload. Ops whose ``Inactive`` tensors have
semantics (e.g. evoattention's additive keep/drop mask) supply a
``make_inputs`` hook on their declaration instead.
"""

from __future__ import annotations

import torch

from evograd.opdecl.activity import Active, OpDecl, Workload, bind_shape

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype {name!r}; known: {sorted(_DTYPES)}") from None


def arg_dtype_name(arg, workload: Workload) -> str:
    """An arg with a fixed dtype (e.g. a float32 mask) keeps it; an unset or
    alternative dtype ("float16|bfloat16") follows the workload."""
    if arg.dtype is None or "|" in arg.dtype:
        return workload.dtype
    return arg.dtype


def case_seed(workload: Workload) -> int:
    """Deterministic generic per-case seed for newly declared operators.

    Migrated operators with historical comparison requirements provide a
    declaration-local ``make_inputs`` hook that preserves their legacy recipe.
    """
    seed = 0
    for name in sorted(workload.dims):
        seed = seed * 131 + workload.dims[name]
    return seed * 131 + sum(map(ord, workload.dtype))


def make_case_inputs(op: OpDecl, workload: Workload, device: str = "cuda") -> dict:
    """All primals plus the upstream gradient, keyed by declared names."""
    if op.make_inputs is not None:
        return op.make_inputs(torch, op, workload, device)

    torch.manual_seed(case_seed(workload))
    values: dict = {}
    for arg in op.args:
        if isinstance(arg, Active):
            shape = bind_shape(arg.shape, workload.dims)
            dtype = resolve_dtype(arg_dtype_name(arg, workload))
            # std=0.5 keeps fp16/bf16 magnitudes in a sane range.
            values[arg.name] = torch.randn(shape, device=device, dtype=dtype) * 0.5
        elif arg.is_tensor:
            shape = bind_shape(arg.shape, workload.dims)
            dtype = resolve_dtype(arg_dtype_name(arg, workload))
            values[arg.name] = torch.zeros(shape, device=device, dtype=dtype)
        else:
            values[arg.name] = arg.default

    # One upstream gradient per declared output, each under its own name. A
    # single-output declaration therefore lands exactly one entry, as before.
    for out in op.outputs:
        out_shape = bind_shape(out.shape, workload.dims)
        out_dtype = resolve_dtype(arg_dtype_name(out, workload))
        values[out.grad_name] = torch.randn(out_shape, device=device, dtype=out_dtype) * 0.5
    return values


def upstream_grad_values(op, values):
    """The upstream gradient(s) in the shape the candidate ABI passes them.

    A Tensor for a single output, an ordered tuple for several. Every caller
    goes through this rather than indexing ``values`` directly, so the two ABIs
    cannot drift apart.
    """
    grads = tuple(values[name] for name in op.upstream_grad_names)
    return grads if op.is_multi_output else grads[0]


def as_output_tuple(op, result):
    """Normalize a forward's result to a tuple, checking arity.

    A single-output candidate must return a Tensor and a multi-output candidate
    a tuple of them; returning the wrong shape is a contract error rather than
    something to be quietly unwrapped, because a one-element tuple and a Tensor
    mean different things to the backward that follows.
    """
    import torch

    if op.is_multi_output:
        if torch.is_tensor(result) or not isinstance(result, (tuple, list)):
            raise ValueError(
                f"{op.name} declares {len(op.outputs)} outputs {op.output_names}; "
                f"forward returned a single {type(result).__name__}"
            )
        values = tuple(result)
        if len(values) != len(op.outputs):
            raise ValueError(
                f"{op.name}: forward returned {len(values)} outputs, contract "
                f"requires {len(op.outputs)}: {op.output_names}"
            )
        return values
    if not torch.is_tensor(result):
        raise ValueError(
            f"{op.name} declares one output {op.output_names[0]!r}; forward "
            f"returned {type(result).__name__}"
        )
    return (result,)
