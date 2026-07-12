"""Concrete input construction for a declared workload.

Builds every primal plus the upstream gradient from the declaration alone,
deterministically seeded by the workload. Ops whose ``Const`` tensors have
semantics (e.g. evoattention's additive keep/drop mask) supply a
``make_inputs`` hook on their declaration instead.
"""

from __future__ import annotations

import torch

from evograd.opdecl.activity import Duplicated, OpDecl, Workload, bind_shape

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
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
        if isinstance(arg, Duplicated):
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

    out_shape = bind_shape(op.output.shape, workload.dims)
    out_dtype = resolve_dtype(arg_dtype_name(op.output, workload))
    values[op.upstream_grad_name] = torch.randn(out_shape, device=device, dtype=out_dtype) * 0.5
    return values
