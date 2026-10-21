"""Operator declaration: layernorm."""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op


def make_layernorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    bias = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "eps": (workload.provenance.free.get("eps", 1e-5) if workload.provenance else 1e-5), "dy": dy}


def layernorm_tolerance(workload, result_name, atol, rtol):
    """Row-scale ``atol`` on the parameter-gradient reductions.

    ``dweight`` and ``dbias`` sum over every row, so their rounding error grows
    like ``sqrt(rows)`` in every dtype, not only bfloat16: at MegaFold's pair
    shape (rows = 384 * 384 = 147456, hidden 128) the float32 sum lands at
    5e-4 against a float64 oracle -- for eager PyTorch and torch.compile as
    much as for any candidate -- while the flat float32 tolerance is 2e-5. The
    anchor is the smallest correctness case (8 rows), so nothing at or below it
    changes; ``rtol`` is untouched.
    """
    if result_name in {"dweight", "dbias"}:
        rows = workload.dims["rows"]
        atol = max(atol, atol * math.sqrt(max(1.0, rows / 8.0)))
    return atol, rtol


op = declare_op(
    name="layernorm",
    forward="evograd.ops.level1.layernorm.forward_ref:layernorm_forward_ref",
    runtime_forward="evograd.ops.level1.layernorm.forward_ref:layernorm_runtime_ref",
    level=1,
    family="norm",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Active("bias", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight", "bias"),
    forward_semantics='Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. Forward must produce the same `y` as row-wise LayerNorm over the last dimension.',
    backward_semantics='Backward must consume only `dy`, `saved_tensors`, and `eps`. Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.',
    # Correctness retains all supported dtypes. Evolution performance follows
    # Liger's BF16 industrial grid; rows < 4096 are the small regime.
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="float16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="float16"),
        Workload(dims=dict(rows=32, hidden=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="bfloat16"),
        Workload(dims=dict(rows=128, hidden=4096), dtype="bfloat16"),
        Workload(dims=dict(rows=1024, hidden=8192), dtype="bfloat16"),
    ),
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    tolerance_hook=layernorm_tolerance,
    make_inputs=make_layernorm_inputs,
)
