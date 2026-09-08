"""Operator declaration: layernorm."""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops._common import make_pair_baseline


def make_layernorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    bias = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "eps": 1e-5, "dy": dy}


def layernorm_tolerance(workload, result_name, atol, rtol):
    """Account for BF16 error growth in long parameter-gradient reductions."""
    if workload.dtype == "bfloat16" and result_name in {"dweight", "dbias"}:
        rows = workload.dims["rows"]
        atol = max(atol, atol * math.sqrt(max(1.0, rows / 8.0)))
    return atol, rtol


def _liger_factory():
    from liger_kernel.ops.layer_norm import layer_norm_backward, layer_norm_forward

    def forward(x, weight, bias, eps):
        y, x_2d, mean, rstd, _block_size, _num_warps = layer_norm_forward(
            x, weight, bias, eps
        )
        return y, (x_2d, weight, bias, mean, rstd)

    def backward(dy, saved):
        x_2d, weight, bias, mean, rstd = saved
        dy_2d = dy.view(-1, dy.shape[-1]).contiguous()
        return layer_norm_backward(dy_2d, x_2d, weight, bias, mean, rstd)

    return forward, backward


measure_liger_baseline = make_pair_baseline(
    _liger_factory, ("x", "weight", "bias", "eps")
)

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
    performance_baselines={"liger": measure_liger_baseline},
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    tolerance_hook=layernorm_tolerance,
    make_inputs=make_layernorm_inputs,
)
