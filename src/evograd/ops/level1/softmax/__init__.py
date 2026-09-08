"""Operator declaration: row-wise softmax."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, workloads_2d


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dy": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.softmax.liger import make_liger_softmax_autograd_pair_fns

    return make_liger_softmax_autograd_pair_fns()


_CORRECTNESS = (
    workloads_2d(((8, 512), (16, 1024)), ("float32",), tolerances=STANDARD_TOLERANCES)
    + workloads_2d(
        ((32, 4096), (64, 2048)),
        ("float16", "bfloat16"),
        tolerances=STANDARD_TOLERANCES,
    )
)

op = declare_op(
    name="softmax",
    level=1,
    family="reduction",
    forward="evograd.ops.level1.softmax.forward_ref:softmax_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Numerically stable softmax over the last dimension in fp32.",
    backward_semantics="Return dx = y * (dy - sum(dy*y, dim=-1, keepdim=True)).",
    correctness=_CORRECTNESS,
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("x",))},
    make_inputs=_inputs,
)
