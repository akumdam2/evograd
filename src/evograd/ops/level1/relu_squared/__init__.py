"""Operator declaration: squared ReLU."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.relu_squared.liger import (
        make_liger_relu_squared_autograd_pair_fns,
    )

    return make_liger_relu_squared_autograd_pair_fns()


op = declare_op(
    name="relu_squared",
    level=1,
    family="activation",
    forward="evograd.ops.level1.relu_squared.forward_ref:relu_squared_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("out", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Element-wise out = relu(x)^2.",
    backward_semantics="Return dx = dout * 2*x where x>0 and zero otherwise.",
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("x",))},
    make_inputs=_inputs,
)
