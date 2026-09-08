"""Operator declaration: polynomial RMS-normalized features."""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness

_REDUCED_ATOL = {"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}


def _tolerance(workload, result_name, atol, rtol):
    if result_name in {"dweight", "dbias"}:
        atol = _REDUCED_ATOL.get(workload.dtype, atol)
    return atol, rtol


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "weight": torch.randn((3, cols), device=device, dtype=dtype),
        "bias": torch.randn((cols,), device=device, dtype=dtype),
        "eps": 1e-6,
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.poly_norm.liger import make_liger_poly_norm_autograd_pair_fns

    return make_liger_poly_norm_autograd_pair_fns()


op = declare_op(
    name="poly_norm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.poly_norm.forward_ref:poly_norm_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("weight", "[3, cols]"),
        Active("bias", "[cols]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("out", "[rows, cols]"),
    parameter_args=("weight", "bias"),
    forward_semantics=(
        "For p in {3,2,1}, row-normalize x**p by its RMS, then return "
        "weight[0]*n3 + weight[1]*n2 + weight[2]*n1 + bias in x dtype."
    ),
    backward_semantics="Return dx, dweight, dbias; parameter gradients reduce across rows.",
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    tolerance_hook=_tolerance,
    performance_baselines={
        "liger": make_pair_baseline(
            _liger_factory, ("x", "weight", "bias", "eps"), ("eps",)
        )
    },
    make_inputs=_inputs,
)
