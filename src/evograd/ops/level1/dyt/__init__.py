"""Operator declaration: dynamic tanh (DyT)."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness

_REDUCED_ATOL = {"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}


def _tolerance(workload, result_name, atol, rtol):
    if result_name in {"dalpha", "dgamma", "dbeta"}:
        atol = _REDUCED_ATOL.get(workload.dtype, atol)
    return atol, rtol


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    alpha = 0.5 + 0.1 * torch.randn((), device=device, dtype=dtype)
    gamma = 1.0 + 0.1 * torch.randn((cols,), device=device, dtype=dtype)
    beta = 0.1 * torch.randn((cols,), device=device, dtype=dtype)
    return {
        "x": x,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.dyt.liger import make_liger_dyt_autograd_pair_fns

    return make_liger_dyt_autograd_pair_fns()


op = declare_op(
    name="dyt",
    level=1,
    family="norm",
    forward="evograd.ops.level1.dyt.forward_ref:dyt_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("alpha", "[]"),
        Active("gamma", "[cols]"),
        Active("beta", "[cols]"),
    ),
    output=Active("out", "[rows, cols]"),
    parameter_args=("alpha", "gamma", "beta"),
    forward_semantics="out = gamma * tanh(alpha*x) + beta, computed in fp32.",
    backward_semantics=(
        "Return dx, dalpha, dgamma, dbeta. Parameter gradients reduce across rows; "
        "dalpha reduces over every element."
    ),
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    tolerance_hook=_tolerance,
    performance_baselines={
        "liger": make_pair_baseline(
            _liger_factory, ("x", "alpha", "gamma", "beta")
        )
    },
    make_inputs=_inputs,
)
