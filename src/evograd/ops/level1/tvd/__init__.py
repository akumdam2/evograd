"""Operator declaration: batchmean total-variation distance."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    p = torch.softmax(torch.randn((rows, cols), device=device, dtype=dtype), dim=-1)
    q = torch.softmax(torch.randn((rows, cols), device=device, dtype=dtype), dim=-1)
    return {
        "p": p,
        "q": q,
        "dout": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.tvd.liger import make_liger_tvd_autograd_pair_fns

    return make_liger_tvd_autograd_pair_fns()


op = declare_op(
    name="tvd",
    level=1,
    family="loss",
    forward="evograd.ops.level1.tvd.forward_ref:tvd_forward_ref",
    dims=("rows", "cols"),
    args=(Active("p", "[rows, cols]"), Active("q", "[rows, cols]")),
    output=Active("out", "[]"),
    parameter_args=(),
    forward_semantics="out = 0.5 * sum(abs(p-q)) / rows.",
    backward_semantics=(
        "Return dp=dout*0.5*sign(p-q)/rows and dq=-dp, using the zero "
        "subgradient for ties."
    ),
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("p", "q"))},
    make_inputs=_inputs,
)
