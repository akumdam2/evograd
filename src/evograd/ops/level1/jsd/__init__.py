"""Operator declaration: generalized Jensen-Shannon divergence."""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    log_q = torch.log_softmax(
        torch.randn((rows, cols), device=device, dtype=dtype), dim=-1
    )
    target = torch.log_softmax(
        torch.randn((rows, cols), device=device, dtype=dtype), dim=-1
    )
    return {
        "log_q": log_q,
        "target": target,
        "dout": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.jsd.liger import make_liger_jsd_autograd_pair_fns

    return make_liger_jsd_autograd_pair_fns()


op = declare_op(
    name="jsd",
    level=1,
    family="loss",
    forward="evograd.ops.level1.jsd.forward_ref:jsd_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("log_q", "[rows, cols]"),
        Inactive("target", "[rows, cols]", note="fixed log-probability target log_p"),
    ),
    output=Active("out", "[]", dtype="float32"),
    parameter_args=(),
    forward_semantics=(
        "Generalized JSD with beta=0.5 between target log_p and predicted log_q, "
        "summed over all elements and divided by rows."
    ),
    backward_semantics=(
        "Return dlog_q only: dout * 0.5*q*(log_q-log_m)/rows; target is inactive."
    ),
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("log_q", "target"))
    },
    memory_inputs=("log_q",),
    make_inputs=_inputs,
)
