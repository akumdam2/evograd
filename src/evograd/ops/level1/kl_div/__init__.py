"""Operator declaration: batchmean KL divergence."""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline, workloads_2d

_TOLERANCES = {
    "float32": (1e-6, 1e-4),
    "float16": (1e-4, 1e-2),
    "bfloat16": (5e-4, 2e-2),
}


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    pred_logits = torch.randn((rows, cols), device=device, dtype=torch.float32)
    target_logits = torch.randn((rows, cols), device=device, dtype=torch.float32)
    return {
        "y_pred": torch.log_softmax(pred_logits, dim=-1).to(dtype),
        "y_true": torch.softmax(target_logits, dim=-1).to(dtype),
        "dloss": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.kl_div.liger import make_liger_kl_div_autograd_pair_fns

    return make_liger_kl_div_autograd_pair_fns()


_CORRECTNESS = (
    workloads_2d(((8, 512), (16, 1024)), ("float32",), tolerances=_TOLERANCES)
    + workloads_2d(
        ((32, 4096), (64, 2048)),
        ("float16", "bfloat16"),
        tolerances=_TOLERANCES,
    )
)

op = declare_op(
    name="kl_div",
    level=1,
    family="loss",
    forward="evograd.ops.level1.kl_div.forward_ref:kl_div_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("y_pred", "[rows, cols]", grad="d_input"),
        Inactive("y_true", "[rows, cols]"),
    ),
    output=Active("loss", "[]"),
    parameter_args=(),
    forward_semantics=(
        "Batchmean KL divergence where y_pred is log-probability and y_true is "
        "probability (log_target=False), using fp32 math."
    ),
    backward_semantics=(
        "Return d_input = dloss * (-y_true) / rows. y_true is an inactive target."
    ),
    correctness=_CORRECTNESS,
    tolerances=_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("y_pred", "y_true"))
    },
    memory_inputs=("y_pred",),
    make_inputs=_inputs,
)
