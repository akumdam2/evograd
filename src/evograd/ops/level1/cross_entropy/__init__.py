"""Operator declaration: hard-label mean cross entropy."""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline, workloads_2d

_TOLERANCES = {
    "float32": (2e-5, 1e-3),
    "float16": (5e-4, 1e-2),
    "bfloat16": (2e-3, 2e-2),
}


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "logits": torch.randn((rows, cols), device=device, dtype=dtype),
        "target": torch.randint(0, cols, (rows,), device=device, dtype=torch.int64),
        "dloss": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.cross_entropy.liger import (
        make_liger_cross_entropy_autograd_pair_fns,
    )

    return make_liger_cross_entropy_autograd_pair_fns()


_CORRECTNESS = (
    workloads_2d(((8, 512), (16, 1024)), ("float32",), tolerances=_TOLERANCES)
    + workloads_2d(
        ((32, 4096), (64, 2048)),
        ("float16", "bfloat16"),
        tolerances=_TOLERANCES,
    )
)

#: The single cross entropy a Qwen3-0.6B step runs, at the shape the flattened
#: call actually receives: [4096, 151936] float32. The BF16 [2, 2048, 151936]
#: logits are upcast and reshaped inside Transformers' causal-loss wrapper, and
#: the snapshot records that wrapper as supporting provenance so the chain from
#: the model's logits to this shape is traceable.

op = declare_op(
    name="cross_entropy",
    level=1,
    family="loss",
    forward="evograd.ops.level1.cross_entropy.forward_ref:cross_entropy_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("logits", "[rows, cols]"),
        Inactive("target", "[rows]", dtype="int64"),
    ),
    output=Active("loss", "[]"),
    parameter_args=(),
    forward_semantics=(
        "Mean-reduced hard-label cross entropy with ignore_index=-100, no label "
        "smoothing, z-loss, or class weights. Compute logsumexp in fp32."
    ),
    backward_semantics=(
        "Return dlogits = dloss * (softmax(logits)-onehot(target)) / rows; "
        "target is inactive and receives no gradient."
    ),
    correctness=_CORRECTNESS,
    tolerances=_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("logits", "target"))
    },
    memory_inputs=("logits",),
    make_inputs=_inputs,
)
