"""Operator declaration: SwiGLU."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, make_pair_baseline, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "a": torch.randn((rows, cols), device=device, dtype=dtype),
        "b": torch.randn((rows, cols), device=device, dtype=dtype),
        "dc": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.swiglu.liger import make_liger_swiglu_autograd_pair_fns

    return make_liger_swiglu_autograd_pair_fns()


#: The observed Qwen3-0.6B pointwise boundary. The harvest records a bare SiLU
#: module, but the production boundary is `silu(gate) * up` -- the activation
#: never appears without the multiply -- so it maps here rather than onto a
#: standalone activation task. The SiLU record and the gate/up projection it
#: sits between are kept as supporting provenance in the snapshot.


op = declare_op(
    name="swiglu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.swiglu.forward_ref:swiglu_forward_ref",
    dims=("rows", "cols"),
    args=(Active("a", "[rows, cols]"), Active("b", "[rows, cols]")),
    output=Active("c", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Element-wise c = silu(a) * b, with SiLU evaluated in fp32.",
    backward_semantics=(
        "Return da and db. With s=sigmoid(a), db=dc*(a*s), and "
        "da=dc*b*((a*s)*(1-s)+s). Preserve input dtypes."
    ),
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("a", "b"))
    },
    make_inputs=_inputs,
    # da and db may be written over a and b. Both gradients have exactly the
    # shape of the activation that produced them, and under autograd that
    # activation is dead once the backward has read it, so a SwiGLU backward can
    # skip allocating two [rows, cols] tensors and write in place instead — this
    # is what Liger does. Declaring it here makes the allowance part of the
    # operator's contract, available to every candidate, rather than something
    # one implementation gets away with.
    backward_may_overwrite=("a", "b"),
)
