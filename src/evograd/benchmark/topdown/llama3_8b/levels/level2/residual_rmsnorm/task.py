"""Level-2 task: the Llama-3-8B residual add followed by RMSNorm.

The same fusion Qwen3-0.6B runs at this boundary, at Llama-3-8B's residual
width -- 4096 against 1024 -- and at its own frequency. Same mathematics,
different case, its own task key.

The timed case is derived from the frozen Llama-3-8B configuration rather than
from a harvest snapshot: the shape is a property of the architecture, and
``tests/test_provenance`` re-derives it. Batch 2 x sequence 2048 gives 4096
tokens, as it does for Qwen3; only the width differs.
"""

from evograd.benchmark.topdown.common.level2_references import (
    residual_rms_norm_forward_ref as _definition,
    residual_rms_norm_runtime_ref as _runtime,
)
from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import (
    STANDARD_TOLERANCES,
    dtype_for,
    model_workloads,
    standard_correctness,
)

#: The workload these cases belong to; also the ``Provenance`` model key.
WORKLOAD = "llama_3_8b"

#: The observed configuration, computed from the published model config.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "residual_rmsnorm",
    ({"tokens": 4096},),
    ("bfloat16",),
    tolerances=STANDARD_TOLERANCES,
)

#: How often the fusion occurs in one Llama-3-8B step, derived from the
#: architecture rather than counted by hand: 32 attention residual adds, 31 MLP
#: adds into the next layer's input_layernorm, and one final MLP add into
#: model.norm. 64, against Qwen3-0.6B's 56 -- the arithmetic is each
#: architecture's own, and neither may borrow the other's.
LLAMA_FUSION_SITES = LLAMA_3_8B.residual_rmsnorm_fusion_sites()
assert LLAMA_FUSION_SITES["total"] == 64, LLAMA_FUSION_SITES

_REDUCED_ATOL = {"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}

def _tolerance(workload, result_name, atol, rtol):
    """Row-count-aware slack for the one result that reduces over rows.

    ``dweight`` sums ``dnormalized * summed * rstd`` over every row, so its
    error grows with the number of terms while every other result's does not.
    Kept from the single-output declaration and re-measured for the two-output
    contract by ``benchmark.topdown.qwen3_0_6b.levels.level2.residual_rmsnorm calibrate``: at the canonical
    4096x1024 BF16 case the measured requirement is 4.6e-02 against the 2.5e-01
    this yields, and no other result needs a hook at all.
    """
    if result_name == "dweight":
        base = _REDUCED_ATOL.get(workload.dtype, atol)
        atol = base * max(1.0, (workload.dims["rows"] / 64.0) ** 0.5)
    return atol, rtol

def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "r": torch.randn((rows, cols), device=device, dtype=dtype),
        "weight": torch.randn((cols,), device=device, dtype=dtype),
        "eps": 1e-6,
        # Two independent upstream gradients, both non-zero. Drawn separately on
        # purpose: a backward that ignored `dsummed`, or that assumed the two
        # were equal, would pass against a shared tensor.
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
        "dsummed": torch.randn((rows, cols), device=device, dtype=dtype),
    }

op = declare_op(
    name="llama3_residual_rmsnorm",
    level=2,
    family="norm",
    forward=(
        "evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.reference:"
        "llama3_residual_rmsnorm_forward_ref"
    ),
    runtime_forward=(
        "evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.reference:"
        "llama3_residual_rmsnorm_runtime_ref"
    ),
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("r", "[rows, cols]"),
        Active("weight", "[cols]"),
        Inactive("eps", default=1e-6),
    ),
    output=(
        Active("out", "[rows, cols]"),
        Active("summed", "[rows, cols]"),
    ),
    parameter_args=("weight",),
    forward_semantics="Set s = x + r; compute rstd = rsqrt(mean(s**2, lastdim) + eps) with "
        "the reduction in float32; return (out, summed) IN THAT ORDER where "
        "out = s * rstd * weight and summed = s. Both are [rows, cols] and "
        "have the input's dtype. `summed` is the un-normalized sum itself, not "
        "a copy or a recomputation: the next block consumes it, which is why "
        "the fusion returns it instead of forcing a second pass.",
    backward_semantics="The backward receives output_grads = (dout, dsummed), one per output, "
        "and returns dx, dr, dweight IN THIS ORDER. Both paths reach s: with "
        "dnorm = dout * weight, "
        "dtotal = dsummed + rstd * (dnorm - s * mean(dnorm * s, lastdim) * "
        "rstd**2). Then dx = dtotal and dr = dtotal -- they are the same "
        "tensor, because s = x + r. dweight comes only from the normalized "
        "path: dweight = sum over rows of dout * s * rstd. Ignoring dsummed is "
        "the characteristic error here; it leaves dx and dr wrong by exactly "
        "the gradient that reaches the residual stream without passing through "
        "the norm, and leaves dweight untouched, so a dweight-only check will "
        "not catch it. Accumulate the row reductions in float32.",
    correctness=standard_correctness(),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"llama_3_8b_observed": _BENCHMARK},
    tolerances=STANDARD_TOLERANCES,
    tolerance_multipliers={"summed": (0.01, 0.01)},
    tolerance_hook=_tolerance,
    make_inputs=_inputs,
    backward_may_overwrite=(),
)
