"""Operator declaration: fused residual add plus RMSNorm, with both outputs.

This is the physical task behind the logical "residual_rmsnorm" boundary: the
decoder's residual stream, where an add is immediately followed by a norm and
the *un-normalized sum* is needed again by the next block. It returns both, and
the backward has to combine the two paths -- see ``forward_ref`` for why one
output was the wrong contract.
"""

from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
    QWEN3_0_6B,
)
from evograd.ops._common import (
    fixed_shape_suites,
    model_workloads,
    STANDARD_TOLERANCES,
    dtype_for,
    log_distance_weight,
    make_pair_baseline,
    regime_suites,
    standard_correctness,
    workloads_2d,
)

_SHAPES = (
    (16, 1024), (64, 1024), (256, 1024), (1024, 1024), (2048, 1024),
    (4096, 1024), (8192, 1024), (16384, 1024), (32768, 1024),
    (65536, 1024), (131072, 1024), (8192, 4096), (32768, 4096),
    (4096, 8192), (12345, 3072), (49152, 1536),
)
_SPLIT = LLAMA_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(
    _SHAPES, ("bfloat16",), tolerances=STANDARD_TOLERANCES
)
# Timed grid derived from Llama-3-8B, so every case names the layer it
# came from. The pre-v1 hand-picked grid above is kept as an ablation
# suite rather than deleted.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'elementwise',
    tuple({'tokens': t} for t in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
    tolerances=STANDARD_TOLERANCES,
)
#: The Qwen3-0.6B configuration this fusion site actually runs at, observed in
#: the Level-4 harvest: one row per token of the canonical batch-2 x
#: sequence-2048 step, one column per residual channel.
_QWEN3_OBSERVED = model_workloads(
    QWEN3_0_6B,
    "residual_rmsnorm",
    ({"tokens": 4096},),
    ("bfloat16",),
    tolerances=STANDARD_TOLERANCES,
)

#: How often that configuration occurs in one Qwen3-0.6B step, derived from the
#: architecture rather than counted by hand. 28 attention residual adds each
#: followed by their layer's post_attention_layernorm, 27 MLP residual adds each
#: followed by the *next* layer's input_layernorm, and one final MLP add
#: followed by model.norm. Layer 0's input_layernorm has no preceding decoder
#: residual add and is deliberately not counted -- which is why this is 56 and
#: not the 57 residual-width RMSNorm invocations the harvest observed.
QWEN3_FUSION_SITES = QWEN3_0_6B.residual_rmsnorm_fusion_sites()
assert QWEN3_FUSION_SITES["total"] == 56, QWEN3_FUSION_SITES

_REDUCED_ATOL = {"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _tolerance(workload, result_name, atol, rtol):
    """Row-count-aware slack for the one result that reduces over rows.

    ``dweight`` sums ``dnormalized * summed * rstd`` over every row, so its
    error grows with the number of terms while every other result's does not.
    Kept from the single-output declaration and re-measured for the two-output
    contract by ``bench.workloads.qwen3.levels.level2.residual_rmsnorm calibrate``: at the canonical
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


def _liger_factory():
    from evograd.ops.level2.fused_add_rms_norm.liger import (
        make_liger_fused_add_rms_norm_autograd_pair_fns,
    )

    return make_liger_fused_add_rms_norm_autograd_pair_fns()


op = declare_op(
    name="fused_add_rms_norm",
    level=2,
    family="norm",
    forward=(
        "evograd.ops.level2.fused_add_rms_norm.forward_ref:"
        "fused_add_rms_norm_forward_ref"
    ),
    runtime_forward=(
        "evograd.ops.level2.fused_add_rms_norm.forward_ref:"
        "fused_add_rms_norm_runtime_ref"
    ),
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("r", "[rows, cols]"),
        Active("weight", "[cols]"),
        Inactive("eps", default=1e-6),
    ),
    # Two outputs, `out` first so the former single output stays primary.
    output=(
        Active("out", "[rows, cols]"),
        Active("summed", "[rows, cols]"),
    ),
    parameter_args=("weight",),
    forward_semantics=(
        "Set s = x + r; compute rstd = rsqrt(mean(s**2, lastdim) + eps) with "
        "the reduction in float32; return (out, summed) IN THAT ORDER where "
        "out = s * rstd * weight and summed = s. Both are [rows, cols] and "
        "have the input's dtype. `summed` is the un-normalized sum itself, not "
        "a copy or a recomputation: the next block consumes it, which is why "
        "the fusion returns it instead of forcing a second pass."
    ),
    backward_semantics=(
        "The backward receives output_grads = (dout, dsummed), one per output, "
        "and returns dx, dr, dweight IN THIS ORDER. Both paths reach s: with "
        "dnorm = dout * weight, "
        "dtotal = dsummed + rstd * (dnorm - s * mean(dnorm * s, lastdim) * "
        "rstd**2). Then dx = dtotal and dr = dtotal -- they are the same "
        "tensor, because s = x + r. dweight comes only from the normalized "
        "path: dweight = sum over rows of dout * s * rstd. Ignoring dsummed is "
        "the characteristic error here; it leaves dx and dr wrong by exactly "
        "the gradient that reaches the residual stream without passing through "
        "the norm, and leaves dweight untouched, so a dweight-only check will "
        "not catch it. Accumulate the row reductions in float32."
    ),
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
        # The observed Qwen3-0.6B configuration, kept as its own suite so the
        # generic Llama-derived grid stays the default timed set.
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
    },
    tolerances=STANDARD_TOLERANCES,
    # `summed` is a plain elementwise add, and every spelling measured requires
    # *exactly* 0 for it -- at float32, float16 and bfloat16 alike, on both the
    # correctness cases and the canonical 4096x1024 invocation. Holding it to
    # the same tolerance as the normalized output would let a candidate return
    # a sum that was 8e-2 wrong and call it fused. 100x tighter still leaves
    # room for a legitimately different rounding order; it is not exactness.
    tolerance_multipliers={"summed": (0.01, 0.01)},
    tolerance_hook=_tolerance,
    performance_baselines={
        "liger": make_pair_baseline(
            _liger_factory, ("x", "r", "weight", "eps"), ("eps",)
        )
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    coverage=_BENCHMARK + _QWEN3_OBSERVED,
    make_inputs=_inputs,
)
