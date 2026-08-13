"""Operator declaration: rmsnorm."""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
)
from evograd.ops._common import (
    fixed_shape_suites,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    regime_suites,
)


def make_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    # gamma centred on 1.0, not on 0.0. A zero-mean gain is not what any trained
    # RMSNorm holds, and it cancels the signal path when this operator is
    # composed into a block.
    weight = (1.0 + 0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(
        dtype
    )
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "eps": 1e-5, "dy": dy}


def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def rmsnorm_tolerance(workload, result_name, atol, rtol):
    """Account for BF16 error growth in the dweight reduction over rows.

    ``dweight`` sums over every row, so its absolute error grows with the token
    count while ``dx`` does not. Same treatment layernorm already gives its
    parameter gradients.
    """
    if workload.dtype == "bfloat16" and result_name == "dweight":
        rows = workload.dims["rows"]
        atol = max(atol, atol * math.sqrt(max(1.0, rows / 8.0)))
    return atol, rtol


def _liger_factory():
    from evograd.ops.rmsnorm.liger import make_liger_rmsnorm_autograd_pair_fns

    return make_liger_rmsnorm_autograd_pair_fns()


_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "rmsnorm",
    tuple({"tokens": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)
# Untimed compile/runtime coverage: the non-power-of-two hidden width and the
# small row counts the pre-v1 grid used to time, which are worth running but are
# not part of the performance objective.
_COVERAGE = _BENCHMARK + tuple(
    Workload(dims=dict(rows=rows, hidden=hidden), dtype="bfloat16")
    for rows, hidden in ((1, 4096), (17, 4096), (128, 4096), (4096, 8192))
)

op = declare_op(
    name="rmsnorm",
    forward="evograd.ops.rmsnorm.forward_ref:rmsnorm_forward_ref",
    level=1,
    family="norm",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    forward_semantics="Forward computes row-wise RMSNorm over the last dimension of x [rows, hidden]: rrms = rsqrt(mean(x^2, axis=-1, keepdim=True) + eps); y = (x * rrms) * weight. Compute the mean-of-squares and rrms in float32, then cast y back to x's dtype. weight is a 1D tensor of length hidden. Do not call torch RMSNorm/LayerNorm or autograd in the generated math.",
    backward_semantics="Backward must return (dx, dweight). With xhat = x * rrms and g = dy * weight: dx = (g - xhat * mean(g * xhat, axis=-1, keepdim=True)) * rrms, returned in x's dtype and shape [rows, hidden]. dweight = sum(dy * xhat, dim=0), returned in weight's dtype and shape [hidden]. Use float32 accumulation for all reductions.",
    extra_constraints='Tensor layout notes:\n- x, dy: [rows, hidden], contiguous CUDA, float32 or float16\n- weight: [hidden], contiguous CUDA, same dtype as x\n- y, dx: [rows, hidden], same dtype as x\n- There is no bias term and no mean-subtraction (RMSNorm, not LayerNorm).',
    # Ported from benchmark/triton_rmsnorm_backward_bench/task_spec.py.
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="float16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="float16"),
        # bfloat16 is the dtype Llama-3 actually trains in; the pre-v1
        # declaration gated only fp32/fp16, so the measured path was untested.
        Workload(dims=dict(rows=32, hidden=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="bfloat16"),
        Workload(dims=dict(rows=128, hidden=4096), dtype="bfloat16"),
    ),
    coverage=_COVERAGE,
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "coverage": _COVERAGE,
        # Pre-v1 grid: max row width 8192 elements, fp32/fp16 only. Kept as an
        # ablation control, not as the measured objective.
        "legacy": tuple(
            Workload(dims=dict(rows=r, hidden=h), dtype=dtype)
            for (r, h) in (
                (1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
                (17, 127), (17, 513), (17, 1000),
            )
            for dtype in ("float32", "float16")
        ),
    },
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("x", "weight", "eps"))
    },
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    tolerance_hook=rmsnorm_tolerance,
    regime_feature=_regime_feature,
    regime_split=LLAMA_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_REGIME_SPLIT),
    make_inputs=make_rmsnorm_inputs,
)
