"""Operator declaration: fused residual add plus RMSNorm."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
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
_REDUCED_ATOL = {"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _tolerance(workload, result_name, atol, rtol):
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
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
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
    output=Active("out", "[rows, cols]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Set s=x+r, compute rstd=rsqrt(mean(s**2,lastdim)+eps) in fp32, "
        "and return s*rstd*weight."
    ),
    backward_semantics=(
        "Return dx, dr, dweight. dx and dr are the same gradient through s; "
        "dweight reduces dout*s*rstd across rows."
    ),
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=STANDARD_TOLERANCES,
    tolerance_hook=_tolerance,
    performance_baselines={
        "liger": make_pair_baseline(
            _liger_factory, ("x", "r", "weight", "eps"), ("eps",)
        )
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
