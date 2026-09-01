"""Operator declaration: polynomial RMS-normalized features."""

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
    (16, 1024), (64, 1024), (256, 1024), (1024, 1024), (3072, 1024),
    (8192, 1024), (16384, 1024), (32768, 1024), (65536, 1024),
    (131072, 1024), (1024, 4096), (8192, 4096), (49152, 4096),
    (2048, 8192),
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
    if result_name in {"dweight", "dbias"}:
        atol = _REDUCED_ATOL.get(workload.dtype, atol)
    return atol, rtol


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "weight": torch.randn((3, cols), device=device, dtype=dtype),
        "bias": torch.randn((cols,), device=device, dtype=dtype),
        "eps": 1e-6,
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.poly_norm.liger import make_liger_poly_norm_autograd_pair_fns

    return make_liger_poly_norm_autograd_pair_fns()


op = declare_op(
    name="poly_norm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.poly_norm.forward_ref:poly_norm_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("weight", "[3, cols]"),
        Active("bias", "[cols]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("out", "[rows, cols]"),
    parameter_args=("weight", "bias"),
    forward_semantics=(
        "For p in {3,2,1}, row-normalize x**p by its RMS, then return "
        "weight[0]*n3 + weight[1]*n2 + weight[2]*n1 + bias in x dtype."
    ),
    backward_semantics="Return dx, dweight, dbias; parameter gradients reduce across rows.",
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
            _liger_factory, ("x", "weight", "bias", "eps"), ("eps",)
        )
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
