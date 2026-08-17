"""Operator declaration: tanh-approximate GeGLU."""

from evograd.opdecl import Active, Workload, declare_op
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
    (1, 512), (8, 1024), (32, 2048), (8, 4096), (1, 8192),
    (64, 4096), (128, 4096), (256, 4096), (512, 4096), (2048, 4096),
    (512, 14336), (2048, 14336), (17, 255), (17, 1001),
)
_SPLIT = LLAMA_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(_SHAPES, STANDARD_TOLERANCES)
# Timed grid derived from Llama-3-8B, so every case names the layer it
# came from. The pre-v1 hand-picked grid above is kept as an ablation
# suite rather than deleted.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'mlp_activation',
    tuple({'tokens': t} for t in LLAMA_TOKEN_SWEEP),
    ("float32", "float16", "bfloat16"),
    tolerances=STANDARD_TOLERANCES,
)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


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
    from evograd.ops.level1.geglu.liger import make_liger_geglu_autograd_pair_fns

    return make_liger_geglu_autograd_pair_fns()


op = declare_op(
    name="geglu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.geglu.forward_ref:geglu_forward_ref",
    dims=("rows", "cols"),
    args=(Active("a", "[rows, cols]"), Active("b", "[rows, cols]")),
    output=Active("c", "[rows, cols]"),
    forward_semantics=(
        "Element-wise c = gelu(a, approximate='tanh') * b. Use the tanh "
        "approximation, fp32 intermediates, and preserve input dtype."
    ),
    backward_semantics="Return da and db for both differentiable inputs.",
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("a", "b"))
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
