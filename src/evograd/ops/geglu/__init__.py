"""Operator declaration: tanh-approximate GeGLU."""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import (
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
_SPLIT = 1_000_000
_BENCHMARK = workloads_2d(_SHAPES, STANDARD_TOLERANCES)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"] * workload.dims["cols"])


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
    from evograd.ops.geglu.liger import make_liger_geglu_autograd_pair_fns

    return make_liger_geglu_autograd_pair_fns()


op = declare_op(
    name="geglu",
    forward="evograd.ops.geglu.forward_ref:geglu_forward_ref",
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
    benchmark_suites=regime_suites(_BENCHMARK, _feature, _SPLIT),
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("a", "b"))
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
