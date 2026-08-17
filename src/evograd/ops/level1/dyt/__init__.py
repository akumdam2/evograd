"""Operator declaration: dynamic tanh (DyT)."""

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
    (16, 1024), (64, 1024), (256, 1024), (1024, 1024), (1536, 1024),
    (2048, 4096), (8192, 1024), (12288, 1024), (16384, 1024),
    (32768, 1024), (49152, 1024), (65024, 1024), (12288, 4096),
    (61440, 3072), (8192, 8192),
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
    if result_name in {"dalpha", "dgamma", "dbeta"}:
        atol = _REDUCED_ATOL.get(workload.dtype, atol)
    return atol, rtol


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    alpha = 0.5 + 0.1 * torch.randn((), device=device, dtype=dtype)
    gamma = 1.0 + 0.1 * torch.randn((cols,), device=device, dtype=dtype)
    beta = 0.1 * torch.randn((cols,), device=device, dtype=dtype)
    return {
        "x": x,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.level1.dyt.liger import make_liger_dyt_autograd_pair_fns

    return make_liger_dyt_autograd_pair_fns()


op = declare_op(
    name="dyt",
    level=1,
    family="norm",
    forward="evograd.ops.level1.dyt.forward_ref:dyt_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("x", "[rows, cols]"),
        Active("alpha", "[]"),
        Active("gamma", "[cols]"),
        Active("beta", "[cols]"),
    ),
    output=Active("out", "[rows, cols]"),
    forward_semantics="out = gamma * tanh(alpha*x) + beta, computed in fp32.",
    backward_semantics=(
        "Return dx, dalpha, dgamma, dbeta. Parameter gradients reduce across rows; "
        "dalpha reduces over every element."
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
            _liger_factory, ("x", "alpha", "gamma", "beta")
        )
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
