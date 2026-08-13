"""Operator declaration: sparsemax."""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP_FP32,
)
from evograd.ops._common import (
    fixed_shape_suites,
    model_workloads,
    dtype_for,
    log_distance_weight,
    make_pair_baseline,
    regime_suites,
    standard_correctness,
    workloads_2d,
)

_TOLERANCES = {"float32": (2e-5, 2e-5)}
_SHAPES = (
    (4096, 128), (4096, 256), (4096, 512), (4096, 1024),
    (4096, 2048), (4096, 4096), (4096, 8192), (4096, 12288),
    (4096, 16384), (4096, 30522), (4096, 32768), (2048, 49152),
    (2048, 65536), (1024, 98304), (1024, 128256),
)
_SPLIT = LLAMA_VOCAB_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(_SHAPES, ("float32",), tolerances=_TOLERANCES)
# Timed grid derived from Llama-3-8B, so every case names the layer it
# came from. The pre-v1 hand-picked grid above is kept as an ablation
# suite rather than deleted.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP_FP32),
    ("float32",),
    tolerances=_TOLERANCES,
)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _liger_factory():
    from evograd.ops.sparsemax.liger import make_liger_sparsemax_autograd_pair_fns

    return make_liger_sparsemax_autograd_pair_fns()


op = declare_op(
    name="sparsemax",
    level=1,
    family="reduction",
    forward="evograd.ops.sparsemax.forward_ref:sparsemax_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]", dtype="float32"),),
    output=Active("out", "[rows, cols]", dtype="float32"),
    forward_semantics="Project each row onto the probability simplex with sparsemax.",
    backward_semantics=(
        "On support S={i:out_i>0}, return dx_i=dout_i-mean_S(dout); "
        "return zero outside S."
    ),
    correctness=workloads_2d(
        ((8, 64), (17, 128), (32, 256), (64, 512)),
        ("float32",),
        tolerances=_TOLERANCES,
    ),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("x",))},
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
