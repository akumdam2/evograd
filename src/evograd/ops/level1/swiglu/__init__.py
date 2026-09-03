"""Operator declaration: SwiGLU."""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
)
from evograd.ops._common import (
    STANDARD_TOLERANCES,
    dtype_for,
    fixed_shape_suites,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    observed_workloads,
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
    from evograd.ops.level1.swiglu.liger import make_liger_swiglu_autograd_pair_fns

    return make_liger_swiglu_autograd_pair_fns()


#: The observed Qwen3-0.6B pointwise boundary. The harvest records a bare SiLU
#: module, but the production boundary is `silu(gate) * up` -- the activation
#: never appears without the multiply -- so it maps here rather than onto a
#: standalone activation task. The SiLU record and the gate/up projection it
#: sits between are kept as supporting provenance in the snapshot.
_QWEN3_OBSERVED = observed_workloads("qwen3_0_6b", "swiglu", tolerances=STANDARD_TOLERANCES)

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
    coverage=_BENCHMARK + _QWEN3_OBSERVED,
    benchmark=_BENCHMARK,
    benchmark_suites={
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
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
    # da and db may be written over a and b. Both gradients have exactly the
    # shape of the activation that produced them, and under autograd that
    # activation is dead once the backward has read it, so a SwiGLU backward can
    # skip allocating two [rows, cols] tensors and write in place instead — this
    # is what Liger does. Declaring it here makes the allowance part of the
    # operator's contract, available to every candidate, rather than something
    # one implementation gets away with.
    backward_may_overwrite=("a", "b"),
)
