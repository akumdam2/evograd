"""Operator declaration: squared ReLU."""

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
    (1, 256), (16, 256), (64, 512), (257, 769), (512, 1024),
    (1024, 1024), (2048, 1024), (4096, 3072), (8192, 4096),
    (16384, 4096), (32768, 4096), (65536, 4096), (8192, 50257),
    (131072, 2048),
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
    'mlp_activation',
    tuple({'tokens': t} for t in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
    tolerances=STANDARD_TOLERANCES,
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
    from evograd.ops.level1.relu_squared.liger import (
        make_liger_relu_squared_autograd_pair_fns,
    )

    return make_liger_relu_squared_autograd_pair_fns()


op = declare_op(
    name="relu_squared",
    level=1,
    family="activation",
    forward="evograd.ops.level1.relu_squared.forward_ref:relu_squared_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("out", "[rows, cols]"),
    forward_semantics="Element-wise out = relu(x)^2.",
    backward_semantics="Return dx = dout * 2*x where x>0 and zero otherwise.",
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("x",))},
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
