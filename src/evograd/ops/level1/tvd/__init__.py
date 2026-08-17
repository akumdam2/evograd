"""Operator declaration: batchmean total-variation distance."""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP,
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
    (16, 64), (64, 128), (128, 512), (257, 1000), (512, 1024),
    (1024, 2048), (1536, 1536), (4096, 1024), (4096, 3072),
    (4096, 4096), (8192, 8192), (65536, 1024), (32768, 4096),
    (131072, 2048),
)
_SPLIT = LLAMA_VOCAB_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(
    _SHAPES, ("bfloat16",), tolerances=STANDARD_TOLERANCES
)
# Timed grid derived from Llama-3-8B, so every case names the layer it
# came from. The pre-v1 hand-picked grid above is kept as an ablation
# suite rather than deleted.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP),
    ("bfloat16",),
    tolerances=STANDARD_TOLERANCES,
)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    p = torch.softmax(torch.randn((rows, cols), device=device, dtype=dtype), dim=-1)
    q = torch.softmax(torch.randn((rows, cols), device=device, dtype=dtype), dim=-1)
    return {
        "p": p,
        "q": q,
        "dout": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.tvd.liger import make_liger_tvd_autograd_pair_fns

    return make_liger_tvd_autograd_pair_fns()


op = declare_op(
    name="tvd",
    level=1,
    family="loss",
    forward="evograd.ops.level1.tvd.forward_ref:tvd_forward_ref",
    dims=("rows", "cols"),
    args=(Active("p", "[rows, cols]"), Active("q", "[rows, cols]")),
    output=Active("out", "[]"),
    forward_semantics="out = 0.5 * sum(abs(p-q)) / rows.",
    backward_semantics=(
        "Return dp=dout*0.5*sign(p-q)/rows and dq=-dp, using the zero "
        "subgradient for ties."
    ),
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={"liger": make_pair_baseline(_liger_factory, ("p", "q"))},
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
