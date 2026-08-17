"""Operator declaration: batchmean KL divergence."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP,
)
from evograd.ops._common import (
    fixed_shape_suites,
    model_workloads,
    dtype_for,
    log_distance_weight,
    make_pair_baseline,
    regime_suites,
    workloads_2d,
)

_TOLERANCES = {
    "float32": (1e-6, 1e-4),
    "float16": (1e-4, 1e-2),
    "bfloat16": (5e-4, 2e-2),
}
_SHAPES = (
    (4096, 512), (8192, 512), (4096, 4096), (16384, 4096),
    (4096, 8192), (2048, 8192), (4096, 16384), (512, 32000),
    (2048, 32000), (4096, 32000), (4096, 65536), (4096, 128256),
    (777, 50257),
)
_SPLIT = LLAMA_VOCAB_REGIME_SPLIT
_LEGACY_BENCHMARK = workloads_2d(_SHAPES, _TOLERANCES, tolerances=_TOLERANCES)
# Timed grid derived from Llama-3-8B, so every case names the layer it
# came from. The pre-v1 hand-picked grid above is kept as an ablation
# suite rather than deleted.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP),
    ("float16", "bfloat16"),
    tolerances=_TOLERANCES,
)


def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    pred_logits = torch.randn((rows, cols), device=device, dtype=torch.float32)
    target_logits = torch.randn((rows, cols), device=device, dtype=torch.float32)
    return {
        "y_pred": torch.log_softmax(pred_logits, dim=-1).to(dtype),
        "y_true": torch.softmax(target_logits, dim=-1).to(dtype),
        "dloss": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.kl_div.liger import make_liger_kl_div_autograd_pair_fns

    return make_liger_kl_div_autograd_pair_fns()


_CORRECTNESS = (
    workloads_2d(((8, 512), (16, 1024)), ("float32",), tolerances=_TOLERANCES)
    + workloads_2d(
        ((32, 4096), (64, 2048)),
        ("float16", "bfloat16"),
        tolerances=_TOLERANCES,
    )
)

op = declare_op(
    name="kl_div",
    level=1,
    family="loss",
    forward="evograd.ops.level1.kl_div.forward_ref:kl_div_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("y_pred", "[rows, cols]", grad="d_input"),
        Inactive("y_true", "[rows, cols]"),
    ),
    output=Active("loss", "[]"),
    forward_semantics=(
        "Batchmean KL divergence where y_pred is log-probability and y_true is "
        "probability (log_target=False), using fp32 math."
    ),
    backward_semantics=(
        "Return d_input = dloss * (-y_true) / rows. y_true is an inactive target."
    ),
    correctness=_CORRECTNESS,
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("y_pred", "y_true"))
    },
    memory_inputs=("y_pred",),
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
