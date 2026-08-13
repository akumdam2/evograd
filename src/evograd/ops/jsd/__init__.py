"""Operator declaration: generalized Jensen-Shannon divergence."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
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
    (1, 1024), (8, 2048), (32, 4096), (257, 1536), (512, 1024),
    (1024, 2048), (4096, 1024), (4096, 4096), (12345, 4096),
    (8192, 8192), (32768, 4096), (4096, 50257), (65536, 4096),
    (2048, 128256),
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
    log_q = torch.log_softmax(
        torch.randn((rows, cols), device=device, dtype=dtype), dim=-1
    )
    target = torch.log_softmax(
        torch.randn((rows, cols), device=device, dtype=dtype), dim=-1
    )
    return {
        "log_q": log_q,
        "target": target,
        "dout": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.jsd.liger import make_liger_jsd_autograd_pair_fns

    return make_liger_jsd_autograd_pair_fns()


op = declare_op(
    name="jsd",
    level=1,
    family="loss",
    forward="evograd.ops.jsd.forward_ref:jsd_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("log_q", "[rows, cols]"),
        Inactive("target", "[rows, cols]", note="fixed log-probability target log_p"),
    ),
    output=Active("out", "[]", dtype="float32"),
    forward_semantics=(
        "Generalized JSD with beta=0.5 between target log_p and predicted log_q, "
        "summed over all elements and divided by rows."
    ),
    backward_semantics=(
        "Return dlog_q only: dout * 0.5*q*(log_q-log_m)/rows; target is inactive."
    ),
    correctness=standard_correctness(),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=STANDARD_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("log_q", "target"))
    },
    memory_inputs=("log_q",),
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
