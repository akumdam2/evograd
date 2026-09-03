"""Operator declaration: hard-label mean cross entropy."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_VOCAB_REGIME_SPLIT,
    LLAMA_VOCAB_TOKEN_SWEEP,
)
from evograd.ops._common import (
    dtype_for,
    fixed_shape_suites,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    qwen3_observed_workloads,
    regime_suites,
    workloads_2d,
)

_TOLERANCES = {
    "float32": (2e-5, 1e-3),
    "float16": (5e-4, 1e-2),
    "bfloat16": (2e-3, 2e-2),
}
_SHAPES = (
    (4096, 512), (8192, 512), (4096, 4096), (16384, 4096),
    (4096, 8192), (2048, 8192), (4096, 16384), (512, 32000),
    (2048, 32000), (4096, 32000), (8192, 32000), (4096, 65536),
    (4096, 128256), (777, 50257),
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
    return {
        "logits": torch.randn((rows, cols), device=device, dtype=dtype),
        "target": torch.randint(0, cols, (rows,), device=device, dtype=torch.int64),
        "dloss": torch.rand((), device=device, dtype=dtype) + 0.5,
    }


def _liger_factory():
    from evograd.ops.level1.cross_entropy.liger import (
        make_liger_cross_entropy_autograd_pair_fns,
    )

    return make_liger_cross_entropy_autograd_pair_fns()


_CORRECTNESS = (
    workloads_2d(((8, 512), (16, 1024)), ("float32",), tolerances=_TOLERANCES)
    + workloads_2d(
        ((32, 4096), (64, 2048)),
        ("float16", "bfloat16"),
        tolerances=_TOLERANCES,
    )
)

#: The single cross entropy a Qwen3-0.6B step runs, at the shape the flattened
#: call actually receives: [4096, 151936] float32. The BF16 [2, 2048, 151936]
#: logits are upcast and reshaped inside Transformers' causal-loss wrapper, and
#: the snapshot records that wrapper as supporting provenance so the chain from
#: the model's logits to this shape is traceable.
_QWEN3_OBSERVED = qwen3_observed_workloads("cross_entropy", tolerances=_TOLERANCES)

op = declare_op(
    name="cross_entropy",
    level=1,
    family="loss",
    forward="evograd.ops.level1.cross_entropy.forward_ref:cross_entropy_forward_ref",
    dims=("rows", "cols"),
    args=(
        Active("logits", "[rows, cols]"),
        Inactive("target", "[rows]", dtype="int64"),
    ),
    output=Active("loss", "[]"),
    parameter_args=(),
    forward_semantics=(
        "Mean-reduced hard-label cross entropy with ignore_index=-100, no label "
        "smoothing, z-loss, or class weights. Compute logsumexp in fp32."
    ),
    backward_semantics=(
        "Return dlogits = dloss * (softmax(logits)-onehot(target)) / rows; "
        "target is inactive and receives no gradient."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK + _QWEN3_OBSERVED,
    benchmark=_BENCHMARK,
    benchmark_suites={
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    tolerances=_TOLERANCES,
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("logits", "target"))
    },
    memory_inputs=("logits",),
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
    make_inputs=_inputs,
)
