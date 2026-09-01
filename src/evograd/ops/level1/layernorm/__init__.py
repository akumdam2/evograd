"""Operator declaration: layernorm."""

import math

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_TOKEN_SWEEP
from evograd.ops._common import (
    fixed_shape_suites,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    regime_suites,
)


def make_layernorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    bias = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "eps": 1e-5, "dy": dy}


_LEGACY_MIXED = ((1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
                 (17, 127), (17, 513), (17, 1000))
# Pre-regime hand-picked grids retained for smoke / ablation only.
_HAND_SMALL = ((1, 768), (8, 1024), (17, 127), (17, 513), (17, 1000))
_HAND_LARGE = ((1024, 1024), (4096, 1024), (16384, 1024), (65536, 1024), (131072, 1024))
_LEGACY_SMALL = ((1, 256), (4, 512), (8, 768), (17, 127))
_LEGACY_LARGE = ((32, 1536), (8, 4096), (1, 8192), (64, 8192))
_REGIME_SPLIT = 4096.0


def _tb(i):
    return (2**i // 1024, 1024)


_TB_SWEEP = tuple(_tb(i) for i in range(12, 28))
_TB_MIXED = tuple(_tb(i) for i in (12, 14, 16, 18, 20, 22, 24, 26, 27))
_LLM_INDUSTRIAL = (
    (17, 1000),      # non-power-of-two hidden boundary
    (128, 4096),     # small token batch, common LLM hidden
    (2048, 4096),    # common training scale
    (8192, 4096),    # industrial token rows
    (1024, 8192),    # large-model hidden
)
_INDUSTRIAL_MIXED = _TB_MIXED + _LLM_INDUSTRIAL
_COVERAGE = _TB_SWEEP + _LLM_INDUSTRIAL
_SHAPE_SUITES = {
    "mixed": _INDUSTRIAL_MIXED,
    "industrial_mixed": _INDUSTRIAL_MIXED,
    "all": _INDUSTRIAL_MIXED,
    "legacy_mixed": _LEGACY_MIXED,
    "hand_small": _HAND_SMALL,
    "hand_large": _HAND_LARGE,
    "legacy_small": _LEGACY_SMALL,
    "legacy_large": _LEGACY_LARGE,
    "tb_small": tuple(_tb(i) for i in range(12, 17)),
    "tb_medium": tuple(_tb(i) for i in range(17, 23)),
    "tb_large": tuple(_tb(i) for i in range(23, 28)),
    "tb_mixed": _TB_MIXED,
    "tb_sweep": _TB_SWEEP,
    # Exact scaling sweep used for compiler-vs-evolved comparisons: total
    # elements double from 2^13 through 2^27 while hidden stays fixed at 1024.
    "tb_sweep_13_27": tuple(_tb(i) for i in range(13, 28)),
    **{f"tb_i{i}": (_tb(i),) for i in range(12, 28)},
}


def _workloads(shapes, dtypes=("bfloat16",)):
    return tuple(
        Workload(dims=dict(rows=rows, hidden=hidden), dtype=dtype)
        for rows, hidden in shapes
        for dtype in dtypes
    )


def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


# The timed suite is derived from Llama-3-8B's residual width so every case is
# traceable to a real layer. The pre-v1 hand-picked and TritonBench grids are
# retained below as named ablation suites: they remain useful controls, they are
# just not what the headline number is measured on.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "layernorm",
    tuple({"tokens": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)
_REGIME_SUITES = regime_suites(_BENCHMARK, _regime_feature, _REGIME_SPLIT)


def layernorm_tolerance(workload, result_name, atol, rtol):
    """Account for BF16 error growth in long parameter-gradient reductions."""
    if workload.dtype == "bfloat16" and result_name in {"dweight", "dbias"}:
        rows = workload.dims["rows"]
        atol = max(atol, atol * math.sqrt(max(1.0, rows / 8.0)))
    return atol, rtol


def _liger_factory():
    from liger_kernel.ops.layer_norm import layer_norm_backward, layer_norm_forward

    def forward(x, weight, bias, eps):
        y, x_2d, mean, rstd, _block_size, _num_warps = layer_norm_forward(
            x, weight, bias, eps
        )
        return y, (x_2d, weight, bias, mean, rstd)

    def backward(dy, saved):
        x_2d, weight, bias, mean, rstd = saved
        dy_2d = dy.view(-1, dy.shape[-1]).contiguous()
        return layer_norm_backward(dy_2d, x_2d, weight, bias, mean, rstd)

    return forward, backward


measure_liger_baseline = make_pair_baseline(
    _liger_factory, ("x", "weight", "bias", "eps")
)

op = declare_op(
    name="layernorm",
    forward="evograd.ops.level1.layernorm.forward_ref:layernorm_forward_ref",
    runtime_forward="evograd.ops.level1.layernorm.forward_ref:layernorm_runtime_ref",
    level=1,
    family="norm",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Active("bias", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight", "bias"),
    forward_semantics='Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. Forward must produce the same `y` as row-wise LayerNorm over the last dimension.',
    backward_semantics='Backward must consume only `dy`, `saved_tensors`, and `eps`. Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.',
    # Correctness retains all supported dtypes. Evolution performance follows
    # Liger's BF16 industrial grid; rows < 4096 are the small regime.
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="float16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="float16"),
        Workload(dims=dict(rows=32, hidden=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="bfloat16"),
        Workload(dims=dict(rows=128, hidden=4096), dtype="bfloat16"),
        Workload(dims=dict(rows=1024, hidden=8192), dtype="bfloat16"),
    ),
    coverage=_workloads(_COVERAGE),
    benchmark=_BENCHMARK,
    benchmark_suites={
        **{name: _workloads(shapes) for name, shapes in _SHAPE_SUITES.items()},
        **_REGIME_SUITES,
        **fixed_shape_suites(_BENCHMARK),
        "coverage": _workloads(_COVERAGE),
    },
    performance_baselines={"liger": measure_liger_baseline},
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    tolerance_hook=layernorm_tolerance,
    regime_feature=_regime_feature,
    regime_split=_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, _REGIME_SPLIT),
    make_inputs=make_layernorm_inputs,
)
