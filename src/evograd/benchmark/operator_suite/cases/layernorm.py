"""Operator-suite performance cases for ``layernorm``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.layernorm`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_TOKEN_SWEEP
from evograd.ops._common import fixed_shape_suites, log_distance_weight, model_workloads, regime_suites

_LEGACY_MIXED = ((1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
                 (17, 127), (17, 513), (17, 1000))

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

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "layernorm",
    tuple({"tokens": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)

_REGIME_SUITES = regime_suites(_BENCHMARK, _regime_feature, _REGIME_SPLIT)


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_workloads(_COVERAGE),
    suites={
        **{name: _workloads(shapes) for name, shapes in _SHAPE_SUITES.items()},
        **_REGIME_SUITES,
        **fixed_shape_suites(_BENCHMARK),
        "coverage": _workloads(_COVERAGE),
    },
    regime_feature=_regime_feature,
    regime_split=_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, _REGIME_SPLIT),
)
