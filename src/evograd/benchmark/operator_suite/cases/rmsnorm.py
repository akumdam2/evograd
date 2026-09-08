"""Operator-suite performance cases for ``rmsnorm``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.rmsnorm`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_REGIME_SPLIT, LLAMA_TOKEN_SWEEP
from evograd.ops._common import fixed_shape_suites, log_distance_weight, model_workloads, regime_suites

def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["rows"])

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "rmsnorm",
    tuple({"tokens": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)

_COVERAGE = _BENCHMARK + tuple(
    Workload(dims=dict(rows=rows, hidden=hidden), dtype="bfloat16")
    for rows, hidden in ((1, 4096), (17, 4096), (128, 4096), (4096, 8192))
)


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_COVERAGE,
    suites={
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "coverage": _COVERAGE,
        # Pre-v1 grid: max row width 8192 elements, fp32/fp16 only. Kept as an
        # ablation control, not as the measured objective.
        "legacy": tuple(
            Workload(dims=dict(rows=r, hidden=h), dtype=dtype)
            for (r, h) in (
                (1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
                (17, 127), (17, 513), (17, 1000),
            )
            for dtype in ("float32", "float16")
        ),
    },
    regime_feature=_regime_feature,
    regime_split=LLAMA_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_REGIME_SPLIT),
)
