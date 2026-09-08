"""Operator-suite performance cases for ``rope``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.rope`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_REGIME_SPLIT, LLAMA_TOKEN_SWEEP
from evograd.ops._common import fixed_shape_suites, log_distance_weight, model_workloads, regime_suites

def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["T"])

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "rope",
    tuple({"batch": 1, "seq": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
) + model_workloads(
    LLAMA_3_8B,
    "rope_kv",
    tuple({"batch": 1, "seq": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK,
    suites={
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
    },
    regime_feature=_regime_feature,
    regime_split=LLAMA_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_REGIME_SPLIT),
)
