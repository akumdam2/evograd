"""Operator-suite performance cases for ``softmax``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.softmax`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_VOCAB_REGIME_SPLIT, LLAMA_VOCAB_TOKEN_SWEEP
from evograd.ops._common import STANDARD_TOLERANCES, fixed_shape_suites, log_distance_weight, model_workloads, regime_suites, workloads_2d

_SHAPES = (
    (8192, 512), (4096, 512), (4096, 4096), (16384, 4096),
    (4096, 8192), (2048, 8192), (4096, 16384), (2048, 16384),
    (4096, 32768), (2048, 65536), (4096, 65536), (1024, 131072),
)

_SPLIT = LLAMA_VOCAB_REGIME_SPLIT

_LEGACY_BENCHMARK = workloads_2d(_SHAPES, STANDARD_TOLERANCES)

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP),
    ("float16", "bfloat16"),
    tolerances=STANDARD_TOLERANCES,
)

def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


cases = SuiteCases(
    benchmark=_BENCHMARK,
    suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
)
