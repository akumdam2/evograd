"""Operator-suite performance cases for ``jsd``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.jsd`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_VOCAB_REGIME_SPLIT, LLAMA_VOCAB_TOKEN_SWEEP
from evograd.ops._common import STANDARD_TOLERANCES, fixed_shape_suites, log_distance_weight, model_workloads, regime_suites, workloads_2d

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

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP),
    ("bfloat16",),
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
