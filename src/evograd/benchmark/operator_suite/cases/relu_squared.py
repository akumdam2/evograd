"""Operator-suite performance cases for ``relu_squared``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.relu_squared`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_REGIME_SPLIT, LLAMA_TOKEN_SWEEP
from evograd.ops._common import STANDARD_TOLERANCES, fixed_shape_suites, log_distance_weight, model_workloads, regime_suites, workloads_2d

_SHAPES = (
    (1, 256), (16, 256), (64, 512), (257, 769), (512, 1024),
    (1024, 1024), (2048, 1024), (4096, 3072), (8192, 4096),
    (16384, 4096), (32768, 4096), (65536, 4096), (8192, 50257),
    (131072, 2048),
)

_SPLIT = LLAMA_REGIME_SPLIT

_LEGACY_BENCHMARK = workloads_2d(
    _SHAPES, ("bfloat16",), tolerances=STANDARD_TOLERANCES
)

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'mlp_activation',
    tuple({'tokens': t} for t in LLAMA_TOKEN_SWEEP),
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
