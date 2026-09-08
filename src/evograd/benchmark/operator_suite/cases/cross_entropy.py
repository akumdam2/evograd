"""Operator-suite performance cases for ``cross_entropy``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.cross_entropy`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_VOCAB_REGIME_SPLIT, LLAMA_VOCAB_TOKEN_SWEEP
from evograd.ops._common import fixed_shape_suites, log_distance_weight, model_workloads, regime_suites, workloads_2d
from evograd.ops.level1.cross_entropy import op as _primitive

_TOLERANCES = _primitive.tolerances

_SHAPES = (
    (4096, 512), (8192, 512), (4096, 4096), (16384, 4096),
    (4096, 8192), (2048, 8192), (4096, 16384), (512, 32000),
    (2048, 32000), (4096, 32000), (8192, 32000), (4096, 65536),
    (4096, 128256), (777, 50257),
)

_SPLIT = LLAMA_VOCAB_REGIME_SPLIT

_LEGACY_BENCHMARK = workloads_2d(_SHAPES, _TOLERANCES, tolerances=_TOLERANCES)

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'logits',
    tuple({'tokens': t} for t in LLAMA_VOCAB_TOKEN_SWEEP),
    ("float16", "bfloat16"),
    tolerances=_TOLERANCES,
)

def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK,
    suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
)
