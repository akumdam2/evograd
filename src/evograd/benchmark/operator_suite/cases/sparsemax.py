"""Operator-suite performance cases for ``sparsemax``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.sparsemax`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import LLAMA_VOCAB_REGIME_SPLIT, LLAMA_VOCAB_TOKEN_SWEEP_FP32
from evograd.ops._common import fixed_shape_suites, log_distance_weight, regime_suites, workloads_2d
from evograd.ops.level1.sparsemax import op as _primitive

_TOLERANCES = _primitive.tolerances

_SHAPES = (
    (4096, 128), (4096, 256), (4096, 512), (4096, 1024),
    (4096, 2048), (4096, 4096), (4096, 8192), (4096, 12288),
    (4096, 16384), (4096, 30522), (4096, 32768), (2048, 49152),
    (2048, 65536), (1024, 98304), (1024, 128256),
)

_SPLIT = LLAMA_VOCAB_REGIME_SPLIT

_LEGACY_BENCHMARK = workloads_2d(_SHAPES, ("float32",), tolerances=_TOLERANCES)

_PROVENANCE = Provenance(
    model="sparsemax_paper",
    component="row_projection",
    source="handpicked",
    note=(
        "no shipped architecture contains sparsemax, so its width is chosen "
        "rather than derived: 32768 is the scale of a mid-sized vocabulary "
        "(Llama-2's is 32000) and sits inside the range Triton implementations "
        "support, unlike Llama-3's 128256. Rows keep the token sweep the other "
        "vocabulary-width operators use, so the shape regimes still split"
    ),
)

_TIMED_SHAPES = tuple((tokens, 32768) for tokens in LLAMA_VOCAB_TOKEN_SWEEP_FP32)

_BENCHMARK = tuple(
    Workload(
        dims={"rows": rows, "cols": cols},
        dtype="float32",
        atol=_TOLERANCES["float32"][0],
        rtol=_TOLERANCES["float32"][1],
        provenance=_PROVENANCE,
    )
    for rows, cols in _TIMED_SHAPES
)

_COVERAGE = _BENCHMARK + tuple(
    Workload(
        dims={"rows": rows, "cols": cols},
        dtype="float32",
        atol=_TOLERANCES["float32"][0],
        rtol=_TOLERANCES["float32"][1],
        provenance=_PROVENANCE,
    )
    for rows, cols in ((1024, 98304), (1024, 128256))
)

def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_COVERAGE,
    suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "legacy": _LEGACY_BENCHMARK,
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
)
