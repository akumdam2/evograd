"""Operator-suite performance cases for ``swiglu``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.swiglu`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_REGIME_SPLIT, LLAMA_TOKEN_SWEEP
from evograd.opdecl.models import ALPHAFOLD3, AF3_RESIDUE_SWEEP
from evograd.ops._common import STANDARD_TOLERANCES, fixed_shape_suites, log_distance_weight, model_workloads, regime_suites, workloads_2d

#: MegaFold's production shape for the pairformer-trunk site at each crop of
#: the residue sweep (batch 1). Declared from the frozen AlphaFold3
#: configuration, not harvested; the Level-4 MegaFold task patches this site at
#: exactly these dims. A named suite plus its fixed/ cases, deliberately not
#: folded into the timed grid.
_MEGAFOLD_TRUNK = model_workloads(
    ALPHAFOLD3, "pair_transition_swiglu",
    tuple({"residues": r} for r in AF3_RESIDUE_SWEEP), ('bfloat16', 'float32',),
)

_SHAPES = (
    (1, 512), (8, 1024), (32, 2048), (8, 4096), (1, 8192),
    (64, 4096), (128, 4096), (256, 4096), (512, 4096), (2048, 4096),
    (512, 14336), (2048, 14336), (17, 255), (17, 1001),
)

_SPLIT = LLAMA_REGIME_SPLIT

_LEGACY_BENCHMARK = workloads_2d(_SHAPES, STANDARD_TOLERANCES)

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    'mlp_activation',
    tuple({'tokens': t} for t in LLAMA_TOKEN_SWEEP),
    ("float32", "float16", "bfloat16"),
    tolerances=STANDARD_TOLERANCES,
)

def _feature(workload: Workload) -> float:
    return float(workload.dims["rows"])


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK,
    suites={
        **regime_suites(_BENCHMARK, _feature, _SPLIT),
        **fixed_shape_suites(_BENCHMARK),
        "megafold_trunk": _MEGAFOLD_TRUNK,
        **fixed_shape_suites(_MEGAFOLD_TRUNK),
        "legacy": _LEGACY_BENCHMARK,
    },
    regime_feature=_feature,
    regime_split=_SPLIT,
    case_weight=log_distance_weight(_feature, _SPLIT),
)
