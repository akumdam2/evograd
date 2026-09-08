"""Operator-suite performance cases for ``evoattention``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.evoattention`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import fixed_shape_suites, model_workloads

_DTYPES = ("float16", "bfloat16")

_PAIR_BIAS = model_workloads(
    ALPHAFOLD3,
    "pair_bias_attention",
    tuple({"batch": 1, "n_seq": 1, "residues": n} for n in AF3_RESIDUE_SWEEP),
    _DTYPES,
    scaled=True,
    note="MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU",
)

_TRIANGLE = model_workloads(
    ALPHAFOLD3,
    "triangle_attention",
    tuple({"batch": 1, "residues": n} for n in AF3_RESIDUE_SWEEP[:2]),
    _DTYPES,
    scaled=True,
    note=(
        "MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU. "
        "Residues stop at 256 because S == N makes this the most expensive family"
    ),
)

_MSA = model_workloads(
    ALPHAFOLD3,
    "msa_attention",
    tuple({"batch": 1, "n_seq": 64, "residues": n} for n in AF3_RESIDUE_SWEEP[:2]),
    _DTYPES,
    scaled=True,
    note="MegaFold benchmarks batch 4; batch 1 keeps the grid on a single GPU",
)

_BENCHMARK = _PAIR_BIAS + _TRIANGLE + _MSA

_STRESS = tuple(
    Workload(dims=dict(B=1, S=1, H=8, N=n, D=128), dtype=dtype)
    for n in (256, 384)
    for dtype in _DTYPES
)


cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK + _STRESS,
    suites={
        "pair_bias": _PAIR_BIAS,
        "triangle": _TRIANGLE,
        "msa": _MSA,
        **fixed_shape_suites(_BENCHMARK),
        # Not an AlphaFold3 configuration: no AF3 module uses head dim 128 (the
        # largest is 64). These probe register pressure, which is worth running
        # but is not evidence about protein-model performance, so they are an
        # ablation suite and untimed coverage rather than part of the objective.
        "register_pressure": _STRESS,
        # Pre-v1 grid. Retained for comparison with earlier runs; note its
        # triangle-attention cases set S != N, which understates the real work.
        "legacy": tuple(
            Workload(dims=dict(B=b, S=s, H=h, N=n, D=d), dtype=dtype)
            for (b, s, h, n, d) in (
                (1, 1, 16, 128, 64), (1, 1, 16, 256, 64), (1, 1, 16, 384, 64),
                (1, 64, 4, 128, 32), (1, 64, 4, 256, 32), (1, 128, 4, 128, 32),
                (1, 1, 8, 256, 128), (1, 1, 8, 384, 128),
            )
            for dtype in ("float16", "bfloat16")
        ),
    },
)
