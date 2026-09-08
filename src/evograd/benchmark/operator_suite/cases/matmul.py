"""Operator-suite performance cases for ``matmul``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.matmul`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import fixed_shape_suites, model_workloads

_GEMM_COMPONENTS = ("attn_qkv", "mlp_up", "mlp_down", "lm_head")

_GEMM_TOKENS = (2048, 8192)

def _derived_gemm_workloads(dtypes=("bfloat16",)):
    return tuple(
        workload
        for component in _GEMM_COMPONENTS
        for workload in model_workloads(
            LLAMA_3_8B,
            component,
            tuple({"tokens": tokens} for tokens in _GEMM_TOKENS),
            dtypes,
        )
    )

_LEGACY_SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 1024, 2048),
    (2048, 1024, 2048),
    (4096, 1024, 1024),
)

_INDUSTRIAL_BF16_SHAPES = (
    (256, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1024, 1024),
    (1024, 4096, 4096),
    (4096, 4096, 1024),
    (2048, 4096, 4096),
    (8192, 4096, 14336),
)

_LARGE_BF16_SHAPES = tuple(
    shape for shape in _INDUSTRIAL_BF16_SHAPES if shape[1] >= 4096
)

_EXTREME_BF16_SHAPES = ((8192, 4096, 14336),)

def _workloads(shapes, dtypes):
    return tuple(
        Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
        for m, k, n in shapes
        for dtype in dtypes
    )


cases = SuiteCases(
    benchmark=_derived_gemm_workloads(),
    coverage=_workloads(_INDUSTRIAL_BF16_SHAPES, ("bfloat16",)),
    suites={
        **fixed_shape_suites(_derived_gemm_workloads()),
        "industrial_bf16": _workloads(_INDUSTRIAL_BF16_SHAPES, ("bfloat16",)),
        "large_bf16": _workloads(_LARGE_BF16_SHAPES, ("bfloat16",)),
        "extreme_bf16": _workloads(_EXTREME_BF16_SHAPES, ("bfloat16",)),
        "legacy": _workloads(_LEGACY_SHAPES, ("float32", "float16")),
    },
)
