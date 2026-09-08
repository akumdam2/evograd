"""Operator-suite performance cases for ``linear_no_bias``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.linear_no_bias`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import fixed_shape_suites, model_workloads

_GEMM_COMPONENTS = ("attn_qkv", "mlp_up", "mlp_down", "lm_head")

_GEMM_TOKENS = (2048, 8192)

_DERIVED = tuple(
    workload
    for component in _GEMM_COMPONENTS
    for workload in model_workloads(
        LLAMA_3_8B,
        component,
        tuple({"tokens": tokens} for tokens in _GEMM_TOKENS),
        ("bfloat16",),
    )
)


cases = SuiteCases(
    benchmark=_DERIVED,
    coverage=_DERIVED,
    suites={
        **fixed_shape_suites(_DERIVED),
    },
)
