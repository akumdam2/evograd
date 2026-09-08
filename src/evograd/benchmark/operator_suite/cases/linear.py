"""Operator-suite performance cases for ``linear``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.linear`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import fixed_shape_suites, model_workloads

_GEMM_COMPONENTS = ("attn_qkv", "mlp_up", "mlp_down", "lm_head")

_GEMM_TOKENS = (2048, 8192)

_BIAS_ABLATION_NOTE = (
    "Llama-3-8B's projection widths measured through a bias-carrying Linear "
    "contract. Llama-3 has no projection biases, so the bias, its broadcast add "
    "and the dbias reduction are an ablation on top of the real configuration; "
    "the faithful grid is in linear_no_bias."
)

_DERIVED = tuple(
    workload
    for component in _GEMM_COMPONENTS
    for workload in model_workloads(
        LLAMA_3_8B,
        component,
        tuple({"tokens": tokens} for tokens in _GEMM_TOKENS),
        ("bfloat16",),
        scaled=True,
        note=_BIAS_ABLATION_NOTE,
    )
)


cases = SuiteCases(
    benchmark=_DERIVED,
    coverage=_DERIVED,
    suites={
        **fixed_shape_suites(_DERIVED),
        # Pre-v1 square grid, retained as an ablation control.
        "legacy": tuple(
            Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
            for (m, n, k) in (
                (512, 512, 512), (1024, 1024, 1024), (2048, 1024, 1024),
                (1024, 2048, 1024), (2048, 2048, 1024), (4096, 1024, 1024),
            )
            for dtype in ("float32", "float16")
        ),
    },
)
