"""Operator-suite performance cases for ``gemma_rmsnorm``.

The residual-stream norm and the per-head q/k norms of gated attention. Derived from the published
Qwen3.5-4B text configuration -- the model in this repository that runs
this primitive -- at its canonical batch-1 x 2048 step and above it. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.gemma_rmsnorm`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl.models import QWEN3_5_4B
from evograd.ops._common import fixed_shape_suites, model_workloads

_TOKENS = tuple({"tokens": t} for t in (2048, 4096, 8192))
_BATCH_SEQ = tuple({"batch": b, "seq": 2048} for b in (1, 2, 4))

_DERIVED = tuple(
    workload
    for component, sweep in (
        ("residual_norm", _TOKENS),
        ("attn_q_norm", _BATCH_SEQ),
        ("attn_k_norm", _BATCH_SEQ),
    )
    for workload in model_workloads(QWEN3_5_4B, component, sweep, ("bfloat16",))
)


cases = SuiteCases(
    benchmark=_DERIVED,
    coverage=_DERIVED,
    suites={**fixed_shape_suites(_DERIVED)},
)
