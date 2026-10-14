"""Operator-suite performance cases for ``gdn_gating``.

The Gated DeltaNet's decay and write-strength gates, one row per token. Derived from the published
Qwen3.5-4B text configuration -- the model in this repository that runs
this primitive -- at its canonical batch-1 x 2048 step and above it. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.gdn_gating`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl.models import QWEN3_5_4B
from evograd.ops._common import fixed_shape_suites, model_workloads

_TOKENS = tuple({"tokens": t} for t in (2048, 4096, 8192))

_DERIVED = tuple(
    workload
    for component, sweep in (
        ("gdn_gating", _TOKENS),
    )
    for workload in model_workloads(QWEN3_5_4B, component, sweep, ("bfloat16",))
)


cases = SuiteCases(
    benchmark=_DERIVED,
    coverage=_DERIVED,
    suites={**fixed_shape_suites(_DERIVED)},
)
