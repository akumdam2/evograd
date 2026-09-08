"""Operator-suite performance cases for ``conv2d``.

Moved out of the primitive declaration: the shapes worth timing, where the
regime splits, how cases are weighted, and the named suites a report selects
by. The contract, the reference and the generic correctness cases stay in
:mod:`evograd.ops.level1.conv2d`.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.ops.level1.conv2d import op as _primitive
from evograd.ops.level1.conv2d import CASE_PROVENANCE, CASE_DIMS, workload_case

_BENCHMARK_CASES = (
    # B, C, H, W, O, KH, KW, OH, OW (stride=1, padding=0)
    (32, 64, 56, 56, 64, 3, 3, 54, 54),
    (32, 64, 56, 56, 128, 3, 3, 54, 54),
    (16, 256, 28, 28, 256, 3, 3, 26, 26),
    (16, 256, 28, 28, 512, 1, 1, 28, 28),
    (8, 512, 14, 14, 512, 3, 3, 12, 12),
    (8, 512, 14, 14, 1024, 1, 1, 14, 14),
)

def _benchmark_workloads():
    return tuple(workload_case(values, "bfloat16") for values in _BENCHMARK_CASES)


cases = SuiteCases(
    benchmark=_benchmark_workloads(),
    coverage=_benchmark_workloads(),
    suites={"cnn_bf16": _benchmark_workloads()},
)
