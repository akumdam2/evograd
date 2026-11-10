"""Operator-suite cases for ``layernorm_no_affine``: AlphaFold3's own widths, untimed.

The grid is the pair and single widths of the registered AlphaFold3 configuration,
as coverage: a timed grid for this primitive comes from a model's observed cases.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Workload

_SHAPES = ((384, 384), (147456, 128))

cases = SuiteCases(coverage=tuple(
    Workload(dims=dict(rows=rows, hidden=cols), dtype=dtype)
    for rows, cols in _SHAPES for dtype in ("float32", "bfloat16")))
