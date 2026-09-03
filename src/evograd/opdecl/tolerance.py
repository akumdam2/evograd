"""Dimension-aware tolerance hooks.

A per-dtype tolerance and a per-result multiplier are two constants, and two
constants cannot describe a quantity that grows with the workload. BF16 rounding
inside a reduction does grow with it: a parameter gradient is a sum over tokens,
and both the value and its error accumulate with the number of terms. A grid
whose largest reduction is 32 terms says nothing about the 4096 the model runs.

:class:`ReductionScaledAtol` is the shape that says it. It leaves the declared
tolerance exactly alone at and below a named **anchor** workload -- the one the
multipliers were measured on -- and grows the *absolute* tolerance above it by a
law with two measured terms:

    raw(result, dims) = sqrt(N / N_anchor) * sqrt(log M / log M_anchor)

``N`` is the length of the result's declared reduction and ``M`` is its element
count. The first term is the random walk: summing ``N`` independently rounded
terms moves the result by ``sqrt(N)`` and its error by ``sqrt(N)`` too. The
second is extreme-value growth: ``allclose`` is a maximum over elements, and the
maximum of ``M`` samples of a bounded-variance error grows like ``sqrt(log M)``.
Both are measured before they are declared -- see the calibration artifact each
declaration cites.

``rtol`` is never touched. Relative error is what stays constant across scales;
it is the absolute floor that has to move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReductionScaledAtol:
    """``atol`` grows above an anchor workload; ``rtol`` and the anchor do not.

    ``gain`` multiplies the *excess* over the anchor, not the tolerance:

        factor = 1 + gain * max(0, raw - 1)

    so a workload at or below the anchor gets ``factor == 1`` and keeps exactly
    the tolerance it has today. That is the property that makes this safe to add
    to a calibrated declaration: nothing that passes now becomes easier to pass.
    The safety margin lands only where the measurement said it was needed.
    """

    #: The workload the declared multipliers were measured on.
    anchor_dims: dict[str, int]
    #: result name -> the declared dims its reduction sums over. A result absent
    #: from this map has no reduction and only the element-count term applies.
    reduction_dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: result name -> the declared shape dims, for the element count.
    result_dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Safety factor on the measured excess. 2.0 means "twice as far past the
    #: anchor as the measurement said", which is the margin the artifact reports.
    gain: float = 2.0

    def _extent(self, names, dims: dict[str, int]) -> int:
        total = 1
        for name in names:
            total *= max(int(dims.get(name, 1)), 1)
        return total

    def raw_factor(self, result_name: str, dims: dict[str, int]) -> float:
        reduction = self.reduction_dims.get(result_name, ())
        shape = self.result_dims.get(result_name, ())
        n_now = self._extent(reduction, dims)
        n_anchor = self._extent(reduction, self.anchor_dims)
        m_now = self._extent(shape, dims)
        m_anchor = self._extent(shape, self.anchor_dims)
        walk = math.sqrt(n_now / n_anchor) if n_anchor > 0 else 1.0
        # log of 1 is 0; a single-element anchor has no extreme-value growth to
        # measure against, so the term is dropped rather than divided by zero.
        extreme = (
            math.sqrt(math.log(m_now) / math.log(m_anchor))
            if m_now > 1 and m_anchor > 1
            else 1.0
        )
        return walk * extreme

    def factor(self, result_name: str, dims: dict[str, int]) -> float:
        return 1.0 + self.gain * max(0.0, self.raw_factor(result_name, dims) - 1.0)

    def __call__(self, workload, result_name, atol, rtol):
        if result_name is None:
            return atol, rtol
        return atol * self.factor(result_name, dict(workload.dims)), rtol

    def describe(self) -> dict:
        return {
            "kind": "reduction_scaled_atol",
            "formula": "atol *= 1 + gain * max(0, sqrt(N/N_a) * sqrt(log M / log M_a) - 1)",
            "gain": self.gain,
            "anchor_dims": dict(self.anchor_dims),
            "reduction_dims": {k: list(v) for k, v in self.reduction_dims.items()},
            "rtol": "unchanged",
        }
