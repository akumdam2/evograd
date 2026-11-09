"""The benchmark's pooling across tasks, as ``evograd summarize`` applies it.

One :class:`TaskResult` per task. Two rules shape the aggregation.

**Pool by family, then across families.** A flat geometric mean over operators
would let whichever family happens to have the most declarations decide the
headline number. Pooling within a family first and then across families gives
each kind of kernel one vote.

**Coverage is reported, never averaged away.** An operator that fails to build,
fails correctness, or dies on one shape contributes no speedup — it must not
quietly raise the mean by being absent. Coverage counts what ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def geomean(values: list[float]) -> float:
    """Geometric mean of positive measurements; 1.0 for an empty input."""
    positive = [value for value in values if value > 0.0]
    return math.exp(sum(math.log(value) for value in positive) / max(len(positive), 1))


@dataclass(frozen=True)
class TaskResult:
    """One operator's contribution to the suite."""

    op: str
    #: Task hierarchy: 1 primitive, 2 composite, 3 whole-model workload.
    #: What kind of task this is. Not to be confused with `tier`.
    level: int
    family: str
    baseline: str
    #: Evaluation tier: what was measured (a kernel pair, an operator through
    #: autograd, a training step). See `evaluation.common.report.TIERS`.
    tier: str = "pair"
    #: Measurement mode — "fair" or "fast". Rows measured under different
    #: modes are not comparable, so a report that mixes them has
    #: to say which is which per row, not only in its header.
    protocol: str = "fair"
    #: Inputs this operator's contract lets the backward overwrite. Published so
    #: a reader can see which operators were measured under the relaxed rule,
    #: and so a submitter knows the same allowance is theirs.
    backward_may_overwrite: tuple[str, ...] = ()
    speedups: tuple[float, ...] = ()
    cases_total: int = 0
    cases_ok: int = 0
    saved_bytes: float = 0.0
    input_bytes: float = 0.0
    error: str | None = None
    #: Distinct reasons the failing cases gave, in first-seen order. A published
    #: benchmark has to say *why* a case did not run: "the baseline refuses a
    #: 128256-wide row" is a finding about the baseline, while a bare "4 of 4
    #: cases failed" reads as a defect in the harness.
    case_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None and self.cases_ok == self.cases_total > 0

    @property
    def coverage(self) -> float:
        return self.cases_ok / self.cases_total if self.cases_total else 0.0

    @property
    def speedup(self) -> float:
        """Geometric mean across this operator's shapes, or 0.0 if none ran."""
        return geomean(list(self.speedups)) if self.speedups else 0.0

    @property
    def saved_memory_ratio(self) -> float:
        return self.saved_bytes / self.input_bytes if self.input_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "level": self.level,
            "family": self.family,
            "tier": self.tier,
            "protocol": self.protocol,
            "baseline": self.baseline,
            "speedup_full_step": self.speedup,
            "per_case_speedups": list(self.speedups),
            "cases_ok": self.cases_ok,
            "cases_total": self.cases_total,
            "coverage": self.coverage,
            "saved_memory_ratio": self.saved_memory_ratio,
            "ok": self.ok,
            "error": self.error,
            "case_errors": list(self.case_errors),
            "backward_may_overwrite": list(self.backward_may_overwrite),
        }


def _pool_by_family(tasks: Iterable[TaskResult]) -> float:
    """Geomean within each family, then across families.

    Families with nothing measurable are dropped rather than counted as zero: a
    family that failed entirely is a coverage fact, and folding a zero into a
    geometric mean would make the performance number identically zero and hide
    everything else.
    """
    by_family: dict[str, list[float]] = {}
    for task in tasks:
        if task.speedups:
            by_family.setdefault(task.family, []).append(task.speedup)
    family_means = [geomean(values) for values in by_family.values() if values]
    return geomean(family_means) if family_means else 0.0
