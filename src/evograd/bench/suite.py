"""Cross-operator benchmark report: the suite-level view of the whole hierarchy.

Every aggregate evograd computed before v1 was *within* one operator — the
geometric mean of one kernel's speedups across its own shapes. That answers
"is this kernel good", which is what the evolutionary search needs, but it is
not a benchmark result. A benchmark result answers "how does this system do on
the suite", and that requires pooling across operators, which nothing did.

Three rules shape the aggregation here.

**Full-step speedup only.** ``speedup_vs_baseline_backward`` compares
asymmetric things: the eager baseline's "backward" runs through the oracle,
which computes the forward *and* the backward, while the candidate's backward is
timed from pre-saved state with the forward outside the timed region. For a
single elementwise kernel the forward is cheap and the distortion is small; for
a level-3 block the forward is roughly a third of the step, so the number is
inflated by about half. ``speedup_vs_baseline_raw_full_step`` compares
like with like, and it is the metric the benchmark specification defines.

**Pool by family, then across families.** Fourteen of the twenty-five operators
are normalizations, activations and losses; a flat geometric mean over
operators would let whichever family happens to have the most declarations
decide the headline number. Pooling within a family first and then across
families gives each kind of kernel one vote.

**Coverage is reported, never averaged away.** An operator that fails to build,
fails correctness, or dies on one shape contributes no speedup — it must not
quietly raise the mean by being absent. Coverage counts what ran.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from evograd.bench.report import BenchReport, from_fair_report, from_harness_report
from evograd.evolve.scoring import geomean

#: The harness key this suite once read to get its speedup. It is no longer
#: read anywhere: `CaseMetrics.speedup_full` derives the ratio from the times a
#: protocol reports, so the asymmetric backward-only metric is not reachable by
#: a wrong lookup. Kept because the published report names its own metric, and
#: because the name is the one a reader of an older report will search for.
FULL_STEP_SPEEDUP_KEY = "speedup_vs_baseline_raw_full_step"

REPORT_VERSION = "evograd-suite-v1"

LEVEL_NAMES = {1: "primitive", 2: "fused", 3: "block"}


@dataclass(frozen=True)
class TaskResult:
    """One operator's contribution to the suite."""

    op: str
    #: Task hierarchy: 1 primitive, 2 fused, 3 block. What kind of operator
    #: this is. Not to be confused with `tier`.
    level: int
    family: str
    baseline: str
    #: Evaluation tier: what was measured (a kernel pair, an operator through
    #: autograd, a training step). See `bench.report.TIERS`.
    tier: str = "pair"
    #: How carefully it was measured — "fair" or "fast". Rows measured under
    #: different protocols are not comparable, so a report that mixes them has
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
    #: benchmark has to say *why* a case did not run: "Liger's sparsemax refuses
    #: a 128256-wide row" is a finding about the baseline, while a bare "4 of 4
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


def task_from_report(
    report: BenchReport,
    *,
    level: int,
    family: str,
) -> TaskResult:
    """Turn a canonical :class:`BenchReport` into one suite row.

    This is the only place a measurement becomes a suite result, so the rules
    that decide what a benchmark number means live here once, whatever protocol
    or tier produced the samples:

    * the speedup is the like-for-like full step, derived from times by
      :class:`CaseMetrics` — there is no key to read, and so no asymmetric
      backward-only key to read by mistake;
    * a case that did not run contributes no speedup and is still counted, so
      an operator that works on four shapes out of five reports 80% coverage
      rather than four shapes' worth of speedup;
    * saved and input bytes are summed over the cases that ran, matching the
      denominator each protocol was given.
    """
    speedups: list[float] = []
    case_errors: list[str] = []
    saved_bytes = 0.0
    input_bytes = 0.0
    for case in report.cases:
        if not case.ok:
            reason = case.error or ""
            if reason and reason not in case_errors:
                case_errors.append(reason)
            continue
        speedup = case.speedup_full
        if speedup is None or not math.isfinite(speedup) or speedup <= 0.0:
            continue
        speedups.append(speedup)
        saved_bytes += case.saved_bytes
        input_bytes += case.input_bytes

    return TaskResult(
        op=report.op,
        level=level,
        family=family,
        tier=report.tier,
        protocol=report.protocol,
        baseline=report.baseline,
        speedups=tuple(speedups),
        cases_total=len(report.cases),
        cases_ok=len(speedups),
        saved_bytes=saved_bytes,
        input_bytes=input_bytes,
        error=report.error,
        case_errors=tuple(case_errors),
    )


def task_from_fair_report(
    op_name: str,
    level: int,
    family: str,
    baseline: str,
    report: dict[str, Any],
    *,
    backward_may_overwrite: tuple[str, ...] = (),
    op=None,
) -> TaskResult:
    """Read one ``run_fair_benchmarks`` report into the suite's task shape."""
    task = task_from_report(
        from_fair_report(op_name, report, baseline=baseline, op=op),
        level=level,
        family=family,
    )
    return replace(task, backward_may_overwrite=tuple(backward_may_overwrite))


def task_from_benchmark_report(
    op_name: str,
    level: int,
    family: str,
    report: dict[str, Any],
) -> TaskResult:
    """Read one ``run_benchmarks(..., on_error="record")`` report."""
    return task_from_report(
        from_harness_report(op_name, report),
        level=level,
        family=family,
    )



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


def _level_block(tasks: list[TaskResult]) -> dict[str, Any]:
    total_cases = sum(task.cases_total for task in tasks)
    ok_cases = sum(task.cases_ok for task in tasks)
    return {
        "operators": len(tasks),
        "operators_ok": sum(1 for task in tasks if task.ok),
        "speedup_full_step_macro": _pool_by_family(tasks),
        "coverage_cases": ok_cases / total_cases if total_cases else 0.0,
        "coverage_operators": (
            sum(1 for task in tasks if task.ok) / len(tasks) if tasks else 0.0
        ),
        "families": sorted({task.family for task in tasks}),
    }


@dataclass
class SuiteReport:
    tasks: list[TaskResult] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    #: What was measured — a candidate directory, or a reviewed baseline
    #: standing in as the candidate. A suite result is unreadable without it:
    #: "1.6x at level 1" means something entirely different when the thing
    #: being timed is Liger than when it is a generated kernel.
    candidate_source: str = "unspecified"
    #: Which timing protocol produced the samples. The suite's own aggregation
    #: is identical either way, but the numbers are not comparable across the
    #: two: the low-overhead harness clears no cache and randomizes no order.
    timing_protocol: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        by_level: dict[int, list[TaskResult]] = {}
        for task in self.tasks:
            by_level.setdefault(task.level, []).append(task)

        total_cases = sum(task.cases_total for task in self.tasks)
        ok_cases = sum(task.cases_ok for task in self.tasks)
        measured = [task for task in self.tasks if task.speedups]
        return {
            "protocol": REPORT_VERSION,
            "candidate": self.candidate_source,
            "timing_protocol": self.timing_protocol,
            "metric": {
                "speedup": FULL_STEP_SPEEDUP_KEY,
                "definition": "S_i = T_reference_full_step / T_candidate_full_step",
                "aggregation": (
                    "geometric mean within an operator, then within a family, "
                    "then across families"
                ),
                "why_not_backward_only": (
                    "the eager baseline's backward timing includes its forward "
                    "while the candidate's does not, which inflates the ratio by "
                    "roughly the forward's share of the step"
                ),
            },
            "environment": self.environment,
            "overall": {
                "speedup_full_step_macro": _pool_by_family(self.tasks),
                "coverage_cases": ok_cases / total_cases if total_cases else 0.0,
                "coverage_operators": (
                    sum(1 for task in self.tasks if task.ok) / len(self.tasks)
                    if self.tasks
                    else 0.0
                ),
                # Only operators that actually reported retained bytes. A
                # geometric mean over a set containing zeros collapses to 1.0,
                # which reads as "saves exactly its inputs" — the most
                # unremarkable value possible — when the truth is that nothing
                # was measured. Reporting 0.0 for an empty set says so.
                "saved_memory_ratio_geomean": (
                    geomean(ratios)
                    if (ratios := [
                        t.saved_memory_ratio for t in measured
                        if t.saved_memory_ratio > 0
                    ])
                    else 0.0
                ),
                "operators": len(self.tasks),
            },
            "levels": {
                str(level): {
                    "name": LEVEL_NAMES.get(level, str(level)),
                    **_level_block(tasks),
                }
                for level, tasks in sorted(by_level.items())
            },
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines = [
            "# evograd benchmark suite",
            "",
            f"**Measured: {data['candidate']}.**",
            "",
            f"**Timing: {data['timing_protocol']}.**",
            "",
            f"Protocol `{data['protocol']}`. "
            f"Speedup is full-step (`{FULL_STEP_SPEEDUP_KEY}`): "
            "forward plus backward, measured the same way on both sides.",
            "",
            "| level | operators | full-step speedup | case coverage | operator coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for level, block in data["levels"].items():
            lines.append(
                f"| {level} ({block['name']}) | {block['operators']} "
                f"| {block['speedup_full_step_macro']:.3f}x "
                f"| {block['coverage_cases']:.0%} "
                f"| {block['coverage_operators']:.0%} |"
            )
        overall = data["overall"]
        lines.extend(
            [
                f"| **all** | **{overall['operators']}** "
                f"| **{overall['speedup_full_step_macro']:.3f}x** "
                f"| **{overall['coverage_cases']:.0%}** "
                f"| **{overall['coverage_operators']:.0%}** |",
                "",
                f"Saved-state memory, geometric mean of saved/input bytes: "
                f"{overall['saved_memory_ratio_geomean']:.3f}.",
                "",
                "## Per operator",
                "",
                "| op | level | family | baseline | speedup | coverage | saved/input |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for task in sorted(
            data["tasks"], key=lambda t: (t["level"], t["family"], t["op"])
        ):
            speedup = (
                f"{task['speedup_full_step']:.3f}x"
                if task["per_case_speedups"]
                else "—"
            )
            lines.append(
                f"| `{task['op']}` | {task['level']} | {task['family']} "
                f"| {task['baseline']} | {speedup} "
                f"| {task['coverage']:.0%} ({task['cases_ok']}/{task['cases_total']}) "
                f"| {task['saved_memory_ratio']:.3f} |"
            )
        relaxed = [t for t in data["tasks"] if t.get("backward_may_overwrite")]
        if relaxed:
            lines.extend(
                [
                    "",
                    "## Inputs the backward may overwrite",
                    "",
                    "These operators' contracts allow the backward to write a "
                    "gradient over the input that produced it, so those buffers "
                    "are exempt from the mutation check — for every candidate, "
                    "not only the baseline. Shape, strides, dtype and storage "
                    "offset are still enforced.",
                    "",
                ]
            )
            for task in relaxed:
                names = ", ".join(f"`{n}`" for n in task["backward_may_overwrite"])
                lines.append(f"- `{task['op']}`: {names}")

        failures = [t for t in data["tasks"] if not t["ok"]]
        if failures:
            lines.extend(["", "## Not fully covered", ""])
            for task in failures:
                detail = task["error"] or (
                    f"{task['cases_total'] - task['cases_ok']} of "
                    f"{task['cases_total']} cases failed"
                )
                lines.append(f"- `{task['op']}`: {detail}")
                # Why, not just how many. A shape that the baseline refuses to
                # run is a result about the baseline and belongs in the report.
                for reason in task.get("case_errors", ()):
                    lines.append(f"  - {reason}")
        return "\n".join(lines) + "\n"


def write_report(report: SuiteReport, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "suite_report.json"
    markdown_path = output_dir / "SUITE_RESULTS.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}
