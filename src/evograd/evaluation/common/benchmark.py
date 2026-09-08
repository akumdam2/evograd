"""Translate evaluation reports into benchmark aggregation rows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from evograd.benchmark.core.report import TaskResult, task_from_report
from evograd.evaluation.common.report import from_fair_report, from_harness_report


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
    return task_from_report(
        from_harness_report(op_name, report),
        level=level,
        family=family,
    )


__all__ = ["task_from_benchmark_report", "task_from_fair_report"]
