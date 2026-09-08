"""Shared benchmark declarations, result aggregation, and provenance helpers."""

from .report import (
    SuiteReport,
    TaskResult,
    task_from_report,
    task_from_tier3_report,
    write_report,
)
from .tasks import WORKLOADS, get_workload

__all__ = [
    "SuiteReport",
    "TaskResult",
    "task_from_report",
    "task_from_tier3_report",
    "write_report",
    "WORKLOADS",
    "get_workload",
]
