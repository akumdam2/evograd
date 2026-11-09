"""Shared benchmark declarations, result aggregation, and provenance helpers."""

from .report import TaskResult
from .tasks import WORKLOADS, get_workload

__all__ = [
    "TaskResult",
    "WORKLOADS",
    "get_workload",
]
