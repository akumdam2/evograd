"""Benchmark tasks, workload provenance, and cross-task aggregation.

Operators remain declared in :mod:`evograd.ops`.  This package selects those
operators into implementation-neutral suites and owns top-down architectural
and whole-model workloads.
"""

from .core.tasks import WORKLOADS, get_workload


__all__ = ["WORKLOADS", "get_workload"]
