"""What is benchmarked: task identity, cases, provenance, and aggregation.

This package owns the benchmark side of the repository. A task's level, the
cases it is measured on, the shapes and frequencies those cases came from, the
frozen manifests that make them reproducible, and the aggregation that turns
per-task results into a report all live here.

:data:`TASKS` is the one registry an executing caller resolves a name through.
It aggregates the reusable Level-1 primitives that :mod:`evograd.ops` owns,
the generic fusions in :mod:`evograd.benchmark.operator_suite.tasks`, and each
model's own tasks under :mod:`evograd.benchmark.topdown` -- and it refuses two
packages claiming one name.

What it does not own is judgment. Whether a candidate passes, within what
tolerance, against which control, and how long it took are
:mod:`evograd.evaluation`'s.
"""

from .core.registry import TASKS, DuplicateTask, get_task, load_task, tasks_at_level
from .core.tasks import WORKLOADS, get_workload


__all__ = [
    "DuplicateTask",
    "TASKS",
    "WORKLOADS",
    "get_task",
    "get_workload",
    "load_task",
    "tasks_at_level",
]
