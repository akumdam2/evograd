"""Level-2 negative controls for Qwen3-0.6B: shared logic, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level2_negative_controls as _common
from evograd.evaluation.workloads.qwen3_0_6b.descriptor import WORKLOAD

#: The boundaries these controls sweep.
TASKS = WORKLOAD.level2_tasks


def run_task(task: str, **kwargs):
    return _common.run_task(task, descriptor=WORKLOAD, **kwargs)


def summarize(report):
    return _common.summarize(report)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
