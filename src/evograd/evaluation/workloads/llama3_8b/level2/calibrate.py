"""Level-2 calibration for Llama-3-8B: the shared logic, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level2_calibrate as _common
from evograd.evaluation.workloads.common.level2_calibrate import (  # noqa: F401
    DEFAULT_REPEATS,
    case,
    compare,
    production_results,
    reduction_dims,
    reduction_length,
    reference_results,
)
from evograd.evaluation.workloads.llama3_8b.descriptor import WORKLOAD

#: The boundaries this model presents, in declaration order.
TASKS = WORKLOAD.level2_tasks


def inventory_task(task, **kwargs):
    return _common.inventory_task(task, descriptor=WORKLOAD, **kwargs)


def scaling_study(task, **kwargs):
    return _common.scaling_study(task, descriptor=WORKLOAD, **kwargs)


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
