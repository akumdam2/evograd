"""Level-1 tolerance calibration for OLMoE-1B-7B: shared logic, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level1_calibrate as _common
from evograd.evaluation.workloads.common.level1_calibrate import (  # noqa: F401
    summarize_calibration,
)
from evograd.evaluation.workloads.olmoe_1b_7b.descriptor import WORKLOAD


def run_calibration(op_name: str, **kwargs):
    return _common.run_calibration(op_name, descriptor=WORKLOAD, **kwargs)
