"""Level-2 calibration inventory for Qwen3.5-4B: the shared logic, bound to this model.

    python -m evograd.evaluation.workloads.qwen3_5_4b.level2.calibrate \
        inventory --task qwen3_5_gated_delta_rule --report <run_dir>/inventory/<task>.json

Oracle (``op.forward``) against production (``op.runtime_forward``) only --
never a library baseline or a candidate. Three populations per task: the
declared grid, the observed suite, and the site's real invocation
(:mod:`.harvested`: the captured layer when ``EVOGRAD_QWEN3_5_LAYER_DIR`` names
a directory holding ``layer14.pt`` / ``layer15.pt``, else a config-derived
block). This step records the inventory; fitting tolerances is a later step's.
"""

from __future__ import annotations

import sys

from evograd.evaluation.workloads.common import level2_calibrate as _common
from evograd.evaluation.workloads.qwen3_5_4b.descriptor import WORKLOAD

#: Every task this model calibrates (no new primitives).
TASKS = (*WORKLOAD.level2_tasks, *WORKLOAD.level1_tasks)


def inventory_task(task, **kwargs):
    return _common.inventory_task(task, descriptor=WORKLOAD, **kwargs)


def scaling_study(task, **kwargs):
    return _common.scaling_study(task, descriptor=WORKLOAD, **kwargs)


def main(argv=None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    sys.exit(main())
