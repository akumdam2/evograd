"""Layer replay for OLMoE-1B-7B: the shared judgment, bound to its descriptor.

    python -m evograd.evaluation.workloads.olmoe_1b_7b.layer.replay \
        --artifact results/experiments/olmoe_1b_7b_cases/<run>/layer8.pt

Rebuilds the captured layer standalone (no full model on the heap), dispatching
the experts as the captured step did, replays its call and compares output,
input gradient and every parameter gradient against what the full model
produced; a correct replay is bitwise.
"""

from __future__ import annotations

from evograd.evaluation.workloads.common import replay as _common
from evograd.evaluation.workloads.common.replay import (  # noqa: F401  (re-export)
    FORWARD_TOL,
    GRADIENT_TOL,
    REPORT_SCHEMA_SUFFIX,
    compare_tensors,
    summarize,
)
from evograd.evaluation.workloads.olmoe_1b_7b.descriptor import WORKLOAD

#: What a report written by this module carries.
REPORT_SCHEMA = WORKLOAD.schema(REPORT_SCHEMA_SUFFIX)


def run_replay(artifact, **kwargs):
    return _common.run_replay(artifact, workload=WORKLOAD, **kwargs)


def live_model_instances():
    return WORKLOAD.module("layer.prepare").live_model_instances()


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, workload=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
