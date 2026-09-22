"""Level-3 replay for this model: the shared judgment, bound to its descriptor.

The logic is in :mod:`evograd.evaluation.workloads.common.replay`. This file
supplies the model and the entry point, so
``python -m evograd.evaluation.workloads.llama3_2_1b.level3.replay`` keeps working.
"""

from __future__ import annotations

from evograd.evaluation.workloads.common import replay as _common
from evograd.evaluation.workloads.common.replay import (  # noqa: F401  (re-export)
    BF16_EPS,
    BF16_UNIT_ROUNDOFF,
    ELEMENTWISE_FLOOR_FRACTION,
    FORWARD_TOL,
    GRADIENT_TOL,
    REPORT_SCHEMA_SUFFIX,
    _max_noise,
    _noise,
    compare_tensors,
    declared_gate,
    required_tolerance,
    summarize,
)
from evograd.evaluation.workloads.llama3_2_1b.descriptor import WORKLOAD

#: What a report written by this module carries.
REPORT_SCHEMA = WORKLOAD.schema(REPORT_SCHEMA_SUFFIX)


def run_replay(artifact, **kwargs):
    return _common.run_replay(artifact, workload=WORKLOAD, **kwargs)


def live_model_instances():
    """Count live full-model objects: the standalone claim's evidence.

    The scan itself is the benchmark package's (``levels.level3.prepare``),
    because which classes count as "the full model" is a fact about the
    model. Exposed here so a consumer of the replay can ask the same question
    the replay asks of itself, without importing the model package by name.
    """
    return WORKLOAD.module("levels.level3.prepare").live_model_instances()


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, workload=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
