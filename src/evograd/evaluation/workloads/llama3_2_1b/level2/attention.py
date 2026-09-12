"""Level-2 attention judgment for Llama-3-8B: shared logic, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level2_attention as _common
from evograd.evaluation.workloads.llama3_2_1b.descriptor import WORKLOAD


def run_verify(payload, **kwargs):
    return _common.run_verify(payload, descriptor=WORKLOAD, **kwargs)


def run_calibration(source, **kwargs):
    return _common.run_calibration(source, descriptor=WORKLOAD, **kwargs)


def summarize_verify(report):
    return _common.summarize_verify(report)


def summarize_calibration(report):
    return _common.summarize_calibration(report, descriptor=WORKLOAD)


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
