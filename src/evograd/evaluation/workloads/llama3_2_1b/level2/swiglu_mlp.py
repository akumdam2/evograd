"""Level-2 swiglu_mlp judgment for Llama-3-8B: shared logic, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level2_swiglu_mlp as _common
from evograd.evaluation.workloads.llama3_2_1b.descriptor import WORKLOAD


def run_verify(*args, **kwargs):
    return _common.run_verify(*args, descriptor=WORKLOAD, **kwargs)


def run_calibration(*args, **kwargs):
    return _common.run_calibration(*args, descriptor=WORKLOAD, **kwargs)


def declaration_problems(*args, **kwargs):
    return _common.declaration_problems(*args, descriptor=WORKLOAD, **kwargs)


def build_parser():
    return _common.build_parser(WORKLOAD)


def main(argv: list[str] | None = None) -> int:
    return _common.main(argv, descriptor=WORKLOAD)


if __name__ == "__main__":
    raise SystemExit(main())
