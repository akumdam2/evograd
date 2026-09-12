"""Level-1 verification for Llama-3-8B: the shared judgment, bound to this model."""

from __future__ import annotations

from evograd.evaluation.workloads.common import level1_verify as _common
from evograd.evaluation.workloads.common.level1_verify import (  # noqa: F401
    Level1Error,
    preserve_layout_cpu,
)
from evograd.evaluation.workloads.llama3_2_1b.descriptor import WORKLOAD


def summarize_mapping(report):
    return _common.summarize_mapping(report, descriptor=WORKLOAD)


def derive_sdpa_invocation(source, **kwargs):
    return _common.derive_sdpa_invocation(source, descriptor=WORKLOAD, **kwargs)


def run_verify(source, **kwargs):
    return _common.run_verify(source, descriptor=WORKLOAD, **kwargs)


def run_loss_check(**kwargs):
    return _common.run_loss_check(descriptor=WORKLOAD, **kwargs)


def run_cross_entropy_check(**kwargs):
    return _common.run_cross_entropy_check(descriptor=WORKLOAD, **kwargs)
