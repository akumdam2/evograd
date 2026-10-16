"""OLMoE's smoke report: the shared schema, under this workload's version string."""

from __future__ import annotations

from ....common.report import (  # noqa: F401  (re-export)
    STATUS_FAILED,
    STATUS_OK,
    SmokeReport,
)

#: Kept here as well as on the workload declaration so a reader of a report
#: file knows which schema produced it without loading the package.
SCHEMA_VERSION = "evograd-olmoe-smoke/1"
