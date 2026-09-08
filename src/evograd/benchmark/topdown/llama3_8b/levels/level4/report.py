"""Llama-3's smoke report: the shared schema, under this workload's version string."""

from __future__ import annotations

from ....common.report import (  # noqa: F401  (re-export)
    STATUS_FAILED,
    STATUS_OK,
    SmokeReport,
)

#: Kept here as well as on the workload declaration because a reader of a report
#: file wants to know which schema produced it without loading the package.
SCHEMA_VERSION = "evograd-llama3-smoke/1"
