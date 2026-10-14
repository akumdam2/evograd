"""Run the canonical Qwen3.5 training step once and describe what happened.

The run, the verification and the report shape are shared -- see
:mod:`....common.smoke`. This binds them to Qwen3.5's declaration, which adds
the import-time kernel dispatch to the settings it reads back and pins it.
"""

from __future__ import annotations

from ....common.smoke import (  # noqa: F401  (re-export)
    environment_info,
    gradient_coverage,
    workload_info,
)
from ....common import smoke as _common
from ....common.report import SmokeReport
from .spec import WorkloadSpec


def run_smoke(spec: WorkloadSpec | None = None) -> SmokeReport:
    """Execute one canonical Qwen3.5 training step and return the report."""
    from ...declaration import WORKLOAD

    return _common.run_smoke(WORKLOAD, spec)
