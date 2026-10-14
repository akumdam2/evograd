"""Qwen3.5-4B's Level-3 declaration: the facts the shared machinery asks for.

Everything that *does* something -- building, observing, harvesting, reporting
-- is in :mod:`..common`. This is the short list only Qwen3.5 can answer: its
spec, its classes, its observation plan (two block kinds, held layer by layer),
its schemas, and the kernel dispatch it reads back and pins to the torch path.
"""

from __future__ import annotations

from ..common.level3 import Level3Workload
from .harvest.manifest import FUNCTION_WRAPPERS, NOT_OBSERVED, SCHEMA_VERSION as MANIFEST_SCHEMA
from .harvest.observe import PLAN, check_layers
from .levels.level3.model import CLASSES, PINNED_DISPATCH, dispatch_settings
from .levels.level3.report import SCHEMA_VERSION as SMOKE_SCHEMA
from .levels.level3.spec import WorkloadSpec

WORKLOAD = Level3Workload(
    name="qwen3_5_4b",
    label="Qwen3.5-4B",
    spec_type=WorkloadSpec,
    classes=CLASSES,
    plan=PLAN,
    smoke_schema=SMOKE_SCHEMA,
    manifest_schema=MANIFEST_SCHEMA,
    function_wrappers=FUNCTION_WRAPPERS,
    package="evograd.benchmark.topdown.qwen3_5_4b",
    extra_settings=dispatch_settings,
    pinned_settings=PINNED_DISPATCH,
    observation_checks=check_layers,
    not_observed=NOT_OBSERVED,
)
