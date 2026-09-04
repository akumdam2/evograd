"""Qwen3-0.6B's Level-4 declaration: the facts the shared machinery asks for.

Everything that *does* something -- building, observing, harvesting, reporting
-- is in :mod:`...common`. This is the short list of things only Qwen3 can
answer, gathered in one object so no two stages can disagree about which model
they are describing.
"""

from __future__ import annotations

from ..common.level4 import Level4Workload
from .harvest.manifest import FUNCTION_WRAPPERS, SCHEMA_VERSION as MANIFEST_SCHEMA
from .harvest.observe import PLAN
from .levels.level4.model import CLASSES
from .levels.level4.spec import WorkloadSpec

#: Reports carry this so a Qwen3 smoke cannot be mistaken for another model's.
SMOKE_SCHEMA = "evograd-qwen3-smoke/1"

WORKLOAD = Level4Workload(
    name="qwen3_0_6b",
    label="Qwen3-0.6B",
    spec_type=WorkloadSpec,
    classes=CLASSES,
    plan=PLAN,
    smoke_schema=SMOKE_SCHEMA,
    manifest_schema=MANIFEST_SCHEMA,
    function_wrappers=FUNCTION_WRAPPERS,
    package="evograd.bench.workloads.qwen3",
)
