"""OLMoE-1B-7B's Level-3 declaration: the facts the shared machinery asks for.

Everything that *does* something -- building, observing, harvesting, reporting
-- is in :mod:`..common`. This is the short list only OLMoE can answer: its
spec, its classes, its observation plan (a routed MoE in the MLP position, held
layer by layer), its schemas, and the experts kernel it reads back.
"""

from __future__ import annotations

from ..common.level3 import Level3Workload
from .harvest.manifest import FUNCTION_WRAPPERS, NOT_OBSERVED, SCHEMA_VERSION as MANIFEST_SCHEMA
from .harvest.observe import PLAN, check_layers
from .levels.level3.model import CLASSES, experts_settings
from .levels.level3.report import SCHEMA_VERSION as SMOKE_SCHEMA
from .levels.level3.spec import WorkloadSpec

WORKLOAD = Level3Workload(
    name="olmoe_1b_7b",
    label="OLMoE-1B-7B",
    spec_type=WorkloadSpec,
    classes=CLASSES,
    plan=PLAN,
    smoke_schema=SMOKE_SCHEMA,
    manifest_schema=MANIFEST_SCHEMA,
    function_wrappers=FUNCTION_WRAPPERS,
    package="evograd.benchmark.topdown.olmoe_1b_7b",
    # The experts kernel is pinned by the *spec* (``experts_implementation`` is a
    # hashed field) and checked by the shared read-back, as the attention
    # backend is; the extra settings add the per-module view.
    extra_settings=experts_settings,
    observation_checks=check_layers,
    not_observed=NOT_OBSERVED,
)
