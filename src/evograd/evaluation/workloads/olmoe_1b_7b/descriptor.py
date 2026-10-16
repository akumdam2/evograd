"""OLMoE-1B-7B, as the evaluation modules refer to it."""

from __future__ import annotations

from pathlib import Path

from evograd.benchmark.topdown.olmoe_1b_7b.architecture import REPRESENTATIVE_LAYER
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.manifest import TASKS
from evograd.evaluation.workloads.common import EvalWorkload

WORKLOAD = EvalWorkload(
    name="olmoe_1b_7b",
    topdown_package="evograd.benchmark.topdown.olmoe_1b_7b",
    results_dir=Path("results/experiments/olmoe_1b_7b_cases"),
    schema_prefix="evograd-olmoe",
    level2_tasks=TASKS,
    # One representative layer (8 of 16): every OLMoE layer is the same block.
    representative_layer=REPRESENTATIVE_LAYER,
    # The q/k/v boundary that is OLMoE's own: full-width q/k norms, then RoPE.
    qkv_module="qkv_norm_rope",
    full_model_classes=("OlmoeForCausalLM", "OlmoeModel"),
    display_label="OLMoE-1B-7B",
    # The calibration's third population: each site's real invocation, from
    # the captured layer when present, else from a config-derived layer.
    harvested_invocation=(
        "evograd.evaluation.workloads.olmoe_1b_7b.level2.harvested:captured_invocation"),
    # No new Level-1 primitives: every observed primitive keeps its own calibration.
    level1_tasks=(),
)
