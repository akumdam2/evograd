"""Qwen3.5-4B, as the evaluation modules refer to it."""

from __future__ import annotations

from pathlib import Path

from evograd.benchmark.topdown.qwen3_5_4b.architecture import REPRESENTATIVE_LAYERS
from evograd.benchmark.topdown.qwen3_5_4b.levels.level2.manifest import TASKS
from evograd.evaluation.workloads.common import EvalWorkload

WORKLOAD = EvalWorkload(
    name="qwen3_5_4b",
    topdown_package="evograd.benchmark.topdown.qwen3_5_4b",
    results_dir=Path("results/experiments/qwen3_5_4b_cases"),
    schema_prefix="evograd-qwen3_5",
    level2_tasks=TASKS,
    # One representative layer per block kind (14, Gated DeltaNet; 15, gated
    # full attention), each captured from the whole-model step.
    representative_layers=dict(REPRESENTATIVE_LAYERS),
    qkv_module="qkv_norm_partial_rope",
    full_model_classes=("Qwen3_5ForCausalLM", "Qwen3_5TextModel"),
    display_label="Qwen3.5-4B",
    # The calibration's third population: each site's real invocation, from
    # the captured layer when present, else from a config-derived block.
    harvested_invocation=(
        "evograd.evaluation.workloads.qwen3_5_4b.level2.harvested:captured_invocation"),
    # The Level-1 primitives a hybrid linear-attention decoder introduces,
    # calibrated on this model's own grid and observed cases.
    level1_tasks=("causal_conv1d", "gated_rmsnorm", "gdn_gating", "gemma_rmsnorm",
                  "l2norm", "partial_rope"),
)
