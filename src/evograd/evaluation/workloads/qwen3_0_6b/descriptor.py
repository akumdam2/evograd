"""Qwen3-0.6B, as the evaluation modules refer to it."""

from __future__ import annotations

from pathlib import Path

from evograd.evaluation.workloads.common import EvalWorkload

WORKLOAD = EvalWorkload(
    name="qwen3_0_6b",
    topdown_package="evograd.benchmark.topdown.qwen3_0_6b",
    results_dir=Path("results/qwen3-level4"),
    schema_prefix="evograd-qwen3",
    # Mid-stack of 28.
    representative_layer=14,
    canonical_loss=12.14388656616211,
    qkv_module="qkv_norm_rope",
    full_model_classes=("Qwen3ForCausalLM", "Qwen3Model"),
    # This model's own Level-2 declarations; `residual_rmsnorm` is excluded
    # from the inventory sweep, as it always was.
    level2_tasks=("qwen3_qkv_norm_rope", "qwen3_attention", "qwen3_swiglu_mlp"),
)
