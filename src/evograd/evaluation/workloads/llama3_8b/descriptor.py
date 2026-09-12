"""Llama-3-8B, as the evaluation modules refer to it."""

from __future__ import annotations

from pathlib import Path

from evograd.evaluation.workloads.common import EvalWorkload

WORKLOAD = EvalWorkload(
    name="llama_3_8b",
    topdown_package="evograd.benchmark.topdown.llama3_8b",
    results_dir=Path("results/llama3-level4"),
    schema_prefix="evograd-llama3",
    # Mid-stack of 32, as Qwen3 uses 14 of 28.
    representative_layer=8,
    # `llama3_qkv_rope` is Llama's own: LlamaAttention has no per-head
    # query/key RMSNorm, so Qwen3's `qwen3_qkv_norm_rope` describes a
    # different computation. The other two are dimension-parameterized and
    # shared.
    canonical_loss=12.594182968139648,
    qkv_module="qkv_rope",
    full_model_classes=("LlamaForCausalLM", "LlamaModel"),
    # This model's own Level-2 declarations. Each model owns all four since
    # the benchmark split them per model; using another model's task here
    # calibrates the wrong declaration against these tensors -- which is how
    # `llama3_swiglu_mlp`'s `out` multiplier went missing. `residual_rmsnorm`
    # is excluded from the inventory sweep, as it always was: it has its own
    # module. The first entry is the q/k/v boundary.
    level2_tasks=("llama3_qkv_rope", "llama3_attention", "llama3_swiglu_mlp"),
)
