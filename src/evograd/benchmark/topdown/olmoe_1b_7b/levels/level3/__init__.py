"""The canonical workload: one OLMoE-1B-7B training step, exactly specified.

``OlmoeForCausalLM``, batch 2, sequence 2048, BF16, CUDA, SDPA, grouped-GEMM
experts, ``model.train()``, ``use_cache=False``, no gradient checkpointing,
random weights from a fixed seed, synthetic tokens.
"""

from .report import SmokeReport
from .spec import CANONICAL, OLMOE_1B_7B, WorkloadSpec, WorkloadSpecError

__all__ = ["CANONICAL", "OLMOE_1B_7B", "SmokeReport", "WorkloadSpec", "WorkloadSpecError"]
