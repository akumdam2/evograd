"""The canonical workload: one Qwen3.5-4B training step, exactly specified.

Text model only (``Qwen3_5ForCausalLM``), batch 1, sequence 2048, BF16, CUDA,
SDPA, ``model.train()``, ``use_cache=False``, no gradient checkpointing, random
weights from a fixed seed, synthetic tokens.
"""

from .report import SmokeReport
from .spec import CANONICAL, QWEN3_5_4B, WorkloadSpec, WorkloadSpecError

__all__ = ["CANONICAL", "QWEN3_5_4B", "SmokeReport", "WorkloadSpec", "WorkloadSpecError"]
