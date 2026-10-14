"""The canonical workload: one Qwen3-0.6B training step, exactly specified.

Batch 2, sequence 2048, BF16, CUDA, SDPA, ``model.train()``,
``use_cache=False``, no gradient checkpointing, a fixed seed, and no optimizer
step. The spec hashes to a workload id every artifact downstream is stamped
with, so a result can always be traced to the execution it came from.
"""

from .report import SmokeReport
from .spec import CANONICAL, QWEN3_0_6B, WorkloadSpec, WorkloadSpecError

__all__ = ["CANONICAL", "QWEN3_0_6B", "SmokeReport", "WorkloadSpec",
           "WorkloadSpecError"]
