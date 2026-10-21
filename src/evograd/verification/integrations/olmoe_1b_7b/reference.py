"""What every OLMoE-1B-7B gate judges against: the model's own computation.

Each case's task is re-declared under :data:`CONTRACT` with its runtime
spelling as the single reference; the Level-2 production spellings already
are the model's arithmetic (``levels/level2``: ``olmoe_rms_norm``, the
experts' own ``grouped_mm_experts_forward``). The one exception is the
Level-1 RMSNorm, whose ``F.rms_norm`` spelling rounds the weight product once
where ``OlmoeRMSNorm`` rounds twice; it is spelled as the module computes it.
"""

from __future__ import annotations

import dataclasses
import hashlib

CONTRACT = "evograd-model-reference/olmoe_1b_7b/1"


def rms_norm(x, weight, eps: float = 1e-5):
    """``OlmoeRMSNorm.forward``."""
    from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.norm import olmoe_rms_norm

    return olmoe_rms_norm(x, weight, eps)


_FORWARDS = {"rmsnorm": f"{__name__}:rms_norm"}


def source_identity() -> dict:
    import transformers
    from transformers.models.olmoe import modeling_olmoe

    with open(modeling_olmoe.__file__, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"transformers": transformers.__version__, "modeling_olmoe_sha256": digest}


def reference_op(task: str):
    from evograd.benchmark import get_task

    op = get_task(task)
    model = dataclasses.replace(op, forward=_FORWARDS.get(task) or op.runtime_forward or op.forward,
                                runtime_forward=None, reference_dtype=None,
                                reference_contract=CONTRACT)
    model.validate()
    return model
