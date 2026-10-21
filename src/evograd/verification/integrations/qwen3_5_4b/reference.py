"""What every Qwen3.5-4B gate judges against: the model's own computation.

Each case's task is re-declared under :data:`CONTRACT` with its runtime
spelling (or, without one, its forward) as the single reference: the HF
computation the model runs at that call, with no float32 promotion. Unlike
Qwen3-0.6B no task needs a re-spelling; the capture's replay checks every
recorded call bit for bit.
"""

from __future__ import annotations

import dataclasses
import hashlib

CONTRACT = "evograd-model-reference/qwen3_5_4b/1"


def source_identity() -> dict:
    """The Transformers source the references follow."""
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    with open(modeling_qwen3_5.__file__, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"transformers": transformers.__version__, "modeling_qwen3_5_sha256": digest}


def reference_op(task: str):
    """``task`` re-declared under :data:`CONTRACT`."""
    from evograd.benchmark import get_task

    op = get_task(task)
    model = dataclasses.replace(op, forward=op.runtime_forward or op.forward, runtime_forward=None,
                                reference_dtype=None, reference_contract=CONTRACT)
    model.validate()
    return model
