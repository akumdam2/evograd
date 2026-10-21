"""What every Llama-3.2-1B gate judges against: the model's own computation.

Each case's task is re-declared under :data:`CONTRACT`: its ``forward`` is the
computation the unmodified ``LlamaForCausalLM`` performs at that call, used by
the Gate 1 oracle, the Gate 2 controls and boundary replay alike, with no
float32 promotion and no second runtime spelling.

The runtime spellings already are that computation, except RMSNorm and the
residual add + norm: ``F.rms_norm`` rounds the weight product once where
``LlamaRMSNorm`` rounds twice (the same spelling as ``Qwen3RMSNorm``). Both
generated-input recipes already use the model's ``rms_norm_eps`` (1e-5).
"""

from __future__ import annotations

import dataclasses
import hashlib

from evograd.verification.integrations.qwen3_0_6b import reference as _qwen3

CONTRACT = "evograd-model-reference/llama_3_2_1b/1"

#: ``rms_norm_eps`` of the published configuration.
RMS_NORM_EPS = 1e-5


def rms_norm(x, weight, eps: float = RMS_NORM_EPS):
    """``LlamaRMSNorm.forward``: float32 normalization, cast back, then the weight product."""
    return _qwen3.rms_norm(x, weight, eps)


def residual_rms_norm(x, r, weight, eps: float = RMS_NORM_EPS):
    """``residual + hidden_states`` of ``LlamaDecoderLayer.forward``, then :func:`rms_norm`."""
    return _qwen3.residual_rms_norm(x, r, weight, eps)


#: Tasks whose runtime spelling is not the model's computation.
_FORWARDS = {"rmsnorm": f"{__name__}:rms_norm",
             "llama3_residual_rmsnorm": f"{__name__}:residual_rms_norm"}


def source_identity() -> dict:
    """The Transformers source these spellings follow (recorded in reports and policies)."""
    import transformers
    from transformers.models.llama import modeling_llama

    with open(modeling_llama.__file__, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"transformers": transformers.__version__, "modeling_llama_sha256": digest}


def reference_op(task: str):
    """``task`` re-declared under :data:`CONTRACT`, with the model's computation as forward."""
    from evograd.benchmark import get_task

    op = get_task(task)
    model = dataclasses.replace(op, forward=_FORWARDS.get(task) or op.runtime_forward or op.forward,
                                runtime_forward=None, reference_dtype=None,
                                reference_contract=CONTRACT)
    model.validate()
    return model
