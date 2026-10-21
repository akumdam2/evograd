"""What every Qwen3-0.6B gate judges against: the model's own computation.

The benchmark tasks keep their declarations (ABI, cases, timed eager
baseline). For verification each case's task is re-declared under one
model-reference contract, :data:`CONTRACT`: its ``forward`` is the computation
the unmodified ``Qwen3ForCausalLM`` performs at that call, used alike by the
Gate 1 oracle, the Gate 2 native and compile controls and boundary replay,
with no float32 promotion and no second runtime spelling.

Eight tasks' runtime spellings already are that computation; the capture's
replay checks it bit for bit on every recorded call. RMSNorm and the residual
add + norm are not: their runtime spelling is ``F.rms_norm``, which rounds the
weight product once where ``Qwen3RMSNorm`` rounds twice. They are spelled
here as the module computes them.
"""

from __future__ import annotations

import dataclasses
import hashlib

import torch

CONTRACT = "evograd-model-reference/qwen3_0_6b/2"

#: ``rms_norm_eps`` of the pinned checkpoint (the generic recipe uses 1e-5).
RMS_NORM_EPS = 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """``Qwen3RMSNorm.forward``: float32 normalization, cast back, then the weight product."""
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def residual_rms_norm(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                      eps: float = RMS_NORM_EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual + hidden_states`` of ``Qwen3DecoderLayer.forward``, then :func:`rms_norm`."""
    summed = r + x
    return rms_norm(summed, weight, eps), summed


#: Tasks whose runtime spelling is not the model's computation.
_FORWARDS = {"rmsnorm": f"{__name__}:rms_norm", "fused_add_rms_norm": f"{__name__}:residual_rms_norm"}


def source_identity() -> dict:
    """The Transformers source these spellings follow (recorded in reports and policies)."""
    import transformers
    from transformers.models.qwen3 import modeling_qwen3

    with open(modeling_qwen3.__file__, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {"transformers": transformers.__version__, "modeling_qwen3_sha256": digest}


def _model_eps(make_inputs):
    def make(torch_, op, workload, device="cuda"):
        values = make_inputs(torch_, op, workload, device)
        values["eps"] = RMS_NORM_EPS
        return values

    return make


def reference_op(task: str):
    """``task`` re-declared under :data:`CONTRACT`, with the model's computation as forward."""
    from evograd.benchmark import get_task

    op = get_task(task)
    changes = {"forward": _FORWARDS.get(task) or op.runtime_forward or op.forward,
               "runtime_forward": None, "reference_dtype": None, "reference_contract": CONTRACT}
    if task == "rmsnorm":
        changes["make_inputs"] = _model_eps(op.make_inputs)
    model = dataclasses.replace(op, **changes)
    model.validate()
    return model
