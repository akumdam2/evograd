"""The verification :class:`~.base.Model` of a vision workload (ConvNeXt V2, Qwen2-VL).

Pinned pretrained weights and the streamed real-data pool
(``VisionWorkload(stream=True)``). Each case's task is re-declared under the
model's contract with its runtime spelling as the single reference; the one
exception is Qwen2-VL's RMSNorm, whose HF module rounds the weight product
twice (the ``Qwen3RMSNorm`` spelling), where ``F.rms_norm`` rounds once.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from .base import GATE3_STEPS, Model
from .objectives import PrimaryOutput, batch_digest
from .vision import capture_for, invocations

MODELING = {"convnextv2_tiny": "transformers.models.convnextv2.modeling_convnextv2",
            "qwen2_vl_2b": "transformers.models.qwen2_vl.modeling_qwen2_vl"}


def contract(name: str) -> str:
    return f"evograd-model-reference/{name}/1"


def source_identity_for(name: str):
    def source_identity() -> dict:
        import importlib

        import transformers

        with open(importlib.import_module(MODELING[name]).__file__, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        return {"transformers": transformers.__version__, "modeling_sha256": digest}

    return source_identity


def rms_norm(x, weight, eps: float):
    """The HF Qwen2 RMSNorm: float32 normalization, cast back, then the weight product."""
    from .qwen3_0_6b.reference import rms_norm as hf_rms_norm

    return hf_rms_norm(x, weight, eps)


def reference_op_for(name: str):
    forwards = {"rmsnorm": f"{__name__}:rms_norm"} if name == "qwen2_vl_2b" else {}

    def reference_op(task: str):
        from evograd.benchmark import get_task

        op = get_task(task)
        model = dataclasses.replace(op, forward=forwards.get(task) or op.runtime_forward or op.forward,
                                    runtime_forward=None, reference_dtype=None,
                                    reference_contract=contract(name))
        model.validate()
        return model

    return reference_op


def make_model(name: str, workload_config: dict[str, Any], *,
               gate3_steps: int = GATE3_STEPS) -> Model:
    def load_workload():
        from evograd.benchmark.topdown.common.vision.spec import canonical
        from evograd.evaluation.tier3.workloads.common.vision.workload import VisionWorkload

        return VisionWorkload.from_config({**canonical(name).to_config(), **workload_config})

    def train_batches(workload, orders):
        out = {order: workload.train_batches(order).batch(0) for order in orders}
        identity = {"data": workload.data_identity(), "rule": "workload.train_batches(order).batch(0)",
                    "batches": {str(order): {"batch_sha256": batch_digest(batch),
                                             "order": workload.train_batches(order).describe()}
                                for order, batch in out.items()}}
        return out, identity

    def objective():
        from types import SimpleNamespace

        from evograd.evaluation.tier3.workloads.common.vision.workload import VisionWorkload

        probe = SimpleNamespace(spec=SimpleNamespace(model=name))
        return PrimaryOutput(VisionWorkload.primary_output_name,
                             VisionWorkload.objective_definition.fget(probe))

    return Model(name=name, suite=f"{name}_observed", workload_config=workload_config,
                 load_workload=load_workload, train_batches=train_batches,
                 live_capture=capture_for(name), invocations=invocations,
                 reference_op=reference_op_for(name), contract=contract(name),
                 source_identity=source_identity_for(name), modeling_modules=(MODELING[name],),
                 objective=objective(), gate3_steps=gate3_steps)
