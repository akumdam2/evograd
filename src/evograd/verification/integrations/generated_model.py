"""The verification :class:`~.base.Model` of a generated workload (DiT-S/2, DLRM).

Both models are evograd's own ``generated.model``: every case's task already
computes exactly what the model calls at that point (``F.linear``,
``F.conv2d``, ``F.embedding_bag``, the site functions), so the model reference
is each task's own forward, re-declared under the model's contract with no
float32 promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from .base import GATE3_STEPS, Model
from .generated import capture_for, invocations
from .objectives import PrimaryOutput, batch_digest

MODELING = "evograd.benchmark.topdown.common.generated.model"


def contract(name: str) -> str:
    return f"evograd-model-reference/{name}/1"


def source_identity() -> dict:
    """The generated-model source every reference follows."""
    import importlib

    with open(importlib.import_module(MODELING).__file__, "rb") as handle:
        return {"generated_model_sha256": hashlib.sha256(handle.read()).hexdigest()}


def reference_op_for(name: str):
    def reference_op(task: str):
        from evograd.benchmark import get_task

        op = get_task(task)
        model = dataclasses.replace(op, forward=op.runtime_forward or op.forward, runtime_forward=None,
                                    reference_dtype=None, reference_contract=contract(name))
        model.validate()
        return model

    return reference_op


def objective(name: str) -> PrimaryOutput:
    """The workload's own primary output and loss definition."""
    from types import SimpleNamespace

    from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload

    probe = SimpleNamespace(spec=SimpleNamespace(model=name))
    return PrimaryOutput(GeneratedWorkload.primary_output_name.fget(probe),
                         GeneratedWorkload.objective_definition.fget(probe))


def make_model(name: str, workload_config: dict[str, Any], *,
               gate3_steps: int = GATE3_STEPS) -> Model:
    def load_workload():
        from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload

        return GeneratedWorkload.from_config(dict(workload_config))

    def train_batches(workload, orders):
        out = {order: workload.train_batches(order).batch(0) for order in orders}
        identity = {"data": workload.data_identity(), "rule": "workload.train_batches(order).batch(0)",
                    "batches": {str(order): {"batch_sha256": batch_digest(batch),
                                             "order": workload.train_batches(order).describe()}
                                for order, batch in out.items()}}
        return out, identity

    return Model(name=name, suite=f"{name}_observed", workload_config=workload_config,
                 load_workload=load_workload, train_batches=train_batches,
                 live_capture=capture_for(name), invocations=invocations,
                 reference_op=reference_op_for(name), contract=contract(name),
                 source_identity=source_identity, modeling_modules=(MODELING,),
                 objective=objective(name), gate3_steps=gate3_steps)
