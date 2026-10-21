"""OLMoE-1B-7B: real training payloads; Gate 1 and operator Gate 2.

* :mod:`.capture` -- what a real training step records, call by call;
* :mod:`.reference` -- the model's own computation, the reference every gate uses.

Model-scope Gate 2 and Gate 3 are not wired: routed experts flip their top-k
choice under tiny numerical differences, so a whole-model comparison needs a
fixed (teacher-forced) routing condition first. Level-2 candidates are judged
by Gate 1 and operator Gate 2, where the routing is a recorded input.

    python -m evograd.verification.integrations olmoe_1b_7b {capture,fidelity,verify} ...
"""

from __future__ import annotations

from ..base import Model
from . import capture, reference

MODEL = Model(
    name="olmoe_1b_7b",
    suite="olmoe_1b_7b_observed",
    workload_config=capture.WORKLOAD_CONFIG,
    load_workload=capture.load_workload,
    train_batches=capture.train_batches,
    live_capture=capture.LiveCapture,
    invocations=capture.invocations,
    reference_op=reference.reference_op,
    contract=reference.CONTRACT,
    source_identity=reference.source_identity,
    modeling_modules=("transformers.models.olmoe.modeling_olmoe",),
    model_gates=False,
)
