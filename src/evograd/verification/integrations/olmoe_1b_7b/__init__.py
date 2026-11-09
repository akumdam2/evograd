"""OLMoE-1B-7B: real training payloads and every gate.

* :mod:`.capture` -- what a real training step records, call by call;
* :mod:`.reference` -- the model's own computation, the reference every gate uses;
* :mod:`.routing` -- how model Gate 2 holds the routing fixed.

Routed experts flip their top-k choice under tiny numerical differences. Model
Gate 2 compares every output and gradient, so it runs on the native run's routing.
Gate 3 keeps free routing: it judges the loss curve against native and compile
runs, and compile flips routing too, so its thresholds already hold that spread.
Its objective is the next-token loss, without the router's 0.01 load-balancing term.

Gate 3's optimizer is Qwen3.5's, and model Gate 2's margin is the protocol's. Neither
has been measured on this model.

    python -m evograd.verification.integrations olmoe_1b_7b {capture,fidelity,verify} ...
"""

from __future__ import annotations

from ..base import Model
from . import capture, reference
from .routing import ForcedRoutingLM

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
    objective=ForcedRoutingLM(),
    gate3_optimizer={"warmup_steps": 100, "clip_grad_norm": 1.0, "lr": 3e-5},
)
