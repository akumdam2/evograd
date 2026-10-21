"""Qwen3-0.6B: real training payloads and the three gates on real data.

* :mod:`.capture` -- what a real training step records, call by call;
* :mod:`.reference` -- the model's own computation, the reference every gate uses.

The capture loop, replay and gates are shared (:mod:`..base`, :mod:`..gates`).

    python -m evograd.verification.integrations.qwen3_0_6b {capture,fidelity,verify} ...
"""

from __future__ import annotations

from ..base import Model
from . import capture, reference

MODEL = Model(
    name="qwen3_0_6b",
    suite="qwen3_0_6b_observed",
    workload_config=capture.WORKLOAD_CONFIG,
    load_workload=capture.load_workload,
    train_batches=capture.train_batches,
    live_capture=capture.LiveCapture,
    invocations=capture.invocations,
    reference_op=reference.reference_op,
    contract=reference.CONTRACT,
    source_identity=reference.source_identity,
    modeling_modules=("transformers.models.qwen3.modeling_qwen3",),
)
