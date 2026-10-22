"""Llama-3.2-1B: real training payloads and the three gates on real data.

* :mod:`.capture` -- what a real training step records, call by call;
* :mod:`.reference` -- the model's own computation, the reference every gate uses.

The capture loop, replay and gates are shared (:mod:`..base`, :mod:`..gates`).
The pinned checkpoint is gated: it is read from a local cache named by
``$EVOGRAD_LLAMA_HF_CACHE`` and never downloaded.

    python -m evograd.verification.integrations llama_3_2_1b {capture,fidelity,verify} ...
"""

from __future__ import annotations

from ..base import Model
from . import capture, reference

MODEL = Model(
    name="llama_3_2_1b",
    suite="llama_3_2_1b_observed",
    workload_config=capture.WORKLOAD_CONFIG,
    load_workload=capture.load_workload,
    train_batches=capture.train_batches,
    live_capture=capture.LiveCapture,
    invocations=capture.invocations,
    reference_op=reference.reference_op,
    contract=reference.CONTRACT,
    source_identity=reference.source_identity,
    modeling_modules=("transformers.models.llama.modeling_llama",),
    # At the protocol's constant 1e-4 the pretrained model's training is
    # chaotic: any perturbation (a compiled site included) ends 2-5 nats away
    # whatever its size, so Gate 3 cannot resolve a fault. Warmup, clipping and
    # 3e-5 keep it stable; compile then drifts R 0.015 and a 5% gradient fault
    # at one site is rejected (gate3 optimizer probe, 2026-10-22).
    gate3_optimizer={"warmup_steps": 100, "clip_grad_norm": 1.0, "lr": 3e-5},
)
