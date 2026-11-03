"""Qwen3.5-4B: real training payloads and the three gates on real data.

* :mod:`.capture` -- what a real training step records, call by call;
* :mod:`.reference` -- the model's own computation, the reference every gate uses.

The capture loop, replay and gates are shared (:mod:`..base`, :mod:`..gates`).

    python -m evograd.verification.integrations.qwen3_5_4b {capture,fidelity,verify} ...
"""

from __future__ import annotations

from ..base import Model
from . import capture, reference

MODEL = Model(
    name="qwen3_5_4b",
    suite="qwen3_5_4b_observed",
    workload_config=capture.WORKLOAD_CONFIG,
    load_workload=capture.load_workload,
    train_batches=capture.train_batches,
    live_capture=capture.LiveCapture,
    invocations=capture.invocations,
    reference_op=reference.reference_op,
    contract=reference.CONTRACT,
    source_identity=reference.source_identity,
    modeling_modules=("transformers.models.qwen3_5.modeling_qwen3_5",),
    # Gate 3 at the protocol's 100 updates (2.7 s each), with Llama's warmup, clipping
    # and 3e-5: at the protocol's constant 1e-4 its compile control drifted M 1.24 by
    # update 100 (2026-10-20). Not yet measured with its seeds and faults.
    gate3_optimizer={"warmup_steps": 100, "clip_grad_norm": 1.0, "lr": 3e-5},
    # Model-scope Gate 2 at 7, not the protocol's 4: under calibration /4 the
    # residual_rmsnorm seed needs 6.31 (Emax tails of GDN and norm parameter grads on
    # the order-404 batch; 2026-11-02), and 7 is the next integer. 1% faults pass it
    # from 1.32 (gdn_in_proj), but operator Gate 2 stops all 17 (they need >= 2.54).
    gate2_model_gamma=7.0,
)
