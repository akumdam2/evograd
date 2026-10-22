"""Qwen2-VL-2B-Instruct: the three gates on real TextVQA rows.

Pinned pretrained weights and a frozen pool of TextVQA shard-0 rows (one image
at 224x224, the question, the answer as the only labelled tokens); capture and
model reference are shared with ConvNeXt V2 (:mod:`..vision`, :mod:`..vision_model`).

    python -m evograd.verification.integrations qwen2_vl_2b {capture,fidelity,verify} ...
"""

import dataclasses

from ..vision_model import make_model

WORKLOAD_CONFIG = {"model": "qwen2_vl_2b", "device": "cuda", "weights": "pretrained",
                   "stream": True}
# Gate 3: one TextVQA row per batch labels ~4 answer tokens, and the protocol's
# constant 1e-4 on every parameter made training chaotic (loss sd 2.2, compile
# drift R 1.0). Eight rows per update over 125 updates (the same 1000 rows),
# with Llama's warmup, clipping and 3e-5, learn (loss 4.2 -> 1.5) and cut the
# compile drift to R 0.077 (gate3 optimizer probe, 2026-10-22).
MODEL = dataclasses.replace(
    make_model("qwen2_vl_2b", WORKLOAD_CONFIG, gate3_steps=125),
    gate3_optimizer={"warmup_steps": 100, "clip_grad_norm": 1.0, "lr": 3e-5},
    gate3_accumulation=8)
