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
# 10 updates, not 125 (2026-10-27): around update 15 the loss falls fastest and every
# perturbation is amplified alike (a seed's and a 5% fault's loss gaps reach 0.19 and
# 0.25), so from 15 to 125 updates the text-MLP seed needs 1.3-4.0 and the fault 1.4-4.8.
# Over the first 10 the seeds need at most 0.67 and the fault 3.97. Interim: a setting
# that also separates them later has not been found.
MODEL = dataclasses.replace(
    make_model("qwen2_vl_2b", WORKLOAD_CONFIG, gate3_steps=10),
    gate3_optimizer={"warmup_steps": 100, "clip_grad_norm": 1.0, "lr": 3e-5},
    gate3_accumulation=8)
