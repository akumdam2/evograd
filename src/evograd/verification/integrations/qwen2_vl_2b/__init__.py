"""Qwen2-VL-2B-Instruct: the three gates on real TextVQA rows.

Pinned pretrained weights and a frozen pool of TextVQA shard-0 rows (one image
at 224x224, the question, the answer as the only labelled tokens); capture and
model reference are shared with ConvNeXt V2 (:mod:`..vision`, :mod:`..vision_model`).

    python -m evograd.verification.integrations qwen2_vl_2b {capture,fidelity,verify} ...
"""

from ..vision_model import make_model

WORKLOAD_CONFIG = {"model": "qwen2_vl_2b", "device": "cuda", "weights": "pretrained",
                   "stream": True}
MODEL = make_model("qwen2_vl_2b", WORKLOAD_CONFIG)
