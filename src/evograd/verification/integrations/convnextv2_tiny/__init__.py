"""ConvNeXt V2 Tiny: the three gates on real Imagenette images.

Pinned pretrained ``facebook/convnextv2-tiny-1k-224`` and a frozen pool of every
Imagenette training image through its image processor; capture and model
reference are shared with Qwen2-VL (:mod:`..vision`, :mod:`..vision_model`).
The four ``convnext_mlp`` stage sites are one task at four configurations.

    python -m evograd.verification.integrations convnextv2_tiny {capture,fidelity,verify} ...
"""

import dataclasses

from ..vision_model import make_model

WORKLOAD_CONFIG = {"model": "convnextv2_tiny", "device": "cuda", "weights": "pretrained",
                   "stream": True}
#: Model-scope Gate 2 at the protocol's 4: under calibration /4 the convnext_mlp seeds
#: need at most 1.4 (2026-11-02). 1% faults pass it from 2.43, but operator Gate 2 stops
#: all of them (they need >= 2.6 there).
#: Gate 3 over 50 updates at 3, not 100 at 2 (2026-10-27): the loss falls to ~0.001
#: within a few hundred updates and is recorded in bf16, so a perturbation shows as a
#: few loss ULPs. Over 10-75 updates the seeds need at most 2.25 and a 5% stage-0 fault
#: at least 3.21; over 100 that fault needs only 1.99.
MODEL = dataclasses.replace(make_model("convnextv2_tiny", WORKLOAD_CONFIG, gate3_steps=50),
                            gate3_gamma=3.0)
