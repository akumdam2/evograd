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
#: Model-scope Gate 2 at 3, not the protocol's 4 (2026-10-27): the stage-0 convnext_mlp
#: seed needs 2.96, while a 1% fault at stage 3 (stage 1) passes at 3.13 (4.0), so 4 lets
#: it through at model scope. Operator Gate 2 rejects those faults either way (>= 100).
#: Gate 3 over 50 updates at 3, not 100 at 2 (2026-10-27): the loss falls to ~0.001
#: within a few hundred updates and is recorded in bf16, so a perturbation shows as a
#: few loss ULPs. Over 10-75 updates the seeds need at most 2.25 and a 5% stage-0 fault
#: at least 3.21; over 100 that fault needs only 1.99.
MODEL = dataclasses.replace(make_model("convnextv2_tiny", WORKLOAD_CONFIG, gate3_steps=50),
                            gate2_model_gamma=3.0, gate3_gamma=3.0)
