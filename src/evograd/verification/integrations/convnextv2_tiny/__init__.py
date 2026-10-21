"""ConvNeXt V2 Tiny: the three gates on real Imagenette images.

Pinned pretrained ``facebook/convnextv2-tiny-1k-224`` and a frozen pool of every
Imagenette training image through its image processor; capture and model
reference are shared with Qwen2-VL (:mod:`..vision`, :mod:`..vision_model`).
The four ``convnext_mlp`` stage sites are one task at four configurations.

    python -m evograd.verification.integrations convnextv2_tiny {capture,fidelity,verify} ...
"""

from ..vision_model import make_model

WORKLOAD_CONFIG = {"model": "convnextv2_tiny", "device": "cuda", "weights": "pretrained",
                   "stream": True}
MODEL = make_model("convnextv2_tiny", WORKLOAD_CONFIG)
