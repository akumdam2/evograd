"""DiT-S/2: the three gates on real Imagenette latents.

A frozen VAE-latent pool of every Imagenette training image, with fresh noise,
timesteps and label drop at every step (``GeneratedWorkload(stream=True)``).
There is no published S/2 checkpoint, and at the initialization adaLN-Zero
leaves every block without gradient; verification starts from a frozen warm
start of the initialization (``integrations.warm_start``: 1000 AdamW steps on
data order 909). Capture and model reference are shared with DLRM.

    python -m evograd.verification.integrations dit_s_2 {capture,fidelity,verify} ...
"""

from ..generated_model import make_model
from ..warm_start import default_path

WORKLOAD_CONFIG = {"model": "dit_s_2", "device": "cuda", "input_mode": "real", "stream": True,
                   "init_checkpoint": str(default_path("dit_s_2"))}
MODEL = make_model("dit_s_2", WORKLOAD_CONFIG)
