"""DiT-S/2: the three gates on real Imagenette latents.

A frozen VAE-latent pool of every Imagenette training image, with fresh noise,
timesteps and label drop at every step (``GeneratedWorkload(stream=True)``);
random weights, as in the benchmark (there is no published S/2 checkpoint).
Capture and model reference are shared with DLRM.

    python -m evograd.verification.integrations dit_s_2 {capture,fidelity,verify} ...
"""

from ..generated_model import make_model

WORKLOAD_CONFIG = {"model": "dit_s_2", "device": "cuda", "input_mode": "real", "stream": True}
MODEL = make_model("dit_s_2", WORKLOAD_CONFIG)
