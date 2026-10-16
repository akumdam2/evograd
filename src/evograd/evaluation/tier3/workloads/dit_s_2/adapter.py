from evograd.evaluation.tier3.workloads import Tier3Adapter
from evograd.evaluation.tier3.workloads.common.generated.adapter import build, providers

ADAPTER = Tier3Adapter(
    name="dit_s_2", build=lambda args: build("dit_s_2", args), providers=providers,
    options=frozenset({"model_config", "real_input", "calibration", "data_seed",
                       "structural_identity", "compile_site", "patch_set"}),
    summary="DiT-S/2 latent denoising training, generated inputs",
)
