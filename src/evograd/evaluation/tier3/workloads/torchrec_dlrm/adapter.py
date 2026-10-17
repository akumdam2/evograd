from evograd.evaluation.tier3.workloads import Tier3Adapter
from evograd.evaluation.tier3.workloads.common.generated.adapter import build, providers

ADAPTER = Tier3Adapter(
    name="torchrec_dlrm", build=lambda args: build("torchrec_dlrm", args), providers=providers,
    options=frozenset({"model_config", "real_input", "data_path", "criteo_dir",
                       "calibration", "data_seed",
                       "structural_identity", "compile_site", "patch_set"}),
    summary="Single-device TorchRec DLRM architecture, generated or Criteo inputs",
)
