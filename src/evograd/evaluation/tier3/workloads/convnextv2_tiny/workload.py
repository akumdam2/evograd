from evograd.evaluation.tier3.workloads.common.vision.workload import VisionWorkload
from evograd.benchmark.topdown.common.vision.spec import canonical, VisionSpec


def from_config(config):
    values = canonical("convnextv2_tiny").to_config()
    values.update({k: v for k, v in config.items() if k in VisionSpec.__dataclass_fields__})
    if values["model"] != "convnextv2_tiny":
        raise ValueError("configuration names a different model")
    return VisionWorkload(
        VisionSpec(**values),
        calibration=config.get("calibration"),
        data_path=config.get("data_path"),
    )
