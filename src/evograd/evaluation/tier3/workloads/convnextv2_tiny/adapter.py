from evograd.evaluation.tier3.workloads import Tier3Adapter
from .workload import from_config


def build(args):
    import json

    config = json.loads(args.vision_config.read_text()) if args.vision_config else {}
    if args.learning_rate != 1e-4:
        raise ValueError("vision calibration uses the benchmark AdamW learning rate 1e-4")
    config.update(device=args.device, seed=args.seed, data_seed=args.data_seed)
    if getattr(args, "real_input", False):
        raise NotImplementedError("--real-input is reserved for a future pinned-data vision workload")
    config["calibration"] = str(args.calibration) if args.calibration else None
    config["data_path"] = str(args.data_path) if args.data_path else None
    for flag, field in [("batch", "batch_size"), ("dtype", "dtype")]:
        if getattr(args, flag, None) is not None:
            config[field] = getattr(args, flag)
    if args.tokens is not None:
        raise ValueError("ConvNeXt uses image_size, not --tokens")
    return from_config(config)


def providers(args, registry):
    from evograd.evaluation.tier3.providers import compile_and_patch_set_providers
    from evograd.evaluation.tier3.workloads.common.vision.workload import (
        structural_identity_kernels,
    )

    result = compile_and_patch_set_providers(args, registry)
    if getattr(args, "structural_identity", False):
        result["structural_identity"] = structural_identity_kernels(registry)
    return result


ADAPTER = Tier3Adapter(
    name="convnextv2_tiny",
    build=build,
    providers=providers,
    options=frozenset(
        {
            "structural_identity",
            "compile_site",
            "patch_set",
            "data_seed",
            "vision_config",
            "data_path",
            "real_input",
            "calibration",
        }
    ),
    summary="ConvNeXt V2 Tiny image classification, B32 x 224x224",
)
