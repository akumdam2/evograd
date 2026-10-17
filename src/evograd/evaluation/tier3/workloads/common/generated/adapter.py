"""DiT-S/2 Tier-3 CLI adapter construction."""

import json

from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from .workload import GeneratedWorkload, structural_identity_kernels


def build(model, args):
    if args.learning_rate != 1e-4:
        raise ValueError("generated model calibration uses AdamW learning rate 1e-4")
    if args.tokens is not None:
        raise ValueError("this model does not take --tokens")
    config = canonical(model).to_config()
    if getattr(args, "model_config", None):
        config.update(json.loads(args.model_config.read_text()))
    config.update(device=args.device, seed=args.seed, data_seed=args.data_seed)
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.dtype is not None:
        config["dtype"] = args.dtype
    config["input_mode"] = "real" if getattr(args, "real_input", False) else "generated"
    if config["model"] != model:
        raise ValueError("config names a different model")
    spec = GeneratedSpec(**config)
    data_path = None
    if spec.input_mode == "real":
        from evograd.benchmark.topdown.common.generated.data import ensure_pinned_data
        calibration_steps = (json.loads(args.calibration.read_text())["steps"]
                             if args.calibration else 5)
        data_path = ensure_pinned_data(spec, getattr(args, "data_path", None),
                                       batch_count=2 * calibration_steps,
                                       source_dir=getattr(args, "criteo_dir", None))
    return GeneratedWorkload(spec, str(args.calibration) if args.calibration else None, data_path)


def providers(args, registry):
    from evograd.evaluation.tier3.providers import compile_and_patch_set_providers
    result = compile_and_patch_set_providers(args, registry)
    if getattr(args, "structural_identity", False):
        result["structural_identity"] = structural_identity_kernels(registry)
    return result
