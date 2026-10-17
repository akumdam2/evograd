"""Freeze generated-input model-gate budgets for DiT-S/2."""

import argparse
import json
from pathlib import Path

from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from .workload import GeneratedWorkload
from evograd.evaluation.tier3.workloads.common.vision.gate import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("dit_s_2", "torchrec_dlrm"))
    parser.add_argument("--config", type=Path, help="JSON execution overrides")
    parser.add_argument("--real-input", action="store_true", help="use prepared model-specific real inputs")
    parser.add_argument("--data", type=Path, help="prepared real-input artifact path")
    parser.add_argument("--criteo-dir", type=Path, help="TorchRec-preprocessed Criteo Kaggle NumPy directory")
    parser.add_argument("--sites", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile-backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = canonical(args.model).to_config()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    if config["model"] != args.model:
        raise ValueError("config names a different model")
    config["input_mode"] = "real" if args.real_input else "generated"
    spec = GeneratedSpec(**config)
    data_path = None
    if args.real_input:
        from evograd.benchmark.topdown.common.generated.data import ensure_pinned_data
        data_path = ensure_pinned_data(spec, args.data, batch_count=2 * args.steps,
                                       source_dir=args.criteo_dir)
    workload = GeneratedWorkload(spec, data_path=data_path)
    row = calibrate(workload, args.sites.split(","), args.steps, args.compile_backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
