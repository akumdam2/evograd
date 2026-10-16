"""Whole-model TrainingWorkload adapter shared by the two vision workloads."""

from dataclasses import dataclass
import hashlib
import torch
from evograd.benchmark.topdown.common.vision.spec import VisionSpec, REVISIONS
from evograd.benchmark.topdown.common.vision.model import build_model, make_inputs, forward
from evograd.benchmark.topdown.common.vision.operators import FUNCTIONS
from .sites import install, preflight_cases

SITES = {
    "convnextv2_tiny": tuple(f"convnext_mlp_stage{i}" for i in range(4)),
    "qwen2_vl_2b": ("text_mlp", "vision_mlp"),
}


def structural_identity_kernels(registry):
    from evograd.evaluation.tier3.patch import KernelSet, patch, KernelSource

    kernels = KernelSet(registry=registry)
    for site in registry:
        kernels = patch(
            kernels,
            site,
            registry.require(site).default,
            source=KernelSource(
                site=site, op_name=registry.require(site).op, origin="structural_identity"
            ),
        )
    return kernels


@dataclass
class VisionWorkload:
    spec: VisionSpec
    calibration: str | None = None
    data_path: str | None = None

    def __post_init__(self):
        from evograd.evaluation.tier3.patch import Site, SiteRegistry

        if self.spec.weights == "pretrained" and not self.data_path:
            raise ValueError("pretrained vision workload requires pinned real-data batches")

        self.name = self.spec.workload_id
        self.unit_name = "images" if self.spec.model == "convnextv2_tiny" else "tokens"
        self.site_registry = SiteRegistry(
            self.spec.model,
            tuple(
                Site(
                    site,
                    self.spec.model
                    + "_"
                    + ("convnext_mlp" if site.startswith("convnext_mlp_stage") else site),
                    FUNCTIONS["convnext_mlp" if site.startswith("convnext_mlp_stage") else site],
                    preflight_cases(self.spec, site),
                )
                for site in SITES[self.spec.model]
            ),
        )

    def to_config(self):
        return {
            **self.spec.to_config(),
            "calibration": self.calibration,
            "data_path": self.data_path,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            VisionSpec(**{k: v for k, v in config.items() if k in VisionSpec.__dataclass_fields__}),
            calibration=config.get("calibration"),
            data_path=config.get("data_path"),
        )

    def units_per_step(self):
        return self.spec.batch_size * (self.spec.seq_len if self.unit_name == "tokens" else 1)

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        if kernels.registry.name != self.spec.model:
            raise ValueError("kernel registry does not match workload")
        model = build_model(self.spec)
        provenance = install(model, kernels)
        expected = (
            {f"convnext_mlp_stage{i}": depth for i, depth in enumerate(self.spec.arch["depths"])}
            if self.spec.model == "convnextv2_tiny"
            else {
                "text_mlp": self.spec.arch["num_hidden_layers"],
                "vision_mlp": self.spec.arch["vision_config"]["depth"],
            }
        )
        for site in kernels.patched:
            if len(provenance.paths[site]) != expected[site]:
                raise ValueError(f"{site}: replacement module count differs from architecture")
        return model, provenance

    def batch_for(self, *, seed):
        if self.data_path:
            batches = torch.load(self.data_path, map_location="cpu", weights_only=True)["batches"]
            if not batches:
                raise ValueError("empty batch artifact")
            return {k: v.to(self.spec.device) for k, v in batches[seed % len(batches)].items()}
        return make_inputs(self.spec, seed)

    def loss(self, model, batch):
        return forward(model, batch, self.spec.model).loss

    def describe(self):
        return {
            "name": self.name,
            "model": self.spec.model_name,
            "workload_hash": self.spec.workload_hash,
            "config": self.to_config(),
            "canonical": self.spec.is_canonical,
            "execution_scope": "model",
            "benchmark_level": 3,
            "unit_name": self.unit_name,
            "units_per_step": self.units_per_step(),
            "weights": self.spec.weights,
            "checkpoint_revision": REVISIONS[self.spec.model] if self.spec.weights == "pretrained" else None,
            "data": "pinned_real_data" if self.spec.weights == "pretrained" else ("prepared_data" if self.data_path else "synthetic"),
            "all_parameters_trainable": True,
            "gradient_checkpointing": False,
            "use_cache": False,
            "input_checksum": self.input_checksum(),
            "architecture": self.spec.arch,
            "model_gate": {"calibration": self.calibration, "requires_real_data": self.spec.weights == "pretrained"},
        }

    def input_checksum(self):
        batch = (
            torch.load(self.data_path, map_location="cpu", weights_only=True)["batches"][0]
            if self.data_path
            else make_inputs(self.spec.replace(device="cpu"), 0)
        )
        digest = hashlib.sha256()
        for key, value in sorted(batch.items()):
            digest.update(key.encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def runtime_report(self, model):
        attention = {
            key: getattr(getattr(model.config, key, None), "_attn_implementation", None)
            for key in ("text_config", "vision_config")
        }
        return {
            "attention_dispatch": attention,
            "training": model.training,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
            "transformers": __import__("transformers").__version__,
        }

    def model_correctness(self, kernels, *, device="cuda"):
        from .gate import evaluate

        return evaluate(self, kernels)
