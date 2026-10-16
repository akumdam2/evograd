"""Whole-model generated-input adapter for DiT-S/2."""

from dataclasses import dataclass
import hashlib
import torch

from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from evograd.benchmark.topdown.common.generated.model import (
    build_model, make_inputs, forward_loss, dit_mlp, dit_modulation,
)
from evograd.evaluation.tier3.patch import Site, SiteRegistry, PatchProvenance
from evograd.opdecl import Workload, Provenance


def structural_identity_kernels(registry):
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch
    result = KernelSet(registry=registry)
    for site in registry:
        info = registry.require(site)
        result = patch(result, site, info.default,
                       source=KernelSource(site=site, op_name=info.op,
                                           origin="structural_identity"))
    return result


def _cases(spec, site):
    from evograd.benchmark.topdown import has_snapshot, load_snapshot
    a = spec.arch
    if site == "dit_modulation":
        dims = dict(B=spec.batch_size, H=a["hidden_size"], J=6*a["hidden_size"])
    elif site == "dit_mlp":
        dims = dict(B=spec.batch_size, T=(a["input_size"] // a["patch_size"]) ** 2,
                    H=a["hidden_size"], I=a["hidden_size"] * a["mlp_ratio"])
    else:
        raise ValueError(site)
    name = spec.model + "_" + site
    if has_snapshot(spec.model):
        snap = load_snapshot(spec.model)
        if snap["workload_id"] == spec.workload_id:
            cases = tuple(Workload(dims=c["dims"], dtype=c["dtype"],
                                   provenance=Provenance(model=spec.model, component=site,
                                                         source="execution", note=c["config_id"]))
                          for c in snap["tasks"][name]["configurations"] if c["dims"] == dims)
            if cases:
                return cases
            raise ValueError(f"{site}: matching snapshot has no matching configuration")
    return (Workload(dims=dims, dtype=spec.dtype,
                     provenance=Provenance(model=spec.model, component=site,
                                           source="specification",
                                           note="Configuration-derived preflight, not a measured suite")),)


@dataclass
class GeneratedWorkload:
    spec: GeneratedSpec
    calibration: str | None = None
    data_path: str | None = None

    def __post_init__(self):
        if self.spec.input_mode == "real" and not self.data_path:
            raise ValueError("real DiT workload requires prepared pinned inputs")
        self.name = self.spec.workload_id
        self.unit_name = "latents"
        sites = (Site("dit_modulation", "dit_s_2_dit_modulation", dit_modulation,
                      _cases(self.spec, "dit_modulation")),
                 Site("dit_mlp", "dit_s_2_dit_mlp", dit_mlp,
                      _cases(self.spec, "dit_mlp")))
        self.site_registry = SiteRegistry(self.spec.model, sites)

    def to_config(self):
        return {**self.spec.to_config(), "calibration": self.calibration, "data_path": self.data_path}

    @classmethod
    def from_config(cls, config):
        base = canonical(config["model"]).to_config()
        base.update({k: v for k, v in config.items() if k in GeneratedSpec.__dataclass_fields__})
        return cls(GeneratedSpec(**base), config.get("calibration"), config.get("data_path"))

    def units_per_step(self):
        return self.spec.batch_size

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        if kernels.registry.name != self.spec.model:
            raise ValueError("kernel registry does not match model")
        model = build_model(self.spec)
        paths = {}
        for site in kernels.patched:
            fn = kernels.kernel_for(site)
            if site == "dit_mlp":
                for i, block in enumerate(model.blocks):
                    block.mlp_fn = fn
                    paths.setdefault(site, []).append(f"blocks.{i}")
            elif site == "dit_modulation":
                for i, block in enumerate(model.blocks):
                    block.mod_fn = fn
                    paths.setdefault(site, []).append(f"blocks.{i}")
            else:
                raise ValueError(site)
        return model, PatchProvenance("module_surgery", tuple(kernels.patched),
                                      tuple(sorted(paths)),
                                      {k: tuple(v) for k, v in paths.items()})

    def batch_for(self, *, seed):
        if self.data_path:
            if not hasattr(self, "_prepared_batches"):
                self._prepared_batches = self.load_data(self)[0]["batches"]
            batch = self._prepared_batches[seed % len(self._prepared_batches)]
            return {k: v.to(self.spec.device) for k, v in batch.items()}
        return make_inputs(self.spec, seed)

    @staticmethod
    def load_data(workload):
        from evograd.benchmark.topdown.common.generated.data import load_data
        return load_data(workload)

    def generated_batch_cpu(self, seed):
        return make_inputs(self.spec.replace(device="cpu"), seed)

    def forward(self, model, batch):
        return forward_loss(model, batch, self.spec.model)

    @property
    def input_gradient_key(self):
        return "latent"

    def loss(self, model, batch):
        return self.forward(model, batch)[0]

    def input_checksum(self):
        h = hashlib.sha256()
        batch = self.load_data(self)[0]["batches"][0] if self.data_path else self.generated_batch_cpu(0)
        for key, value in sorted(batch.items()):
            h.update(key.encode())
            h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def describe(self):
        real_source = {}
        if self.spec.input_mode == "real":
            from evograd.benchmark.topdown.common.generated.data import DATASET, VAE
            real_source = {"dataset_repo": DATASET[0], "dataset_revision": DATASET[1],
                           "dataset_file": DATASET[2], "vae_repo": VAE[0],
                           "vae_revision": VAE[1]}
        return {"name": self.name, "model": "DiT-S/2",
                "workload_hash": self.spec.workload_hash, "config": self.to_config(),
                "canonical": self.spec.is_canonical, "execution_scope": "model",
                "benchmark_level": 3, "unit_name": self.unit_name,
                "units_per_step": self.units_per_step(), "weights": "random",
                "data": "pinned_real_data" if self.spec.input_mode == "real" else "seeded_generated",
                **real_source,
                "all_parameters_trainable": True,
                "input_checksum": self.input_checksum(), "architecture": self.spec.arch,
                "model_gate": {"calibration": self.calibration,
                               "requires_real_data": self.spec.input_mode == "real"},
                "execution_note": "DiT epsilon MSE plus learned-range variance loss"}

    def runtime_report(self, model):
        return {"training": model.training,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device), "torch": torch.__version__}

    def model_correctness(self, kernels, *, device="cuda"):
        from evograd.evaluation.tier3.workloads.common.vision.gate import evaluate
        return evaluate(self, kernels)
