"""Whole-model adapters for DiT-S/2 and the TorchRec DLRM architecture."""

from dataclasses import dataclass
import hashlib
import torch

from evograd.benchmark.topdown.common.generated.spec import GeneratedSpec, canonical
from evograd.benchmark.topdown.common.generated.model import (
    build_model, make_inputs, forward_loss, dit_mlp, dit_modulation, dlrm_interaction,
)
from evograd.ops.level1.embedding_bag.reference import forward as embedding_bag
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
    elif site == "dlrm_interaction":
        s, d = a["num_sparse_features"], a["embedding_dim"]
        dims = dict(B=spec.batch_size, S=s, D=d, O=d + s * (s + 1) // 2)
    else:
        raise ValueError("embedding_bag uses multiple table configurations")
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


def _bag_cases(spec):
    from evograd.benchmark.topdown import has_snapshot, load_snapshot
    a = spec.arch
    dimensions = {
        (count, width): dict(N=count, D=a["embedding_dim"], B=spec.batch_size,
                             M=width, L=spec.batch_size * width)
        for count, width in zip(a["num_embeddings_per_feature"], a["multi_hot_sizes"])
    }
    if has_snapshot(spec.model):
        snap = load_snapshot(spec.model)
        if snap["workload_id"] == spec.workload_id:
            cases = tuple(Workload(dims=c["dims"], dtype=c["dtype"],
                                   provenance=Provenance(model=spec.model, component="embedding_bag",
                                                         source="execution", note=c["config_id"]))
                          for c in snap["level1"]["embedding_bag"]["configurations"]
                          if c["dims"] in dimensions.values())
            if len(cases) == len(dimensions):
                return cases
            raise ValueError("embedding_bag: matching snapshot lacks a table configuration")
    return tuple(Workload(dims=dims, dtype=spec.dtype,
                          provenance=Provenance(model=spec.model, component="embedding_bag",
                                                source="specification",
                                                note="Configuration-derived preflight, not a measured suite"))
                 for dims in dimensions.values())


@dataclass
class GeneratedWorkload:
    spec: GeneratedSpec
    calibration: str | None = None
    data_path: str | None = None
    #: Real data streamed per data order (verification): ``batch_for(seed)`` is
    #: the first batch of order ``data_seed + seed`` and ``train_batches(order)``
    #: its whole sequence -- Criteo rows for DLRM, a frozen VAE-latent pool
    #: with per-step noise for DiT. Off by default; then nothing changes.
    stream: bool = False
    #: A frozen model state loaded after the deterministic initialization
    #: (DiT's warm start: adaLN-Zero gates are zero at the init, so no gradient
    #: reaches its blocks there). Its SHA-256 is part of ``describe()``.
    init_checkpoint: str | None = None

    def __post_init__(self):
        if self.stream and (self.spec.input_mode != "real" or self.data_path):
            raise ValueError("a streamed workload is real-input and has no prepared artifact")
        if self.spec.input_mode == "real" and not self.data_path and not self.stream:
            raise ValueError("real generated-model workload requires prepared inputs")
        self.name = self.spec.workload_id
        self.unit_name = "latents" if self.spec.model == "dit_s_2" else "examples"
        if self.spec.model == "dit_s_2":
            sites = (Site("dit_modulation", "dit_s_2_dit_modulation", dit_modulation,
                          _cases(self.spec, "dit_modulation")),
                     Site("dit_mlp", "dit_s_2_dit_mlp", dit_mlp,
                          _cases(self.spec, "dit_mlp")))
        else:
            sites = (Site("embedding_bag", "embedding_bag", embedding_bag,
                          _bag_cases(self.spec)),
                     Site("dlrm_interaction", "torchrec_dlrm_dlrm_interaction",
                          dlrm_interaction, _cases(self.spec, "dlrm_interaction")))
        self.site_registry = SiteRegistry(self.spec.model, sites)

    def to_config(self):
        return {**self.spec.to_config(), "calibration": self.calibration, "data_path": self.data_path,
                **({"stream": True} if self.stream else {}),
                **({"init_checkpoint": self.init_checkpoint} if self.init_checkpoint else {})}

    @classmethod
    def from_config(cls, config):
        base = canonical(config["model"]).to_config()
        base.update({k: v for k, v in config.items() if k in GeneratedSpec.__dataclass_fields__})
        return cls(GeneratedSpec(**base), config.get("calibration"), config.get("data_path"),
                   stream=bool(config.get("stream", False)),
                   init_checkpoint=config.get("init_checkpoint"))

    def units_per_step(self):
        return self.spec.batch_size

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        if kernels.registry.name != self.spec.model:
            raise ValueError("kernel registry does not match model")
        model = build_model(self.spec)
        if self.init_checkpoint:
            model.load_state_dict(self.initial_state()[0], strict=True)
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
            elif site == "dlrm_interaction":
                model.interaction_fn = fn
                paths[site] = ["interaction"]
            elif site == "embedding_bag":
                import types
                for i, table in enumerate(model.tables):
                    def call(self, indices, offsets=None, _fn=fn):
                        return _fn(self.weight, indices, offsets)
                    table.forward = types.MethodType(call, table)
                    paths.setdefault(site, []).append(f"tables.{i}")
            else:
                raise ValueError(site)
        return model, PatchProvenance("module_surgery", tuple(kernels.patched),
                                      tuple(sorted(paths)),
                                      {k: tuple(v) for k, v in paths.items()})

    def initial_state(self):
        """``(state_dict, identity)`` of ``init_checkpoint``, read once."""
        if getattr(self, "_initial", None) is None:
            from evograd.evaluation.tier3.workloads.common import init_state

            self._initial = init_state.load(self.init_checkpoint, self.spec.model)
        return self._initial

    def batch_for(self, *, seed):
        if self.stream:
            return self.train_batches(self.spec.data_seed + seed).batch(0)
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
        return "latent" if self.spec.model == "dit_s_2" else "dense"

    def loss(self, model, batch):
        return self.forward(model, batch)[0]

    # ── what the verification gates read (VerifiableWorkload) ──────────────

    @property
    def primary_output_name(self):
        return "prediction" if self.spec.model == "dit_s_2" else "click_logit"

    @property
    def objective_definition(self):
        if self.spec.model == "dit_s_2":
            return ("guided-diffusion loss: epsilon MSE plus the learned-range variational bound, "
                    "mean over the batch (evograd generated.model.diffusion_loss)")
        return "binary cross entropy with logits of the click logit, mean over the batch"

    def primary_output(self, model, batch):
        """``(loss, primary)``: the loss ``loss`` returns and the model output beside it."""
        return self.forward(model, batch)

    def expected_site_calls(self):
        if self.spec.model == "dit_s_2":
            return {"dit_modulation": self.spec.arch["depth"], "dit_mlp": self.spec.arch["depth"]}
        return {"embedding_bag": self.spec.arch["num_sparse_features"], "dlrm_interaction": 1}

    def data_recipe(self):
        return {"batch_size": self.spec.batch_size, "dtype": self.spec.dtype}

    def train_batches(self, order):
        if not self.stream:
            raise ValueError("train_batches needs a streamed real-input workload")
        if self.spec.model == "dit_s_2":
            from evograd.benchmark.topdown.common.generated.data import LatentStream
            return LatentStream(self.spec, order)
        from evograd.benchmark.topdown.common.generated.dlrm_data import CriteoStream
        return CriteoStream(self.spec, order)

    def data_identity(self):
        if not self.stream:
            raise ValueError("data_identity needs a streamed real-input workload")
        if self.spec.model == "dit_s_2":
            from evograd.benchmark.topdown.common.generated.data import pool_identity
            return pool_identity()
        from evograd.benchmark.topdown.common.generated.dlrm_data import stream_identity
        return stream_identity(self.spec)

    def input_checksum(self):
        h = hashlib.sha256()
        if self.stream:
            batch = {k: v.cpu() for k, v in self.batch_for(seed=0).items()}
        else:
            batch = self.load_data(self)[0]["batches"][0] if self.data_path else self.generated_batch_cpu(0)
        for key, value in sorted(batch.items()):
            h.update(key.encode())
            h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def describe(self):
        real_source = {}
        if self.stream:
            real_source = {"stream": self.data_identity()}
        elif self.spec.input_mode == "real":
            if self.spec.model == "dit_s_2":
                from evograd.benchmark.topdown.common.generated.data import DATASET, VAE
                real_source = {"dataset_repo": DATASET[0], "dataset_revision": DATASET[1],
                               "dataset_file": DATASET[2], "vae_repo": VAE[0],
                               "vae_revision": VAE[1]}
            else:
                data, data_hash = self.load_data(self)
                real_source = {"dataset": list(data["dataset"]),
                               "source_dir": data["source_dir"],
                               "source_files": data["source_files"],
                               "source_rows_sha256": data["source_rows_sha256"],
                               "source_artifact_sha256": data_hash,
                               "selected_rows": data["selection"],
                               "sparse_mapping": data["mapping"]}
        return {"name": self.name,
                "model": "DiT-S/2" if self.spec.model == "dit_s_2" else "TorchRec DLRM example",
                "workload_hash": self.spec.workload_hash, "config": self.to_config(),
                "canonical": self.spec.is_canonical, "execution_scope": "model",
                "benchmark_level": 3, "unit_name": self.unit_name,
                "units_per_step": self.units_per_step(),
                "weights": "random" if not self.init_checkpoint else "warm_start",
                **({"init_checkpoint": self.initial_state()[1]} if self.init_checkpoint else {}),
                "data": ("streamed_real_data" if self.stream else "pinned_real_data"
                         if self.spec.input_mode == "real" else "seeded_generated"),
                **real_source,
                "all_parameters_trainable": True,
                "input_checksum": self.input_checksum(), "architecture": self.spec.arch,
                "model_gate": {"calibration": self.calibration,
                               "requires_real_data": self.spec.input_mode == "real"},
                "execution_note": (
                    "DiT epsilon MSE plus learned-range variance loss"
                    if self.spec.model == "dit_s_2" else
                    "single-device architecture and AdamW; TorchRec distributed sharding and fused sparse optimizer are outside this workload")}

    def runtime_report(self, model):
        return {"training": model.training,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device), "torch": torch.__version__}

    def model_correctness(self, kernels, *, device="cuda"):
        from evograd.evaluation.tier3.workloads.common.vision.gate import evaluate
        return evaluate(self, kernels)
