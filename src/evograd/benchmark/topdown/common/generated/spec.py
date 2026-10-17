"""Serializable DiT-S/2 and TorchRec DLRM execution configurations."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GeneratedSpec:
    model: str
    batch_size: int = 32
    dtype: str = "float32"
    device: str = "cuda"
    seed: int = 0
    data_seed: int = 0
    input_mode: str = "generated"
    arch_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model not in ("dit_s_2", "torchrec_dlrm"):
            raise ValueError(f"unknown generated model {self.model}")
        if self.batch_size < 1 or self.dtype not in ("float32", "bfloat16"):
            raise ValueError("positive batch and float32/bfloat16 required")
        if self.input_mode not in ("generated", "real"):
            raise ValueError("input_mode must be generated or real")
        if self.model == "torchrec_dlrm" and self.input_mode == "real":
            raise NotImplementedError("--real-input is reserved for TorchRec DLRM")
        if self.input_mode == "real" and self.arch_overrides:
            raise ValueError("real-input requires the published DiT-S/2 architecture")
        base = architecture(self.model)
        unknown = set(self.arch_overrides) - set(base)
        if unknown:
            raise ValueError(f"unknown architecture overrides: {sorted(unknown)}")
        arch = self.arch
        if self.model == "dit_s_2":
            if arch["input_size"] % arch["patch_size"] or arch["hidden_size"] % arch["num_heads"]:
                raise ValueError("invalid DiT patch/head configuration")
            if min(arch[k] for k in ("input_size", "patch_size", "in_channels", "hidden_size", "depth", "num_heads", "num_classes")) < 1 or arch["diffusion_steps"] < 2:
                raise ValueError("DiT dimensions must be positive")
        elif (len(arch["num_embeddings_per_feature"]) != arch["num_sparse_features"]
              or len(arch["multi_hot_sizes"]) != arch["num_sparse_features"]
              or min(arch["num_embeddings_per_feature"]) < 2
              or min(arch["multi_hot_sizes"]) < 1
              or min(arch["dense_arch_layer_sizes"]) < 1
              or min(arch["over_arch_layer_sizes"]) < 1
              or arch["dense_arch_layer_sizes"][-1] != arch["embedding_dim"]
              or arch["over_arch_layer_sizes"][-1] != 1):
            raise ValueError("invalid DLRM table or MLP configuration")

    @property
    def arch(self):
        return {**architecture(self.model), **self.arch_overrides}

    @property
    def workload_hash(self):
        execution = self.to_config()
        if self.input_mode == "generated":
            execution.pop("input_mode")  # Preserve the existing generated workload identity.
        else:
            from .data import DATASET, VAE
            execution.update(dataset=DATASET, vae=VAE)
        return digest({"execution": execution, "architecture": self.arch})

    @property
    def workload_id(self):
        return f"{self.model}-{self.workload_hash[:16]}"

    @property
    def is_canonical(self):
        return self == canonical(self.model)

    def to_config(self):
        return asdict(self)

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def architecture(model):
    if model == "dit_s_2":
        return dict(input_size=32, patch_size=2, in_channels=4, hidden_size=384,
                    depth=12, num_heads=6, mlp_ratio=4, num_classes=1000,
                    class_dropout_prob=0.1, diffusion_steps=1000)
    if model == "torchrec_dlrm":
        return dict(num_dense_features=13, num_sparse_features=26, embedding_dim=64,
                    num_embeddings_per_feature=[100_000] * 26,
                    dense_arch_layer_sizes=[512, 256, 64],
                    over_arch_layer_sizes=[512, 512, 256, 1],
                    multi_hot_sizes=[1] * 26)
    raise ValueError(model)


def canonical(model):
    return GeneratedSpec(model=model, batch_size=256 if model == "dit_s_2" else 32)
