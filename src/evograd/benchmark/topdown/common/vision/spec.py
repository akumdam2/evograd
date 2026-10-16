"""Explicit, serializable execution contracts; no checkpoint/network access."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

MODELS = {
    "convnextv2_tiny": ("facebook/convnextv2-tiny-1k-224", 32, 0),
    "qwen2_vl_2b": ("Qwen/Qwen2-VL-2B-Instruct", 1, 512),
}


REVISIONS = {
    "convnextv2_tiny": "f4db009e63145e02b3c075aa64d90ce41bcca4b1",
    "qwen2_vl_2b": "895c3a49bc3fa70a340399125c650a463535e71c",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def architecture(model):
    return json.loads((Path(__file__).parents[2] / model / "config.json").read_text())


@dataclass(frozen=True)
class VisionSpec:
    model: str
    batch_size: int
    seq_len: int
    image_size: int = 224
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    data_seed: int = 0
    attn_implementation: str = "sdpa"
    arch_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown vision workload {self.model}")
        unknown = set(self.arch_overrides) - set(architecture(self.model))
        if unknown:
            raise ValueError(f"unknown architecture override keys: {sorted(unknown)}")
        if self.batch_size <= 0 or self.image_size <= 0:
            raise ValueError("batch_size and image_size must be positive")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("supported dtypes: float32, bfloat16")
        if self.attn_implementation not in ("sdpa", "eager"):
            raise ValueError("supported attention: sdpa, eager")
        if self.model == "convnextv2_tiny":
            if (
                self.arch["hidden_act"] != "gelu"
                or self.arch["num_channels"] != 3
                or self.arch["patch_size"] != 4
            ):
                raise ValueError("ConvNeXt boundaries require RGB, patch size 4 and exact GELU")
            if (
                len(self.arch["depths"]) != 4
                or len(self.arch["hidden_sizes"]) != 4
                or min(self.arch["depths"] + self.arch["hidden_sizes"]) <= 0
            ):
                raise ValueError("ConvNeXt requires four nonempty stages with positive widths")
            if self.image_size < 32 or self.image_size % 32 or self.seq_len != 0:
                raise ValueError("ConvNeXt needs image_size divisible by 32 and seq_len=0")
        else:
            v = self.arch["vision_config"]
            if (
                self.arch["hidden_act"] != "silu"
                or v.get("hidden_act", "quick_gelu") != "quick_gelu"
            ):
                raise ValueError("Qwen2-VL boundaries require SiLU text and QuickGELU vision MLPs")
            if v["in_chans"] != 3 or v["hidden_size"] != self.arch["hidden_size"]:
                raise ValueError("Qwen2-VL requires RGB and matching merger/text widths")
            tile = v["patch_size"] * v["spatial_merge_size"]
            if self.image_size % tile or self.seq_len < (self.image_size // tile) ** 2 + 4:
                raise ValueError(
                    "Qwen2-VL needs complete merged patches and space for image markers/text"
                )

    @property
    def arch(self):
        config = architecture(self.model)
        for key, value in self.arch_overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
        return config

    @property
    def model_name(self):
        return MODELS[self.model][0]

    def to_config(self):
        return asdict(self)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @property
    def workload_hash(self):
        return digest({"execution": self.to_config(), "architecture": self.arch})

    @property
    def workload_id(self):
        return f"{self.model}-{self.workload_hash[:16]}"

    @property
    def is_canonical(self):
        return self == canonical(self.model)


def canonical(model):
    _, batch, seq = MODELS[model]
    return VisionSpec(model=model, batch_size=batch, seq_len=seq)
