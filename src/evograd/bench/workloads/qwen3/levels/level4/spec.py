"""The canonical Qwen3 training workload: what is fixed, and what may vary.

A workload that can be tuned is not a reference. Everything that changes what
the GPU executes -- architecture, batch, sequence length, dtype, attention
backend, cache and checkpointing state, seed -- is pinned here in one frozen
object, and the identity of a run is a hash over exactly those fields. A run
that deviates in any of them gets a different id and is reported as
non-canonical, so a shrunken debug run can never be mistaken for the reference.

Three settings are not merely defaults but *rules*, and :meth:`WorkloadSpec.validate`
rejects any spec that breaks them:

``use_cache=False``
    A KV cache is a decode-time structure. With it on, the module allocates and
    returns per-layer caches that training never reads, so the traced graph is
    not the graph training runs.

``gradient_checkpointing=False``
    Checkpointing re-executes forward regions inside backward. Every operator
    invocation would then be counted twice, and the backward-pass timings a
    later stage collects would describe recomputation rather than the backward
    itself.

``training=True``
    ``model.eval()`` changes dropout and, in other architectures, normalization
    statistics. This workload exists to represent training.

The architecture below is Qwen3-0.6B's published configuration, written out
rather than downloaded: the reference must be reproducible on a node with no
network, and a config fetched at runtime would make the workload identity
depend on what the Hub served that day.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

#: Qwen3-0.6B, exactly as published (``Qwen/Qwen3-0.6B`` ``config.json``).
#: 596M parameters total, 440M excluding the tied embedding -- the "0.6B" and
#: "0.44B non-embedding" of the model card, which
#: ``analytic_parameter_count`` reproduces from these numbers alone.
QWEN3_0_6B: Mapping[str, Any] = {
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": None,
    "use_sliding_window": False,
    "max_window_layers": 28,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}

MODEL_NAME = "Qwen3-0.6B"

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")

#: Only backends this milestone has verified end to end. Anything else is
#: rejected here rather than in Transformers, so the message names the workload.
SUPPORTED_ATTENTION = ("sdpa", "eager")


class WorkloadSpecError(ValueError):
    """A workload spec that violates the Level-4 workload contract."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkloadSpec:
    """One fully-determined training execution.

    ``arch`` is stored as a sorted tuple of items rather than a dict so the
    dataclass stays hashable and two specs built from differently-ordered
    mappings compare equal.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(QWEN3_0_6B.items()))
    batch_size: int = 2
    seq_len: int = 2048
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    use_cache: bool = False
    gradient_checkpointing: bool = False
    training: bool = True
    seed: int = 0

    # ---- derived views -------------------------------------------------

    @property
    def arch(self) -> dict[str, Any]:
        return dict(self.arch_items)

    @property
    def token_count(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def config_hash(self) -> str:
        """Identity of the *architecture*, independent of how it is run."""
        return _sha256({"model_name": self.model_name, "arch": self.arch})[:16]

    @property
    def workload_hash(self) -> str:
        """Identity of the whole execution: architecture plus every run setting."""
        return _sha256(self.to_dict())[:16]

    @property
    def workload_id(self) -> str:
        """A stable, readable name. The hash suffix covers the fields the slug
        leaves out (seed, architecture), so two ids never collide silently."""
        slug = self.model_name.lower().replace("/", "-")
        mode = "train" if self.training else "eval"
        short = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}[self.dtype]
        return (
            f"{slug}.{mode}.bs{self.batch_size}.seq{self.seq_len}."
            f"{short}.{self.device}.{self.attn_implementation}.{self.workload_hash[:8]}"
        )

    @property
    def is_canonical(self) -> bool:
        return self == CANONICAL

    # ---- construction --------------------------------------------------

    def replace(self, **overrides: Any) -> "WorkloadSpec":
        """A debug or test variant. Size may shrink; the rules may not bend --
        :meth:`validate` still runs, so ``use_cache=True`` is refused here too."""
        arch_overrides = overrides.pop("arch", None)
        if arch_overrides is not None:
            merged = {**self.arch, **dict(arch_overrides)}
            overrides["arch_items"] = tuple(sorted(merged.items()))
        spec = dataclasses.replace(self, **overrides)
        spec.validate()
        return spec

    # ---- the contract --------------------------------------------------

    def validate(self) -> "WorkloadSpec":
        arch = self.arch
        if self.batch_size < 1:
            raise WorkloadSpecError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.seq_len < 1:
            raise WorkloadSpecError(f"seq_len must be >= 1, got {self.seq_len}")
        max_pos = arch.get("max_position_embeddings")
        if max_pos is not None and self.seq_len > max_pos:
            raise WorkloadSpecError(
                f"seq_len={self.seq_len} exceeds the architecture's "
                f"max_position_embeddings={max_pos}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise WorkloadSpecError(
                f"dtype={self.dtype!r} is not supported; choose one of {SUPPORTED_DTYPES}"
            )
        if self.attn_implementation not in SUPPORTED_ATTENTION:
            raise WorkloadSpecError(
                f"attn_implementation={self.attn_implementation!r} is not a Level-4 "
                f"backend for this milestone; choose one of {SUPPORTED_ATTENTION}. "
                "The canonical workload is 'sdpa'."
            )
        if self.use_cache:
            raise WorkloadSpecError(
                "use_cache must be False. A KV cache is a decode-time structure: with "
                "it on the model allocates and returns per-layer caches that a training "
                "step never reads, so the executed graph is not the training graph this "
                "workload is defined to represent."
            )
        if self.gradient_checkpointing:
            raise WorkloadSpecError(
                "gradient_checkpointing must be False. Checkpointing re-executes forward "
                "regions inside backward, which would double-count every operator "
                "invocation a later observation stage records."
            )
        if not self.training:
            raise WorkloadSpecError(
                "training must be True; this workload is defined as a training step."
            )
        heads = arch.get("num_attention_heads")
        kv_heads = arch.get("num_key_value_heads")
        if heads and kv_heads and heads % kv_heads:
            raise WorkloadSpecError(
                f"num_attention_heads={heads} must be divisible by "
                f"num_key_value_heads={kv_heads}"
            )
        return self

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "dtype": self.dtype,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "use_cache": self.use_cache,
            "gradient_checkpointing": self.gradient_checkpointing,
            "training": self.training,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkloadSpec":
        payload = dict(payload)
        arch = payload.pop("arch", None)
        if arch is not None:
            payload["arch_items"] = tuple(sorted(dict(arch).items()))
        return cls(**payload)


#: The reference execution. Every number this milestone reports comes from a run
#: whose spec equals this one.
CANONICAL = WorkloadSpec().validate()


def analytic_parameter_count(arch: Mapping[str, Any]) -> dict[str, int]:
    """Parameter counts derived from the config alone, without building a model.

    Lets a CPU-only test assert the pinned architecture really is Qwen3-0.6B
    (596M total / 440M non-embedding) instead of trusting the numbers were typed
    in correctly.
    """
    h = arch["hidden_size"]
    layers = arch["num_hidden_layers"]
    heads = arch["num_attention_heads"]
    kv_heads = arch["num_key_value_heads"]
    head_dim = arch.get("head_dim") or h // heads
    inter = arch["intermediate_size"]
    vocab = arch["vocab_size"]
    bias = 1 if arch.get("attention_bias") else 0

    q = h * heads * head_dim + bias * heads * head_dim
    k = h * kv_heads * head_dim + bias * kv_heads * head_dim
    v = k
    o = heads * head_dim * h
    # Qwen3's distinguishing feature: RMSNorm on q and k, over head_dim.
    qk_norm = 2 * head_dim
    mlp = 3 * h * inter
    norms = 2 * h  # input_layernorm + post_attention_layernorm
    per_layer = q + k + v + o + qk_norm + mlp + norms

    embedding = vocab * h
    non_embedding = layers * per_layer + h  # + final norm
    lm_head = 0 if arch.get("tie_word_embeddings") else vocab * h
    return {
        "per_layer": per_layer,
        "embedding": embedding,
        "lm_head": lm_head,
        "non_embedding": non_embedding,
        "total": embedding + non_embedding + lm_head,
    }
