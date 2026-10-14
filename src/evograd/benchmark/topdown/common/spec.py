"""One fully-determined training execution: what is fixed, and what may vary.

A workload that can be tuned is not a reference. Everything that changes what
the GPU executes -- architecture, batch, sequence length, dtype, attention
backend, cache and checkpointing state, seed -- is pinned in one frozen object,
and the identity of a run is a hash over exactly those fields. A run that
deviates in any of them gets a different id and is reported as non-canonical, so
a shrunken debug run can never be mistaken for the reference.

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

**How a workload uses this.** Subclass it and override the field defaults with
that model's architecture and canonical run settings. The defaults *are* the
canonical spec -- :attr:`is_canonical` compares against ``type(self)()`` -- so
there is exactly one place a canonical value is written. Subclassing changes
neither :meth:`to_dict` nor the hashes computed from it, so a workload's
identity depends on its values and not on which class holds them.

Architectures are written out rather than downloaded: a reference must be
reproducible on a node with no network, and a config fetched at runtime would
make the workload identity depend on what the Hub served that day.
"""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")

#: Only backends verified end to end. Anything else is rejected here rather than
#: in Transformers, so the message names the workload.
SUPPORTED_ATTENTION = ("sdpa", "eager")


class WorkloadSpecError(ValueError):
    """A workload spec that violates the Level-3 workload contract."""


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

    Subclass this to declare a workload; see the module docstring.
    """

    #: Overridden by each workload; the defaults below are placeholders that a
    #: subclass replaces with its own model's values.
    model_name: str = "unnamed"
    arch_items: tuple[tuple[str, Any], ...] = ()
    batch_size: int = 1
    seq_len: int = 1
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
        """The declared architecture, deep-copied.

        A shallow ``dict(self.arch_items)`` hands out the *same* nested objects
        the declaration holds, and ``build_config`` passes them straight to
        ``AutoConfig``. Transformers normalises RoPE by writing ``rope_theta``
        into ``rope_scaling`` **in place**, which silently rewrote the frozen
        declaration: after one ``build_model`` the module-level config had an
        extra key, so ``config_hash`` and ``workload_id`` changed mid-process
        and a harvest recorded a different id than the capture expected.

        Latent until a model declared a non-``None`` ``rope_scaling``; Qwen3-0.6B
        and Meta-Llama-3-8B both declare ``None``, Llama-3.2-1B does not.
        """
        return copy.deepcopy(dict(self.arch_items))

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
        """Is this the reference execution for its own workload?

        ``type(self)()`` rather than a module-level constant: a subclass's field
        defaults are its canonical spec, so this stays correct for every
        workload without any of them restating it.
        """
        return self == type(self)()

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
                f"attn_implementation={self.attn_implementation!r} is not a Level-3 "
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



def analytic_parameter_count(arch: Mapping[str, Any]) -> dict[str, int]:
    """Parameter counts derived from the config alone, without building a model.

    Lets a CPU-only test assert the pinned architecture really is Qwen3-0.6B
    (596M total / 440M non-embedding) instead of trusting the numbers were typed
    in correctly.

    Mixed token mixers with a dense MLP (``layer_types`` or
    ``full_attention_interval`` set, as Qwen3.5 has) are counted by
    :func:`hybrid_dense_parameter_count`; every dense decoder takes the branch
    below unchanged. A routed MoE is not supported.
    """
    if arch.get("num_experts"):
        raise ValueError("analytic_parameter_count does not count routed-MoE architectures")
    if arch.get("layer_types") or arch.get("full_attention_interval"):
        return hybrid_dense_parameter_count(arch)
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


def _hybrid_layer_types(arch: Mapping[str, Any]) -> list[str]:
    """``layer_types`` as written, else derived from ``full_attention_interval``
    the way Transformers derives it."""
    if arch.get("layer_types"):
        return list(arch["layer_types"])
    interval = arch["full_attention_interval"]
    return ["linear_attention" if (i + 1) % interval else "full_attention"
            for i in range(arch["num_hidden_layers"])]


def _hybrid_mixers(arch: Mapping[str, Any]) -> dict[str, int]:
    """Per-layer parameters of the two token mixers, attention and Gated DeltaNet.

    Laid out as Qwen3.5 holds them (four separate GDN input projections).
    """
    h = arch["hidden_size"]
    bias = 1 if arch.get("attention_bias") else 0
    heads, kv_heads, head_dim = (arch["num_attention_heads"], arch["num_key_value_heads"],
                                 arch["head_dim"])
    q = h * 2 * heads * head_dim + bias * 2 * heads * head_dim   # query and gate
    kv = 2 * (h * kv_heads * head_dim + bias * kv_heads * head_dim)
    o = heads * head_dim * h + bias * h
    full_mixer = q + kv + o + 2 * head_dim  # + q_norm, k_norm

    hk, hv = arch["linear_num_key_heads"], arch["linear_num_value_heads"]
    dk, dv = arch["linear_key_head_dim"], arch["linear_value_head_dim"]
    key_dim, value_dim = hk * dk, hv * dv
    conv_dim = 2 * key_dim + value_dim
    linear_mixer = (
        h * conv_dim                             # in_proj_qkv
        + h * value_dim                          # in_proj_z
        + 2 * h * hv                             # in_proj_b, in_proj_a
        + conv_dim * arch["linear_conv_kernel_dim"]  # depthwise conv, no bias
        + 2 * hv                                 # dt_bias, A_log
        + dv                                     # gated norm
        + value_dim * h                          # out_proj
    )
    return {"linear_attention": linear_mixer, "full_attention": full_mixer}


def hybrid_dense_parameter_count(arch: Mapping[str, Any]) -> dict[str, int]:
    """Parameters of a hybrid Gated-DeltaNet / gated-attention decoder with a
    dense SwiGLU MLP in every layer (Qwen3.5), from the config alone.

    ``per_layer`` maps each layer type to its count, as for the MoE variant.
    """
    h = arch["hidden_size"]
    mixers = _hybrid_mixers(arch)
    mlp = 3 * h * arch["intermediate_size"]
    norms = 2 * h
    per_type = {kind: mixer + mlp + norms for kind, mixer in mixers.items()}
    types = _hybrid_layer_types(arch)
    counts = {kind: types.count(kind) for kind in per_type}
    embedding = arch["vocab_size"] * h
    non_embedding = sum(per_type[kind] * counts[kind] for kind in per_type) + h
    lm_head = 0 if arch.get("tie_word_embeddings") else embedding
    return {
        "per_layer": per_type,
        "layers_by_type": counts,
        "embedding": embedding,
        "lm_head": lm_head,
        "non_embedding": non_embedding,
        "total": embedding + non_embedding + lm_head,
    }
