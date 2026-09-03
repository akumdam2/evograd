"""The Llama-3 training workload: one model the tier-3 harness can measure.

Nothing here is required by the tier. ``bench.tier3_runner`` measures whatever
:class:`~evograd.bench.tier3_model.TrainingWorkload` it is handed; this module is the
first such workload, and it exists because the repository already contains a
Llama-3 decoder layer whose normalization and activation are injected rather
than hardcoded, which makes it patchable without module surgery.

A model you did not write — a HuggingFace ``LlamaForCausalLM``, someone else's
research model — takes the other route: ``ModuleWorkload`` plus
``ModulePatch``, which replaces matching submodules in a built tree. Both end
up handing the harness the same thing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from evograd.bench.tier3_patch import LLAMA_SITES, KernelSet
from evograd.opdecl.models import ModelConfig
from evograd.ops.level3.llama3_decoder_layer.forward_ref import _llama3_decoder_layer




def rope_tables(config: ModelConfig, tokens: int, *, device, dtype):
    """``cos``/``sin`` with duplicated halves, as the layer reference expects."""
    half = torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)
    inverse = 1.0 / (config.rope_theta ** (half / config.head_dim))
    position = torch.arange(tokens, device=device, dtype=torch.float32)
    angle = torch.outer(position, inverse)
    full = torch.cat((angle, angle), dim=-1)
    return full.cos().to(dtype), full.sin().to(dtype)


class DecoderLayer(nn.Module):
    """One Llama-3 decoder layer, holding its own weights.

    The body is the declaration's own ``_llama3_decoder_layer``: attention with
    grouped-query heads and RoPE, then a SwiGLU MLP, both residual. Only the
    normalization and the activation are reached for through the
    :class:`KernelSet`, because those are the sites a kernel replaces.
    """

    def __init__(self, config: ModelConfig, kernels: KernelSet, *, eps: float = 1e-5):
        super().__init__()
        self.kernels = kernels
        self.eps = eps
        hidden, inter = config.hidden, config.intermediate

        def weight(out_features: int, in_features: int) -> nn.Parameter:
            # nn.Linear orientation [out, in], matching the reference so these
            # drop into a HuggingFace or Liger-patched layer untransposed.
            return nn.Parameter(torch.empty(out_features, in_features))

        self.input_norm_weight = nn.Parameter(torch.empty(hidden))
        self.q_weight = weight(config.q_out, hidden)
        self.k_weight = weight(config.kv_out, hidden)
        self.v_weight = weight(config.kv_out, hidden)
        self.o_weight = weight(hidden, config.q_out)
        self.post_norm_weight = nn.Parameter(torch.empty(hidden))
        self.gate_weight = weight(inter, hidden)
        self.up_weight = weight(inter, hidden)
        self.down_weight = weight(hidden, inter)

    def forward(self, x, cos, sin):
        return _llama3_decoder_layer(
            self.kernels.rms_norm,
            x,
            self.input_norm_weight,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
            self.post_norm_weight,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            cos,
            sin,
            self.eps,
            self.kernels.swiglu,
        )


class TinyLlama(nn.Module):
    """Embedding, N decoder layers, final norm, and a language-model head.

    Small enough to iterate on and architecturally identical to the real thing:
    every width is Llama-3-8B's, only ``layers`` is reduced. That is the one
    dimension that can be cut without changing what a kernel does to the answer,
    because per-layer effects scale linearly in it and the loss head's memory —
    a ``[tokens, vocab]`` logits tensor — does not depend on it at all.
    """

    def __init__(self, config: ModelConfig, kernels: KernelSet, *, eps: float = 1e-5):
        super().__init__()
        self.config = config
        self.kernels = kernels
        self.eps = eps
        self.embedding = nn.Parameter(torch.empty(config.vocab, config.hidden))
        self.layers = nn.ModuleList(
            DecoderLayer(config, kernels, eps=eps) for _ in range(config.layers)
        )
        self.final_norm_weight = nn.Parameter(torch.empty(config.hidden))
        self.lm_head = nn.Parameter(torch.empty(config.vocab, config.hidden))

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = F.embedding(tokens, self.embedding)
        cos, sin = self._tables(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.kernels.rms_norm(x, self.final_norm_weight, self.eps)
        return self.kernels.cross_entropy(x, self.lm_head, targets)

    def _tables(self, x):
        # Built once per shape and cached on the module: a training loop does
        # not recompute them per step, and putting that work inside the timed
        # region would charge every provider for something none of them do.
        key = (x.shape[1], x.dtype, x.device)
        cached = getattr(self, "_rope_cache", None)
        if cached is None or cached[0] != key:
            tables = rope_tables(
                self.config, x.shape[1], device=x.device, dtype=x.dtype
            )
            self._rope_cache = (key, tables)
        return self._rope_cache[1]


def build_model(
    config: ModelConfig,
    kernels: KernelSet,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> TinyLlama:
    """A model with identical weights for every provider.

    Seeded and initialized here rather than left to each module's own default:
    two providers initialized independently would be compared on different
    weights, which changes both the loss trajectory and — through the data —
    the kernels' numerics.
    """
    torch.manual_seed(seed)
    model = TinyLlama(config, kernels).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.fill_(1.0)  # norm scales
            else:
                parameter.normal_(0.0, 0.02)
    return model


def make_batch(
    config: ModelConfig, *, batch: int, tokens: int, device: str = "cuda", seed: int = 0
):
    """Synthetic token ids and next-token targets, identical across providers."""
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(
        0, config.vocab, (batch, tokens), device=device, generator=generator
    )
    targets = torch.randint(
        0, config.vocab, (batch, tokens), device=device, generator=generator
    )
    return ids, targets




class LlamaWorkload:
    """Llama-3 next-token training, as a workload the harness can drive.

    Implements the :class:`~evograd.bench.tier3.TrainingWorkload` protocol.
    Patching happens by construction rather than by surgery: the model holds a
    :class:`KernelSet` and calls through it, so building with a different set
    is the whole of "patched".
    """

    unit_name = "tokens"
    #: This model's three sites. Declared here rather than read from a module
    #: global, so a second architecture adds a registry instead of editing one.
    site_registry = LLAMA_SITES

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch: int,
        tokens: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
    ):
        self.config = config
        self.batch = batch
        self.tokens = tokens
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.name = f"{config.name}[batch={batch},tokens={tokens}]"

    def units_per_step(self) -> int:
        return self.batch * self.tokens

    def build(self, kernels: KernelSet) -> nn.Module:
        return build_model(
            self.config, kernels, device=self.device, dtype=self.dtype, seed=self.seed
        )

    def batch_for(self, *, seed: int) -> Any:
        return make_batch(
            self.config, batch=self.batch, tokens=self.tokens,
            device=self.device, seed=seed,
        )

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        ids, targets = batch
        return model(ids, targets)

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "llama_next_token",
            "model": self.config.name,
            "layers": self.config.layers,
            "batch": self.batch,
            "tokens": self.tokens,
            "dtype": str(self.dtype).removeprefix("torch."),
        }


# ── tier-3 CLI adapter ───────────────────────────────────────────────────────


def _build_for(model_config):
    """A ``(args) -> LlamaWorkload`` bound to one ``ModelConfig``.

    Llama's batch and token defaults live here rather than on the parser: they
    are this workload's answer to "how big is a step", and a parser default
    would be one workload's number applied to every other.
    """

    def build(args) -> "LlamaWorkload":
        return LlamaWorkload(
            model_config,
            batch=args.batch if args.batch is not None else 4,
            tokens=args.tokens if args.tokens is not None else 1024,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
        )

    return build


def _adapter(name: str, model_config, summary: str):
    from evograd.bench.workloads import Tier3Adapter

    return Tier3Adapter(
        name=name,
        build=_build_for(model_config),
        # No extra providers: this workload's kernels go in by construction, so
        # `eager`, the identity control and any pair baseline already cover it.
        providers=None,
        # Neither `--layers` nor `--data-seed` means anything here: the layer
        # count is the ModelConfig's, and the batch stream is drawn from the
        # workload seed rather than a separate one.
        options=frozenset(),
        summary=summary,
    )


def __getattr__(name: str):
    # Built on demand so importing this module does not import the model
    # registry, and so `bench.workloads` can name these without a cycle.
    from evograd.opdecl import models as model_registry

    if name == "ADAPTER":
        return _adapter("llama_3_8b", model_registry.LLAMA_3_8B,
                        "Llama-3-8B, 32 layers, patched by construction")
    if name == "ADAPTER_4L":
        return _adapter("llama_3_8b_4l", model_registry.LLAMA_3_8B_4L,
                        "Llama-3-8B shrunk to 4 layers; the one to iterate on")
    raise AttributeError(name)
