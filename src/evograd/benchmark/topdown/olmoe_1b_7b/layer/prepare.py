"""Building one OLMoE decoder layer the way the full model holds it.

Two sources feed a decoder layer: a **captured** layer (weights, arguments,
upstream gradient and reference gradients from the Level-3 step, see
:mod:`.capture`) and a **config-derived** one, built here from the published
architecture. Either way the layer is a bare ``OlmoeDecoderLayer``, and a bare
layer differs from the same layer inside ``OlmoeForCausalLM`` in one way this
module undoes (checked on Transformers 5.16.1):

**T-init -- no ``post_init``.** A standalone layer never runs
``OlmoePreTrainedModel._init_weights``: its linears keep PyTorch's default
(kaiming-uniform) initialization and the router and the 3-D expert tensors keep
their constructor values (zeros and ``torch.empty``). :func:`initial_weights`
applies the model's own rules explicitly -- on the CPU, from a fixed seed -- and
:func:`check_initialization` asserts the result.

The experts kernel is a config setting, not an import-time dispatch: a bare
``OlmoeExperts`` reads ``config._experts_implementation`` on every call, so
:func:`build_config` pins it (the canonical grouped GEMM unless told otherwise)
and :func:`effective_settings` reads it back.

Also pinned: ``sdpa``, ``use_cache=False``, train mode, no checkpointing.
Weights are drawn once per (seed, layer) on the CPU and loaded into however many
builds a measurement needs; a build never draws a random number.

The replay half of the dense packages' contract (``prepare_layer``,
``verify_layout``, ``live_model_instances``, ``validate_noise_repeats``) is here
too, because the shared layer replay reads it off this module.
"""

from __future__ import annotations

import copy
import gc
import math
from typing import Any, Mapping

import torch

from ..architecture import CANONICAL_EXPERTS, OLMOE_1B_7B, layer_kind

#: The Transformers release these layers were written and checked against.
TESTED_TRANSFORMERS = "5.16.1"
ATTN_IMPLEMENTATION = "sdpa"

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

#: The pinned run settings every layer case is held to. The experts kernel is
#: held separately, to the spec's value.
PINNED_SETTINGS: Mapping[str, Any] = {
    "attn_implementation": ATTN_IMPLEMENTATION,
    "use_cache": False,
    "gradient_checkpointing": False,
    "training": True,
}

#: The initialization recipe, as text for a case's source: one line per rule of
#: ``OlmoePreTrainedModel._init_weights`` and the base rules it extends.
INIT_RULES = (
    "nn.Linear weight ~ normal(0, initializer_range)",
    "OlmoeRMSNorm.weight = 1",
    "OlmoeTopKRouter.weight ~ normal(0, initializer_range)",
    "OlmoeExperts.gate_up_proj, OlmoeExperts.down_proj ~ normal(0, initializer_range)",
    "each parameter drawn in float32 from one CPU torch.Generator, in "
    "named_parameters() order, then cast to the case dtype",
)


class PrepareError(RuntimeError):
    """The layer could not be built, or was not built the way the model holds it."""


class ReplayError(RuntimeError):
    """The replay could not be performed, or disagreed with the capture."""


def require_transformers():
    """The installed OLMoE modeling module, or a clear refusal."""
    try:
        from transformers.models.olmoe import modeling_olmoe
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PrepareError(
            "OLMoE layers need transformers >= 5.0 with olmoe "
            f"(tested {TESTED_TRANSFORMERS}): {exc}"
        ) from exc
    return modeling_olmoe


def build_config(arch: Mapping[str, Any] = OLMOE_1B_7B, *,
                 experts_implementation: str = CANONICAL_EXPERTS):
    """The configuration with every run setting pinned.

    Deep-copied first: Transformers normalizes the RoPE fields in place, which
    would otherwise rewrite the caller's frozen mapping.
    """
    require_transformers()
    from transformers import OlmoeConfig

    return OlmoeConfig(**copy.deepcopy(dict(arch)), use_cache=False,
                       attn_implementation=ATTN_IMPLEMENTATION,
                       experts_implementation=experts_implementation)


def _meta_layer(config, layer_index: int):
    modeling = require_transformers()
    with torch.device("meta"):
        return modeling.OlmoeDecoderLayer(config, layer_index)


def weight_seed(seed: int, layer_index: int) -> int:
    """One generator per (recipe seed, layer)."""
    return int(seed) * 1000 + int(layer_index)


def _rule_for(layer, name: str) -> str:
    """Which initialization rule owns one parameter, by the owning module's type."""
    modeling = require_transformers()
    owner_path, _, leaf = name.rpartition(".")
    owner = layer.get_submodule(owner_path) if owner_path else layer
    if isinstance(owner, torch.nn.Linear) and leaf == "weight":
        return "normal"
    if isinstance(owner, modeling.OlmoeRMSNorm) and leaf == "weight":
        return "ones"
    if isinstance(owner, modeling.OlmoeTopKRouter) and leaf == "weight":
        return "normal"
    if isinstance(owner, modeling.OlmoeExperts) and leaf in ("gate_up_proj", "down_proj"):
        return "normal"
    raise PrepareError(
        f"no initialization rule for {name} ({type(owner).__name__}); refusing to "
        "leave a parameter at its constructor value"
    )


def initial_weights(arch: Mapping[str, Any], layer_index: int, *, dtype: str,
                    seed: int = 0) -> dict[str, torch.Tensor]:
    """The layer's parameters under the model's own initialization rules (T-init)."""
    config = build_config(arch)
    layer = _meta_layer(config, layer_index)
    generator = torch.Generator().manual_seed(weight_seed(seed, layer_index))
    std = float(config.initializer_range)
    torch_dtype = DTYPES[dtype]
    weights: dict[str, torch.Tensor] = {}
    for name, param in layer.named_parameters():
        rule = _rule_for(layer, name)
        if rule == "normal":
            value = torch.empty(param.shape, dtype=torch.float32).normal_(
                0.0, std, generator=generator)
        else:  # ones
            value = torch.ones(param.shape, dtype=torch.float32)
        weights[name] = value.to(torch_dtype)
        del value
    problems = check_initialization(weights, std=std)
    if problems:
        raise PrepareError("initialization did not produce the model's layer: "
                           + "; ".join(problems))
    return weights


def check_initialization(weights: Mapping[str, torch.Tensor], *, std: float = 0.02) -> list[str]:
    """Why these weights are not what ``post_init`` would have produced, if not.

    Each T-init symptom gets a named check: projections, the router and the
    expert tensors not at the initializer's scale (PyTorch's default
    kaiming-uniform is ~1/sqrt(fan_in), not 0.02; a constructor ``torch.empty``
    is anything), and norm weights off one.
    """
    problems: list[str] = []
    for name, value in weights.items():
        if not bool(torch.isfinite(value).all()):
            problems.append(f"{name} is not finite")
    for name, value in weights.items():
        if value.ndim >= 2:
            measured = float(value.float().std())
            tolerance = max(0.05, 4.0 / math.sqrt(2.0 * value.numel()))
            if not math.isclose(measured, std, rel_tol=tolerance):
                problems.append(f"{name} std {measured:.4g} is not the initializer's {std}")
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                          "q_norm.weight", "k_norm.weight")) and not bool((value == 1).all()):
            problems.append(f"{name} is not one: OlmoeRMSNorm starts at w = 1")
    return problems


def build_single_layer(arch: Mapping[str, Any], layer_index: int, *, device: str,
                       dtype: str, weights: Mapping[str, torch.Tensor] | None = None,
                       experts_implementation: str = CANONICAL_EXPERTS):
    """One ``OlmoeDecoderLayer`` on ``device``; nothing above it is constructed.

    With ``weights`` it is built on the meta device and given copies of them
    (never the caller's own tensors). Without, it is constructed normally --
    for a caller that loads a state dict next, as the shared replay does.
    """
    config = build_config(arch, experts_implementation=experts_implementation)
    torch_dtype = DTYPES[dtype]
    if weights is None:
        modeling = require_transformers()
        layer = modeling.OlmoeDecoderLayer(config, layer_index)
        layer = layer.to(device=device, dtype=torch_dtype)
    else:
        layer = _meta_layer(config, layer_index).to(dtype=torch_dtype)
        layer.load_state_dict(
            {k: v.to(device=device, dtype=torch_dtype, copy=True) for k, v in weights.items()},
            strict=True, assign=True,
        )
        meta = [n for n, p in layer.named_parameters() if p.is_meta]
        if meta:  # pragma: no cover - load_state_dict(strict=True) prevents it
            raise PrepareError(f"parameters left on the meta device: {meta[:4]}")
    if list(layer.named_buffers()):  # pragma: no cover - version dependent
        raise PrepareError("this OlmoeDecoderLayer owns buffers; the build does not know "
                           "how to place them")
    layer.train()
    layer.gradient_checkpointing = False
    return layer


def effective_settings(layer) -> dict[str, Any]:
    """What the built layer will actually execute, read back rather than assumed."""
    import transformers

    config = layer.self_attn.config
    experts_config = getattr(layer.mlp.experts, "config", None)
    return {
        "block_type": "decoder_layer",
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "experts_implementation": getattr(experts_config, "_experts_implementation", None),
        "use_cache": bool(getattr(config, "use_cache", True)),
        "gradient_checkpointing": bool(getattr(layer, "gradient_checkpointing", True)),
        "training": bool(layer.training),
        "parameter_dtypes": sorted({str(p.dtype).removeprefix("torch.")
                                    for p in layer.parameters()}),
        "transformers": transformers.__version__,
    }


def check_effective_settings(effective: Mapping[str, Any], *, dtype: str,
                             experts_implementation: str = CANONICAL_EXPERTS) -> list[str]:
    problems = [
        f"{key} is {effective.get(key)!r}, pinned {want!r}"
        for key, want in PINNED_SETTINGS.items()
        if effective.get(key) != want
    ]
    if effective.get("experts_implementation") != experts_implementation:
        problems.append(f"experts_implementation is {effective.get('experts_implementation')!r}, "
                        f"pinned {experts_implementation!r}")
    if effective.get("parameter_dtypes") != [dtype]:
        problems.append(f"parameters are {effective.get('parameter_dtypes')}, not [{dtype!r}]")
    return problems


#: The config recipe's cotangent scale, the choice every layer workload makes: a
#: residual-stream upstream gradient small relative to the activations.
COTANGENT_SCALE = "1/sqrt(H)"


def rotary_tables(arch: Mapping[str, Any], *, seq: int, dtype: str, batch: int = 1):
    """The cos/sin the full model hands every layer for ``position_ids = arange(T)``.

    ``OlmoeRotaryEmbedding`` at theta 10,000 over the full 128-wide head. One
    row (``[1, T, D]``) by default -- what every row of a larger batch's table
    equals; the layers broadcast it.
    """
    modeling = require_transformers()
    config = build_config(arch)
    rotary = modeling.OlmoeRotaryEmbedding(config=config)
    probe = torch.zeros(batch, seq, 1, dtype=DTYPES[dtype])
    position_ids = torch.arange(seq, dtype=torch.int64).unsqueeze(0).expand(batch, -1)
    with torch.no_grad():
        cos, sin = rotary(probe, position_ids)
    return cos.detach().contiguous(), sin.detach().contiguous()


def block_inputs(arch: Mapping[str, Any], layer_index: int, *, batch: int, seq: int,
                 dtype: str, seed: int = 0) -> dict[str, Any]:
    """The layer's call arguments and output cotangent, drawn on the CPU.

    ``hidden_states`` and the cotangent come from one generator seeded with
    ``seed``. Every layer receives the rotary tables, as the full model's loop
    passes them.
    """
    layer_kind(layer_index, arch)  # refuses an index outside the architecture
    torch_dtype = DTYPES[dtype]
    hidden_size = int(arch["hidden_size"])
    generator = torch.Generator().manual_seed(int(seed))
    hidden = torch.randn(batch, seq, hidden_size, generator=generator).to(torch_dtype)
    cotangent = (torch.randn(batch, seq, hidden_size, generator=generator)
                 / math.sqrt(hidden_size)).to(torch_dtype)
    return {"hidden_states": hidden, "cotangent": cotangent,
            "position_embeddings": rotary_tables(arch, seq=seq, dtype=dtype)}


def recipe(arch: Mapping[str, Any], layer_index: int, *, seed: int) -> dict[str, Any]:
    """The config-derived recipe, as recorded in a case's source."""
    return {
        "recipe": "config-derived",
        "seed": int(seed),
        "weights": {"kind": "explicit_post_init_rules", "rules": list(INIT_RULES),
                    "generator_seed": weight_seed(seed, layer_index), "device": "cpu"},
        "inputs": {
            "hidden_states": "randn(B, T, H) from torch.Generator(seed), cast to the dtype",
            "position_embeddings": ("OlmoeRotaryEmbedding(config)(., arange(T)[None]) -- theta "
                                    "10000 over the whole 128-wide head, one [1, T, D] row "
                                    "broadcast over the batch"),
            "attention_mask": None,
        },
        "cotangent": f"randn(B, T, H) * {COTANGENT_SCALE} from the same generator",
        "settings": {**PINNED_SETTINGS, "experts_implementation": CANONICAL_EXPERTS},
        "claim": "layer-kernel correctness and performance; not observed activation "
                 "distributions and not real weights",
    }


# ── the replay half (read by evaluation.workloads.common.replay) ─────────────


class AuxiliaryObjective:
    """The load-balancing loss's gradient path into one layer, rebuilt standalone.

    With ``output_router_logits`` on, ``OlmoeForCausalLM`` adds
    ``coef * load_balancing_loss_func(all layers' router logits)`` to the loss.
    That objective reaches this layer's router (and, through the router's
    input, the post-attention norm and everything before it) from outside the
    layer's output, so a replay driven by the output cotangent alone cannot
    reproduce their gradients. The loss function sums per-expert statistics over
    every layer and divides by the total row count, so its gradient into one
    layer's logits is ``coef * E * tokens_per_expert[e] / total_rows`` through
    that layer's own softmax -- the other layers enter only as constants. This
    rebuilds exactly that chain (same ops, same order, same dtypes) on the
    router logits the standalone forward produces, and the replay backpropagates
    it together with the output cotangent.
    """

    def __init__(self, layer, payload: dict[str, Any]):
        self.aux = payload.get("auxiliary")
        self.logits: torch.Tensor | None = None
        self._handle = None
        self._gate = layer.mlp.gate

    def __enter__(self) -> "AuxiliaryObjective":
        if self.aux is not None:
            def hook(module, args, output):
                self.logits = output[0]
            self._handle = self._gate.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def loss(self) -> torch.Tensor | None:
        """``coef * (sum_e tokens_per_expert[e] * prob_sum[e] / total_rows) * E``, as the
        model computes it, with the other layers' probability sums as constants."""
        if self.aux is None:
            return None
        if self.logits is None:
            raise ReplayError("the replayed layer's router did not run; the auxiliary "
                              "objective has no logits")
        device = self.logits.device
        routing_weights = torch.nn.functional.softmax(self.logits, dim=-1)
        other = self.aux["router_prob_sum_other_layers"].to(device)
        router_prob_sum = other + routing_weights.float().sum(dim=0)
        total_rows = float(self.aux["total_rows"])
        tokens_per_expert = self.aux["tokens_per_expert"].to(device)
        router_prob_per_expert = router_prob_sum / total_rows
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        aux_loss = overall_loss * int(self.aux["num_experts"])
        return float(self.aux["coef"]) * aux_loss


def auxiliary_objective(layer, payload: dict[str, Any]) -> AuxiliaryObjective:
    """What the shared replay drives beside the output cotangent (``None`` loss
    for an artifact captured without the auxiliary loss)."""
    return AuxiliaryObjective(layer, payload)



def validate_noise_repeats(value: Any) -> int:
    """0 to skip the measurement, or at least 2 to make one."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"noise_repeats must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"noise_repeats must not be negative, got {value}")
    if value == 1:
        raise ValueError("noise_repeats=1 cannot measure anything: pass 0 to skip, or >= 2")
    return value


def prepare_layer(payload: dict[str, Any], device: str):
    """Rebuild a captured layer and its arguments on ``device``, dispatching the
    experts the way the captured step did."""
    from .artifact import to_device

    identity = payload["identity"]
    dtype = str(payload["output"].dtype).replace("torch.", "")
    layer = build_single_layer(
        payload["arch"], identity["layer_index"], device=device, dtype=dtype,
        weights=payload["state_dict"],
        experts_implementation=identity.get("experts_implementation", CANONICAL_EXPERTS))
    args = to_device(payload["args"], device)
    kwargs = to_device(payload["kwargs"], device)
    grad_output = to_device(payload["grad_output"], device)
    return layer, args, kwargs, grad_output, dtype


def verify_layout(cpu_value: Any, device_value: Any, path: str = "$") -> list[str]:
    """Layout and dtype must survive the move to the device."""
    problems: list[str] = []
    if torch.is_tensor(cpu_value):
        if not torch.is_tensor(device_value):  # pragma: no cover - defensive
            return [f"{path}: expected a tensor"]
        if cpu_value.dtype != device_value.dtype:
            problems.append(f"{path}: dtype {device_value.dtype} != captured {cpu_value.dtype}")
        if tuple(cpu_value.shape) != tuple(device_value.shape):
            problems.append(f"{path}: shape {tuple(device_value.shape)} != {tuple(cpu_value.shape)}")
        if tuple(cpu_value.stride()) != tuple(device_value.stride()):
            problems.append(f"{path}: stride {tuple(device_value.stride())} != captured "
                            f"{tuple(cpu_value.stride())}")
        return problems
    if isinstance(cpu_value, (tuple, list)):
        for index, item in enumerate(cpu_value):
            problems += verify_layout(item, device_value[index], f"{path}[{index}]")
        return problems
    if isinstance(cpu_value, dict):
        for key, item in cpu_value.items():
            problems += verify_layout(item, device_value[key], f"{path}.{key}")
        return problems
    if cpu_value != device_value:  # pragma: no cover - scalars pass through
        problems.append(f"{path}: {device_value!r} != captured {cpu_value!r}")
    return problems


def live_model_instances() -> dict[str, int]:
    """Count live OLMoE objects on the heap: the standalone claim's evidence."""
    modeling = require_transformers()
    classes = {
        name: getattr(modeling, name)
        for name in ("OlmoeForCausalLM", "OlmoeModel", "OlmoeDecoderLayer",
                     "OlmoeAttention", "OlmoeSparseMoeBlock")
        if hasattr(modeling, name)
    }
    counts = {name: 0 for name in classes}
    gc.collect()
    for obj in gc.get_objects():
        try:
            for name, cls in classes.items():
                if type(obj) is cls:
                    counts[name] += 1
                    break
        except ReferenceError:  # pragma: no cover - weakref proxies
            continue
    return counts


__all__ = [
    "ATTN_IMPLEMENTATION", "COTANGENT_SCALE", "DTYPES", "INIT_RULES", "PINNED_SETTINGS",
    "PrepareError", "ReplayError", "TESTED_TRANSFORMERS", "block_inputs", "build_config",
    "build_single_layer", "check_effective_settings", "check_initialization",
    "effective_settings", "initial_weights", "live_model_instances", "prepare_layer",
    "recipe", "require_transformers", "rotary_tables", "validate_noise_repeats",
    "verify_layout", "weight_seed", "AuxiliaryObjective", "auxiliary_objective",
]
