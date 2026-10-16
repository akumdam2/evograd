"""Drop-in sites inside an OLMoE decoder layer, and across the whole model.

Four sites in one block kind (``levels.level2.manifest.SITES``)::

    qkv_norm_rope       projections, full-width q/k norms, RoPE
    attention           causal SDPA and o_proj
    residual_rmsnorm    the post-attention add and its norm
    moe_experts         the routed experts (the router stays native)

**Nothing is replaced.** As for every workload here, each adapter rebinds
``forward`` on the existing module instance; parameters are the same objects,
so state dict, parameter order, dtype and ``requires_grad`` are unchanged by
construction. Three adapters, one per module that holds sites:

* ``OlmoeAttention.forward``      -- sites 1, 2;
* ``OlmoeDecoderLayer.forward``   -- the post-attention residual site: the
  input norm and the closing residual add stay native;
* ``OlmoeSparseMoeBlock.forward`` -- the experts site, with the router run
  natively first to supply ``top_k_index`` and ``top_k_weights``.

**Production spellings are pinned.** Each registry default is the exact code
the installed Transformers runs, written against the module. The experts site's
production spelling is the experts module's own call, which dispatches to the
kernel the config pins (``experts_implementation``), so a provider never
silently runs the other kernel. Patching every site with its production
spelling is ``structural identity``: bitwise the native model.

The interface (``patch_model``, ``module_patches``, ``set_tap``,
``structural_identity_kernels``, ``bound_pair_identity_kernels``,
``expected_counts``, ``PatchedModel``) has the other workloads' names and
shapes, so model-agnostic verification can bridge to this workload unchanged.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from evograd.evaluation.tier3.patch import ModulePatch, PatchProvenance, Site, SiteRegistry

from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import hf_modules
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.manifest import (
    SITE_ATTENTION,
    SITE_MOE,
    SITE_QKV,
    SITE_RESIDUAL,
    SITE_TASKS,
    SITES,
)

WORKLOAD = "olmoe_1b_7b"
OBSERVED_SUITE = "olmoe_1b_7b_observed"

#: Sites served by one installed adapter. Patching any member installs the
#: adapter, and the adapter runs every member's boundary on the same call -- the
#: unpatched ones through their production spelling.
ADAPTER_GROUPS: tuple[tuple[str, ...], ...] = (
    (SITE_QKV, SITE_ATTENTION),
    (SITE_RESIDUAL,),
    (SITE_MOE,),
)

_TAG = "_evograd_olmoe_"


def _modeling():
    return hf_modules.modeling


def live_sites(requested) -> tuple[str, ...]:
    """Every site that becomes a live boundary when ``requested`` is patched."""
    wanted = set(requested)
    live: set[str] = set()
    for group in ADAPTER_GROUPS:
        if wanted.intersection(group):
            live.update(group)
    return tuple(sorted(live))


def supporting_sites(requested) -> tuple[str, ...]:
    return tuple(sorted(set(live_sites(requested)) - set(requested)))


@dataclass
class SiteCounters:
    """How many times each site actually ran."""

    counts: dict[str, int] = field(default_factory=dict)

    def hit(self, site: str) -> None:
        self.counts[site] = self.counts.get(site, 0) + 1

    def reset(self) -> None:
        self.counts.clear()

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


# ── production spellings ─────────────────────────────────────────────────────


def production_qkv_norm_rope(module, hidden_states, cos, sin):
    """Exactly ``OlmoeAttention.forward`` up to the attention call, on ``module``."""
    if getattr(module.config, "clip_qkv", None) is not None:
        raise NotImplementedError("the OLMoE qkv_norm_rope site is declared without clip_qkv")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(module.q_proj(hidden_states))
    key_states = module.k_norm(module.k_proj(hidden_states))
    value_states = module.v_proj(hidden_states)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    query_states, key_states = _modeling().apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, value_states


def production_attention(module, query, key, value, attention_mask, input_shape, **kwargs):
    """Exactly ``OlmoeAttention.forward``'s attention interface and output projection,
    ``sliding_window`` forwarded as the module forwards it."""
    modeling = _modeling()
    attention_interface: Callable = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, modeling.eager_attention_forward)
    attn_output, _ = attention_interface(
        module, query, key, value, attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=getattr(module.config, "sliding_window", None),
        **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output)


def production_residual_rmsnorm(norm_module, branch, residual):
    summed = residual + branch
    return norm_module(summed), summed


def production_moe_experts(experts_module, x, top_k_index, top_k_weights):
    """The experts module's own call: dispatched to the kernel its config pins."""
    return experts_module(x, top_k_index, top_k_weights)


PRODUCTION: dict[str, Callable] = {
    SITE_QKV: production_qkv_norm_rope,
    SITE_ATTENTION: production_attention,
    SITE_RESIDUAL: production_residual_rmsnorm,
    SITE_MOE: production_moe_experts,
}


def _is_production(kernel, site: str) -> bool:
    return kernel is None or kernel is PRODUCTION[site]


# ── registry ─────────────────────────────────────────────────────────────────

_REGISTRY: SiteRegistry | None = None


def olmoe_sites() -> SiteRegistry:
    """The four sites. Preflight checks each task's declared correctness grid;
    the observed shapes are checked at real size by Tier 2 and the calibration
    inventory, not here."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SiteRegistry(
            name=WORKLOAD,
            sites=tuple(Site(site, SITE_TASKS[site], PRODUCTION[site]) for site in SITES),
        )
    return _REGISTRY


def structural_identity_kernels(registry: SiteRegistry | None = None):
    """Every site patched with the production spelling it already had: bitwise native."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    registry = registry or olmoe_sites()
    kernels = KernelSet(registry=registry)
    for site in registry.sites:
        kernels = patch(kernels, site.name, site.default,
                        source=KernelSource(site=site.name, op_name=site.op, module=None,
                                            origin="structural_identity"))
    return kernels


def bound_pair_identity_kernels(ops, sites: tuple[str, ...] | None = None,
                                registry: SiteRegistry | None = None):
    """Every site patched with its declared pair through ``bind`` (gated by tolerance)."""
    from evograd.evaluation.tier3.patch import identity_control_kernels

    return identity_control_kernels(ops, sites, registry=registry or olmoe_sites())


# ── the tap: live tensors for the boundary validator ─────────────────────────


def _alias(value, store: dict, key):
    """An identity view whose gradient hook reads exactly what flows through it."""
    if torch.is_tensor(value) and value.requires_grad:
        alias = value.view_as(value)
        alias.register_hook(lambda grad, _k=key: store.__setitem__(_k, grad.detach()))
        return alias
    return value


def _run_site(state, site: str, inputs: dict, activations: tuple[str, ...], call):
    """Run one site, handing its live tensors to a validator when one listens.

    With a listener, every activation input and every output is replaced by an
    identity view whose hook records the gradient crossing the boundary -- the
    upstream gradient on the way in, the emitted one on the way out.
    """
    listener = state.get("tap")
    if listener is None:
        return call(inputs)
    emitted: dict[str, torch.Tensor] = {}
    probed = dict(inputs)
    for name in activations:
        probed[name] = _alias(inputs[name], emitted, name)
    outputs = call(probed)
    single = not isinstance(outputs, tuple)
    items = (outputs,) if single else outputs
    upstream: dict[int, torch.Tensor] = {}
    aliased = tuple(_alias(t, upstream, i) for i, t in enumerate(items))
    listener(site, inputs, items, upstream, emitted, layer=state.get("layer_idx"))
    return aliased[0] if single else aliased


def _shared_positions(table: torch.Tensor) -> torch.Tensor:
    """``[B, T, D]`` rotary tables reduced to the declared ``[1, T, D]`` (exact)."""
    if table.shape[0] == 1:
        return table
    if not bool(torch.equal(table[0], table[-1])):
        raise NotImplementedError("rotary tables differ across the batch; the declared "
                                  "[1, T, D] cos/sin cannot represent them")
    return table[:1]


# ── adapters ─────────────────────────────────────────────────────────────────


def _attention_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, **kwargs):
    """``OlmoeAttention.forward`` with its two halves as sites. Same signature as
    the original, so ``position_ids`` and ``use_cache`` arrive in ``kwargs`` and
    reach the attention interface exactly as they do natively."""
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    if past_key_values is not None:
        raise NotImplementedError("the OLMoE attention adapter is a training boundary; "
                                  "cache-enabled execution is not supported")
    cos, sin = position_embeddings
    input_shape = hidden_states.shape[:-1]

    kernel = kernels.get(SITE_QKV)
    counters.hit(SITE_QKV)
    query, key, value = _run_site(
        state, SITE_QKV,
        {"x": hidden_states, "q_weight": self.q_proj.weight, "k_weight": self.k_proj.weight,
         "v_weight": self.v_proj.weight, "q_norm_weight": self.q_norm.weight,
         "k_norm_weight": self.k_norm.weight, "cos": _shared_positions(cos),
         "sin": _shared_positions(sin), "eps": float(self.q_norm.variance_epsilon)},
        ("x",),
        lambda i: (production_qkv_norm_rope(self, i["x"], cos, sin)
                   if _is_production(kernel, SITE_QKV)
                   else kernel(i["x"], i["q_weight"], i["k_weight"], i["v_weight"],
                               i["q_norm_weight"], i["k_norm_weight"], i["cos"], i["sin"],
                               i["eps"])))

    kernel = kernels.get(SITE_ATTENTION)
    counters.hit(SITE_ATTENTION)
    if not _is_production(kernel, SITE_ATTENTION) and attention_mask is not None:
        raise NotImplementedError("olmoe_attention is declared for the mask-free causal "
                                  "branch; this call supplied an attention mask")
    out = _run_site(
        state, SITE_ATTENTION,
        {"q": query, "k": key, "v": value, "o_weight": self.o_proj.weight},
        ("q", "k", "v"),
        lambda i: (production_attention(self, i["q"], i["k"], i["v"], attention_mask,
                                        input_shape, **kwargs)
                   if _is_production(kernel, SITE_ATTENTION)
                   else kernel(i["q"], i["k"], i["v"], i["o_weight"])))
    return out, None


def _moe_forward(self, hidden_states):
    """``OlmoeSparseMoeBlock.forward`` with the experts as a site; the router native."""
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    _, top_k_weights, top_k_index = self.gate(flat)
    experts = self.experts
    kernel = kernels.get(SITE_MOE)
    counters.hit(SITE_MOE)
    out = _run_site(
        state, SITE_MOE,
        {"x": flat, "gate_up_proj": experts.gate_up_proj, "down_proj": experts.down_proj,
         "top_k_index": top_k_index, "top_k_weights": top_k_weights},
        ("x", "top_k_weights"),
        lambda i: (production_moe_experts(experts, i["x"], i["top_k_index"], i["top_k_weights"])
                   if _is_production(kernel, SITE_MOE)
                   else kernel(i["x"], i["gate_up_proj"], i["down_proj"], i["top_k_index"],
                               i["top_k_weights"])))
    return out.reshape(batch_size, sequence_length, hidden_dim)


def _layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
    """``OlmoeDecoderLayer.forward`` with the post-attention add + norm as a site.

    Same operations in the same order: input norm native, attention through
    its own (possibly adapted) forward, the residual site, the MoE block, and
    the closing ``residual + branch`` as the original writes it.
    """
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    if past_key_values is not None:
        raise NotImplementedError("the OLMoE decoder adapter is a training boundary")
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    branch, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                               position_ids=position_ids, past_key_values=past_key_values,
                               use_cache=use_cache, position_embeddings=position_embeddings,
                               **kwargs)
    kernel = kernels.get(SITE_RESIDUAL)
    counters.hit(SITE_RESIDUAL)
    norm = self.post_attention_layernorm
    normed, summed = _run_site(
        state, SITE_RESIDUAL,
        {"branch": branch, "residual": residual, "weight": norm.weight,
         "eps": float(norm.variance_epsilon)},
        ("branch", "residual"),
        lambda i: (production_residual_rmsnorm(norm, i["branch"], i["residual"])
                   if _is_production(kernel, SITE_RESIDUAL)
                   else kernel(i["branch"], i["residual"], i["weight"], i["eps"])))
    return summed + self.mlp(normed)


def _install(module, forward, kernels: dict, counters: SiteCounters, layer_idx) -> None:
    state = getattr(module, _TAG + "state", None)
    if state is None:
        state = {}
        setattr(module, _TAG + "state", state)
        module.forward = types.MethodType(forward, module)
    state.update({"kernels": dict(kernels), "counters": counters, "layer_idx": layer_idx})


_ATTN_SITES = (SITE_QKV, SITE_ATTENTION)


def install_sites(layer, kernels: dict, counters: SiteCounters,
                  layer_idx: int | None = None) -> dict[str, tuple[str, ...]]:
    """Install every requested site's adapter on one decoder layer; return local paths."""
    paths: dict[str, tuple[str, ...]] = {}
    requested = set(kernels)
    if requested & set(_ATTN_SITES):
        _install(layer.self_attn, _attention_forward, kernels, counters, layer_idx)
        for site in requested & set(_ATTN_SITES):
            paths[site] = ("self_attn",)
    if SITE_RESIDUAL in requested:
        _install(layer, _layer_forward, kernels, counters, layer_idx)
        paths[SITE_RESIDUAL] = ("post_attention_layernorm",)
    if SITE_MOE in requested:
        _install(layer.mlp, _moe_forward, kernels, counters, layer_idx)
        paths[SITE_MOE] = ("mlp.experts",)
    return paths


# ── the whole model ──────────────────────────────────────────────────────────


def expected_counts(num_layers: int) -> dict[str, int]:
    """What one forward of a model with this depth invokes, per site: once per layer."""
    return {site: int(num_layers) for site in SITES}


def patch_model(model, kernels, counters: SiteCounters | None = None,
                *, expected_layers: int | None = None):
    """Install every requested site on a built ``OlmoeForCausalLM``.

    Returns ``(provenance, counters, carrier)``; ``carrier`` is always ``None``
    (no cross-layer fusion in this step), kept for the other workloads' shape.
    A requested site that reaches fewer modules than the depth says raises: a
    provider labelled "patched" that is partly native would report native
    numbers as the kernel's.
    """
    counters = counters or SiteCounters()
    requested = tuple(kernels.patched)
    selected = {site: kernels.kernel_for(site) for site in requested}
    declared = (expected_layers if expected_layers is not None
                else getattr(model.config, "num_hidden_layers", None))
    layers = model.model.layers
    if declared is not None and len(layers) != declared:
        raise ValueError(f"the model has {len(layers)} decoder layers but its configuration "
                         f"declares {declared}; a partially patched model would report the "
                         "unpatched path's numbers as the kernel's")
    paths: dict[str, list[str]] = {}
    for index, layer in enumerate(layers):
        for site, local in install_sites(layer, selected, counters, index).items():
            paths.setdefault(site, []).extend(f"model.layers.{index}.{p}" for p in local)
    want = expected_counts(len(layers))
    for site in requested:
        found = len(paths.get(site, ()))
        if found != want[site]:
            raise ValueError(f"patch site {site} reached {found} modules, the model has "
                             f"{want[site]} layers")
    provenance = PatchProvenance(
        method="module_surgery", requested_sites=tuple(sorted(requested)),
        actual_sites=tuple(sorted(paths)), paths={k: tuple(v) for k, v in paths.items()})
    return provenance, counters, None


def module_patches() -> tuple[ModulePatch, ...]:
    """The declarative view of what this workload rebinds, for a report."""
    def _named(kind: str):
        return lambda module: type(module).__name__ == kind

    return (
        ModulePatch(SITE_QKV, _named("OlmoeAttention"), lambda o, k: o,
                    sites=(SITE_ATTENTION,)),
        ModulePatch(SITE_RESIDUAL, _named("OlmoeDecoderLayer"), lambda o, k: o),
        ModulePatch(SITE_MOE, _named("OlmoeSparseMoeBlock"), lambda o, k: o),
    )


def set_tap(model, listener) -> None:
    """Attach (or clear, with ``None``) a boundary listener on every adapter."""
    for module in model.modules():
        state = getattr(module, _TAG + "state", None)
        if isinstance(state, dict):
            state["tap"] = listener


@dataclass(frozen=True)
class PatchedModel:
    """A built, patched model plus everything the report needs to describe it."""

    model: Any
    provenance: PatchProvenance
    counters: SiteCounters
    carrier: Any
    expected: dict[str, int]

    def observed(self) -> dict[str, int]:
        return self.counters.snapshot()

    def count_problems(self) -> list[str]:
        """Live sites whose observed invocation count is not the declared one."""
        observed = self.observed()
        live = live_sites(self.provenance.actual_sites)
        return [f"{site}: expected {self.expected.get(site)} invocations, observed "
                f"{observed.get(site, 0)}"
                for site in live if observed.get(site, 0) != self.expected.get(site)]


__all__ = [
    "ADAPTER_GROUPS", "OBSERVED_SUITE", "PRODUCTION", "PatchedModel", "SiteCounters",
    "bound_pair_identity_kernels", "expected_counts", "install_sites", "live_sites",
    "module_patches", "olmoe_sites", "patch_model", "set_tap", "structural_identity_kernels",
    "supporting_sites",
]
