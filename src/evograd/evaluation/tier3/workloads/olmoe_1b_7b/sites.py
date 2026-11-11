"""Drop-in sites inside an OLMoE decoder layer, and across the whole model.

Three sites in one block kind (``levels.level2.manifest.SITES``)::

    attention_block     OlmoeAttention.forward whole: projections, full-width q/k
                        norms, RoPE, causal SDPA, o_proj
    residual_rmsnorm    a residual add and the RMSNorm that follows it (2 per layer)
    moe_experts         the routed experts (the router stays native)

**Nothing is replaced.** As for every workload here, each adapter rebinds
``forward`` on the existing module instance; parameters are the same objects,
so state dict, parameter order, dtype and ``requires_grad`` are unchanged by
construction. One adapter per site:

* ``OlmoeAttention.forward``      -- the attention block;
* ``OlmoeDecoderLayer.forward`` and ``model.norm`` -- the residual site. Its
  calls span modules: each layer hands its closing ``residual + moe`` add,
  un-added, to the next layer's ``input_layernorm`` (or, after the last layer,
  to ``model.norm``) through Qwen3's :class:`ResidualCarrier`, so every residual
  add an RMSNorm follows is one call -- the post-attention one inside the layer
  and the cross-layer one, ``2 * layers`` in all. Layer 0's input norm follows
  the embedding and stays native;
* ``OlmoeSparseMoeBlock.forward`` -- the experts site, with the router run
  natively first to supply ``top_k_index`` and ``top_k_weights``.

**Production spellings are pinned.** Each registry default is the exact code
the installed Transformers runs, on the module: the attention module's own
class ``forward``, ``residual + branch`` followed by the norm module, and the
experts module's own call, which dispatches to the kernel the config pins
(``experts_implementation``), so a provider never silently runs the other
kernel. Patching every site with its production spelling is ``structural
identity``: bitwise the native model.

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
from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import ResidualCarrier

from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import hf_modules
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.manifest import (
    SITE_ATTENTION_BLOCK,
    SITE_MOE,
    SITE_RESIDUAL,
    SITE_TASKS,
    SITES,
)

WORKLOAD = "olmoe_1b_7b"
OBSERVED_SUITE = "olmoe_1b_7b_observed"

_TAG = "_evograd_olmoe_"


def _modeling():
    return hf_modules.modeling


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


def production_attention_block(module, hidden_states, position_embeddings, attention_mask,
                               **kwargs):
    """``OlmoeAttention.forward`` itself, on ``module``: the class's code, whatever
    the instance's ``forward`` is now bound to."""
    if getattr(module.config, "clip_qkv", None) is not None:
        raise NotImplementedError("the OLMoE attention block is declared without clip_qkv")
    out, _ = _modeling().OlmoeAttention.forward(module, hidden_states, position_embeddings,
                                                attention_mask, **kwargs)
    return out


def production_residual_rmsnorm(norm_call, branch, residual):
    summed = residual + branch
    return norm_call(summed), summed


def production_moe_experts(experts_module, x, top_k_index, top_k_weights):
    """The experts module's own call: dispatched to the kernel its config pins."""
    return experts_module(x, top_k_index, top_k_weights)


PRODUCTION: dict[str, Callable] = {
    SITE_ATTENTION_BLOCK: production_attention_block,
    SITE_RESIDUAL: production_residual_rmsnorm,
    SITE_MOE: production_moe_experts,
}


def _is_production(kernel, site: str) -> bool:
    return kernel is None or kernel is PRODUCTION[site]


# ── registry ─────────────────────────────────────────────────────────────────

_REGISTRY: SiteRegistry | None = None


def olmoe_sites() -> SiteRegistry:
    """The three sites. The observed shapes are checked at real size by the
    gates, not here."""
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


def _run_site(state, site: str, inputs: dict, activations: tuple[str, ...], call, key=None):
    """Run one site, handing its live tensors to a validator when one listens.

    With a listener, every activation input and every output is replaced by an
    identity view whose hook records the gradient crossing the boundary -- the
    upstream gradient on the way in, the emitted one on the way out. ``key``
    names the call (default: the layer).
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
    listener(site, inputs, items, upstream, emitted,
             layer=state.get("layer_idx") if key is None else key)
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
    """``OlmoeAttention.forward`` as one site. Same signature as the original, so
    ``position_ids`` and ``use_cache`` arrive in ``kwargs`` and reach the
    production path exactly as they do natively."""
    state = getattr(self, _TAG + "state")
    kernel = state["kernels"].get(SITE_ATTENTION_BLOCK)
    if past_key_values is not None:
        raise NotImplementedError("the OLMoE attention adapter is a training boundary; "
                                  "cache-enabled execution is not supported")
    production = _is_production(kernel, SITE_ATTENTION_BLOCK)
    if not production and attention_mask is not None:
        raise NotImplementedError("olmoe_attention_block is declared for the mask-free causal "
                                  "branch; this call supplied an attention mask")
    cos, sin = position_embeddings
    state["counters"].hit(SITE_ATTENTION_BLOCK)
    out = _run_site(
        state, SITE_ATTENTION_BLOCK,
        {"x": hidden_states, "q_weight": self.q_proj.weight, "k_weight": self.k_proj.weight,
         "v_weight": self.v_proj.weight, "q_norm_weight": self.q_norm.weight,
         "k_norm_weight": self.k_norm.weight, "o_weight": self.o_proj.weight,
         "cos": _shared_positions(cos), "sin": _shared_positions(sin),
         "eps": float(self.q_norm.variance_epsilon)},
        ("x",),
        lambda i: (production_attention_block(self, i["x"], position_embeddings, attention_mask,
                                              **kwargs)
                   if production
                   else kernel(i["x"], i["q_weight"], i["k_weight"], i["v_weight"],
                               i["q_norm_weight"], i["k_norm_weight"], i["o_weight"], i["cos"],
                               i["sin"], i["eps"])))
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


def _residual_site(state, norm, branch, residual, category: str, norm_call=None):
    """One residual add and the norm after it, either spelling, handed to any
    validator under ``(layer, category)``: the post-attention, cross-layer and
    final fusions are wired differently, and a validator must see which it got.
    ``norm_call`` is how the production path computes the norm (default: the
    module)."""
    kernel = state["kernels"].get(SITE_RESIDUAL)
    state["counters"].hit(SITE_RESIDUAL)
    return _run_site(
        state, SITE_RESIDUAL,
        {"branch": branch, "residual": residual, "weight": norm.weight,
         "eps": float(norm.variance_epsilon)},
        ("branch", "residual"),
        lambda i: (production_residual_rmsnorm(norm_call or norm, i["branch"], i["residual"])
                   if _is_production(kernel, SITE_RESIDUAL)
                   else kernel(i["branch"], i["residual"], i["weight"], i["eps"])),
        key=(state.get("layer_idx"), category))


def _decoder_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_values=None, use_cache=False, position_embeddings=None,
                           **kwargs):
    """``OlmoeDecoderLayer.forward`` that hands its closing add to the next norm.

    Same operations in the same order. Past layer 0, what arrives is the
    previous layer's un-added MoE branch, and its add is fused with this layer's
    ``input_layernorm``; the post-attention add and its norm are the second call;
    the closing ``residual + moe`` is left in the carrier and the branch
    returned, for the next layer or ``model.norm``.
    """
    state = getattr(self, _TAG + "state")
    carrier = state["carrier"]
    if kwargs.get("output_hidden_states"):
        raise NotImplementedError(
            "the fused residual path changes what flows between decoder layers to the "
            "un-added branch, so output_hidden_states would report a tensor that is not "
            "the residual stream")
    if past_key_values is not None or use_cache:
        raise NotImplementedError("the OLMoE decoder adapter is a training boundary; "
                                  "cache-enabled execution is not supported")
    if state["layer_idx"] == 0:
        # No decoder residual add precedes this norm: native, as the model runs it.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        branch, previous = carrier.take()
        hidden_states, residual = _residual_site(state, self.input_layernorm, branch, previous,
                                                 "mlp_to_next_input")
    branch, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                               position_ids=position_ids, past_key_values=past_key_values,
                               use_cache=use_cache, position_embeddings=position_embeddings,
                               **kwargs)
    normed, summed = _residual_site(state, self.post_attention_layernorm, branch, residual,
                                    "post_attention")
    moe = self.mlp(normed)
    carrier.put(moe, summed)
    return moe


def _final_norm_forward(self, hidden_states):
    """``model.norm`` fused with the last layer's closing add. What the model loop
    passes is that layer's un-added branch; its residual is in the carrier."""
    state = getattr(self, _TAG + "state")
    branch, residual = state["carrier"].take()
    if branch is not hidden_states:
        raise RuntimeError("the final norm received a tensor the last decoder layer did not "
                           "produce; the model loop is not the one this adapter was built for")
    normed, _summed = _residual_site(state, self, branch, residual, "final_model_norm",
                                     state["norm_forward"])
    return normed


def _install(module, forward, kernels: dict, counters: SiteCounters, layer_idx,
             **extra) -> None:
    state = getattr(module, _TAG + "state", None)
    if state is None:
        state = {}
        setattr(module, _TAG + "state", state)
        module.forward = types.MethodType(forward, module)
    state.update({"kernels": dict(kernels), "counters": counters, "layer_idx": layer_idx,
                  **extra})


def install_residual_adapters(model, kernels: dict, counters: SiteCounters):
    """Rebind every decoder layer and ``model.norm`` onto one shared carrier.

    Returns the module paths touched and the carrier. The layers and the final
    norm are one coordinated site: patching only some of them would leave a
    residual stream half carried and half added.
    """
    inner = model.model
    carrier = ResidualCarrier()
    paths = []
    for index, layer in enumerate(inner.layers):
        _install(layer, _decoder_layer_forward, kernels, counters, index, carrier=carrier)
        paths.append(f"model.layers.{index}")
    # The original forward, captured before rebinding: the production path must
    # reach the norm's own computation, not the adapter that replaced it.
    state = getattr(inner.norm, _TAG + "state", None)
    norm_forward = state["norm_forward"] if state else inner.norm.forward
    _install(inner.norm, _final_norm_forward, kernels, counters, None, carrier=carrier,
             norm_forward=norm_forward)
    paths.append("model.norm")
    return paths, carrier


# ── the whole model ──────────────────────────────────────────────────────────


def expected_counts(num_layers: int) -> dict[str, int]:
    """What one forward of a model with this depth invokes, per site: once per
    layer, the residual site twice (``layers - 1`` cross-layer fusions plus the
    final one with ``model.norm``; layer 0's input norm is not one)."""
    layers = int(num_layers)
    return {SITE_ATTENTION_BLOCK: layers, SITE_RESIDUAL: 2 * layers, SITE_MOE: layers}


def patch_model(model, kernels, counters: SiteCounters | None = None,
                *, expected_layers: int | None = None):
    """Install every requested site on a built ``OlmoeForCausalLM``.

    Returns ``(provenance, counters, carrier)``; ``carrier`` is the residual
    site's, ``None`` when that site is not requested. A requested site that
    reaches fewer modules than the depth says raises: a provider labelled
    "patched" that is partly native would report native numbers as the kernel's.
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
        if SITE_ATTENTION_BLOCK in selected:
            _install(layer.self_attn, _attention_forward, selected, counters, index)
            paths.setdefault(SITE_ATTENTION_BLOCK, []).append(f"model.layers.{index}.self_attn")
        if SITE_MOE in selected:
            _install(layer.mlp, _moe_forward, selected, counters, index)
            paths.setdefault(SITE_MOE, []).append(f"model.layers.{index}.mlp.experts")
    carrier = None
    if SITE_RESIDUAL in selected:
        paths[SITE_RESIDUAL], carrier = install_residual_adapters(model, selected, counters)
    reached = {SITE_ATTENTION_BLOCK: len(layers), SITE_RESIDUAL: len(layers) + 1,
               SITE_MOE: len(layers)}
    for site in requested:
        found = len(paths.get(site, ()))
        if found != reached[site]:
            raise ValueError(f"patch site {site} reached {found} modules; the model's "
                             f"{len(layers)} layers need {reached[site]}")
    provenance = PatchProvenance(
        method="module_surgery", requested_sites=tuple(sorted(requested)),
        actual_sites=tuple(sorted(paths)), paths={k: tuple(v) for k, v in paths.items()})
    return provenance, counters, carrier


def module_patches() -> tuple[ModulePatch, ...]:
    """The declarative view of what this workload rebinds, for a report. The
    residual site also rebinds ``model.norm``; ``patch_model`` does the installing."""
    def _named(kind: str):
        return lambda module: type(module).__name__ == kind

    return (
        ModulePatch(SITE_ATTENTION_BLOCK, _named("OlmoeAttention"), lambda o, k: o),
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
    carrier: ResidualCarrier | None
    expected: dict[str, int]

    def observed(self) -> dict[str, int]:
        return self.counters.snapshot()

    def count_problems(self) -> list[str]:
        """Patched sites whose observed invocation count is not the declared one."""
        observed = self.observed()
        return [f"{site}: expected {self.expected.get(site)} invocations, observed "
                f"{observed.get(site, 0)}"
                for site in self.provenance.actual_sites
                if observed.get(site, 0) != self.expected.get(site)]


__all__ = [
    "OBSERVED_SUITE", "PRODUCTION", "PatchedModel", "SiteCounters", "bound_pair_identity_kernels",
    "expected_counts", "install_residual_adapters", "module_patches", "olmoe_sites",
    "patch_model", "set_tap", "structural_identity_kernels",
]
