"""Drop-in sites inside a Qwen3.5 decoder layer, and across the whole model.

Seven sites in two block kinds (``levels.level2.manifest.SITES_BY_BLOCK``)::

    Gated DeltaNet layer (24 of 32)       full-attention layer (8 of 32)
      gdn_in_proj                           qkv_norm_partial_rope
      gated_delta_rule                      gated_attention
      gdn_norm_out
      residual_rmsnorm                      residual_rmsnorm
      swiglu_mlp                            swiglu_mlp

One registry holds all seven sites and patches every layer by its type; a
per-kind registry (one layer type's sites) refuses a candidate for a site that
layer type does not have, by name.

**Nothing is replaced.** As for Qwen3-0.6B, every adapter rebinds
``forward`` on the existing module instance; parameters are the same objects, so
state dict, parameter order, dtype and ``requires_grad`` are unchanged by
construction. Four adapters, one per module that holds sites:

* ``Qwen3_5GatedDeltaNet.forward`` -- sites 1-3 (four input projections, no
  deinterleave);
* ``Qwen3_5Attention.forward``     -- sites 4, 5;
* ``Qwen3_5MLP.forward``           -- the dense SwiGLU (Qwen3-0.6B's adapter);
* ``Qwen3_5DecoderLayer.forward``  -- the post-attention residual site: the
  input norm and the closing residual add stay native.
  The cross-layer fusion and the final-norm fusion (Qwen3-0.6B's
  ``ResidualCarrier``) are a later step's.

**Production spellings are pinned.** Each registry default is the exact code the
installed Transformers runs, written against the module, with the two
import-time-dispatched functions reached through ``__wrapped__``
(``torch_chunk_gated_delta_rule``, ``causal_conv1d_fn``). Patching every site
with its production spelling is ``structural identity``: bitwise the native
layer and the native model.

The interface (``patch_model``, ``module_patches``, ``set_tap``,
``structural_identity_kernels``, ``bound_pair_identity_kernels``,
``expected_counts``, ``PatchedModel``) has Qwen3-0.6B's names and shapes, so
model-agnostic verification can bridge to this workload unchanged.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from evograd.evaluation.tier3.patch import ModulePatch, PatchProvenance, Site, SiteRegistry

from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import hf_modules
from evograd.benchmark.topdown.qwen3_5_4b.levels.level2.manifest import (
    BLOCK_FULL,
    BLOCK_GDN,
    SITE_ATTENTION,
    SITE_GDN_IN,
    SITE_GDN_OUT,
    SITE_GDR,
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    SITE_TASKS,
    SITES,
    SITES_BY_BLOCK,
)

WORKLOAD = "qwen3_5_4b"
OBSERVED_SUITE = "qwen3_5_4b_observed"

#: Sites served by one installed adapter. Patching any member installs the
#: adapter, and the adapter runs every member's boundary on the same call -- the
#: unpatched ones through their production spelling.
ADAPTER_GROUPS: tuple[tuple[str, ...], ...] = (
    (SITE_GDN_IN, SITE_GDR, SITE_GDN_OUT),
    (SITE_QKV, SITE_ATTENTION),
    (SITE_RESIDUAL,),
    (SITE_MLP,),
)

_TAG = "_evograd_qwen3_5_"


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


def production_gdn_in_proj(module, hidden_states):
    """``Qwen3_5GatedDeltaNet.forward`` from its input to the delta rule, cache-free."""
    modeling = _modeling()
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = module.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = module.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)
    mixed_qkv = modeling.causal_conv1d_fn.__wrapped__(
        mixed_qkv, module.conv1d.weight.squeeze(1), module.conv1d.bias,
        activation=module.activation)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [module.key_dim, module.key_dim,
                                                module.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    return query, key, value, z, beta, g


def production_gated_delta_rule(module, query, key, value, g, beta):
    """The torch chunk implementation, through ``__wrapped__``."""
    core_attn_out, _ = _modeling().torch_chunk_gated_delta_rule.__wrapped__(
        query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    return core_attn_out


def production_gdn_norm_out(module, core_attn_out, z):
    batch_size, seq_len = z.shape[0], z.shape[1]
    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return module.out_proj(core_attn_out)


def production_qkv_norm_partial_rope(module, hidden_states, cos, sin):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states, gate = torch.chunk(
        module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = _modeling().apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, value_states, gate


def production_gated_attention(module, query, key, value, gate, attention_mask, input_shape,
                               **kwargs):
    modeling = _modeling()
    attention_interface: Callable = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, modeling.eager_attention_forward)
    attn_output, _ = attention_interface(
        module, query, key, value, attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return module.o_proj(attn_output)


def production_residual_rmsnorm(norm_module, branch, residual):
    summed = residual + branch
    return norm_module(summed), summed


def production_swiglu_mlp(module, x):
    """Exactly ``Qwen3_5MLP.forward``."""
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


PRODUCTION: dict[str, Callable] = {
    SITE_GDN_IN: production_gdn_in_proj,
    SITE_GDR: production_gated_delta_rule,
    SITE_GDN_OUT: production_gdn_norm_out,
    SITE_QKV: production_qkv_norm_partial_rope,
    SITE_ATTENTION: production_gated_attention,
    SITE_RESIDUAL: production_residual_rmsnorm,
    SITE_MLP: production_swiglu_mlp,
}


def _is_production(kernel, site: str) -> bool:
    return kernel is None or kernel is PRODUCTION[site]


# ── registries ───────────────────────────────────────────────────────────────

_REGISTRIES: dict[str | None, SiteRegistry] = {}


def qwen3_5_sites(kind: str | None = None) -> SiteRegistry:
    """The seven sites (``kind=None``), or one layer kind's subset.

    Preflight checks each task's declared correctness grid. The observed shapes
    are checked at real size by Tier 2 and the calibration inventory, not here.
    """
    if kind is not None and kind not in SITES_BY_BLOCK:
        raise ValueError(f"unknown block kind {kind!r}; known: {sorted(SITES_BY_BLOCK)}")
    if kind not in _REGISTRIES:
        sites = SITES if kind is None else SITES_BY_BLOCK[kind]
        name = WORKLOAD if kind is None else f"{WORKLOAD}.{kind}"
        _REGISTRIES[kind] = SiteRegistry(
            name=name,
            sites=tuple(Site(site, SITE_TASKS[site], PRODUCTION[site]) for site in sites),
        )
    return _REGISTRIES[kind]


def structural_identity_kernels(registry: SiteRegistry | None = None):
    """Every site patched with the production spelling it already had: bitwise native."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    registry = registry or qwen3_5_sites()
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

    return identity_control_kernels(ops, sites, registry=registry or qwen3_5_sites())


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
    """``[B, T, R]`` rotary tables reduced to the declared ``[1, T, R]`` (exact)."""
    if table.shape[0] == 1:
        return table
    if not bool(torch.equal(table[0], table[-1])):
        raise NotImplementedError("rotary tables differ across the batch; the declared "
                                  "[1, T, R] cos/sin cannot represent them")
    return table[:1]


# ── adapters ─────────────────────────────────────────────────────────────────


def _gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    if cache_params is not None:
        raise NotImplementedError("the Qwen3.5 GDN adapter is a training boundary; "
                                  "cache-enabled execution is not supported")
    if attention_mask is not None:
        hidden_states = _modeling().apply_mask_to_padding_states(hidden_states, attention_mask)

    kernel = kernels.get(SITE_GDN_IN)
    counters.hit(SITE_GDN_IN)
    query, key, value, z, beta, g = _run_site(
        state, SITE_GDN_IN,
        {"x": hidden_states, "w_qkv": self.in_proj_qkv.weight, "w_z": self.in_proj_z.weight,
         "w_b": self.in_proj_b.weight, "w_a": self.in_proj_a.weight,
         "conv_w": self.conv1d.weight.squeeze(1), "A_log": self.A_log, "dt_bias": self.dt_bias},
        ("x",),
        lambda i: (production_gdn_in_proj(self, i["x"]) if _is_production(kernel, SITE_GDN_IN)
                   else kernel(i["x"], i["w_qkv"], i["w_z"], i["w_b"], i["w_a"], i["conv_w"],
                               i["A_log"], i["dt_bias"])))

    kernel = kernels.get(SITE_GDR)
    counters.hit(SITE_GDR)
    core = _run_site(
        state, SITE_GDR, {"q": query, "k": key, "v": value, "g": g, "beta": beta},
        ("q", "k", "v", "g", "beta"),
        lambda i: (production_gated_delta_rule(self, i["q"], i["k"], i["v"], i["g"], i["beta"])
                   if _is_production(kernel, SITE_GDR)
                   else kernel(i["q"], i["k"], i["v"], i["g"], i["beta"])))

    kernel = kernels.get(SITE_GDN_OUT)
    counters.hit(SITE_GDN_OUT)
    return _run_site(
        state, SITE_GDN_OUT,
        {"o": core, "z": z, "w_norm": self.norm.weight, "w_out": self.out_proj.weight,
         "eps": float(self.norm.variance_epsilon)},
        ("o", "z"),
        lambda i: (production_gdn_norm_out(self, i["o"], i["z"])
                   if _is_production(kernel, SITE_GDN_OUT)
                   else kernel(i["o"], i["z"], i["w_norm"], i["w_out"], i["eps"])))


def _attention_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                       past_key_values=None, **kwargs):
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    if past_key_values is not None:
        raise NotImplementedError("the Qwen3.5 attention adapter is a training boundary; "
                                  "cache-enabled execution is not supported")
    cos, sin = position_embeddings
    input_shape = hidden_states.shape[:-1]

    kernel = kernels.get(SITE_QKV)
    counters.hit(SITE_QKV)
    query, key, value, gate = _run_site(
        state, SITE_QKV,
        {"x": hidden_states, "q_weight": self.q_proj.weight, "k_weight": self.k_proj.weight,
         "v_weight": self.v_proj.weight, "q_norm_weight": self.q_norm.weight,
         "k_norm_weight": self.k_norm.weight, "cos": _shared_positions(cos),
         "sin": _shared_positions(sin), "eps": float(self.q_norm.eps)},
        ("x",),
        lambda i: (production_qkv_norm_partial_rope(self, i["x"], cos, sin)
                   if _is_production(kernel, SITE_QKV)
                   else kernel(i["x"], i["q_weight"], i["k_weight"], i["v_weight"],
                               i["q_norm_weight"], i["k_norm_weight"], i["cos"], i["sin"],
                               i["eps"])))

    kernel = kernels.get(SITE_ATTENTION)
    counters.hit(SITE_ATTENTION)
    if not _is_production(kernel, SITE_ATTENTION) and attention_mask is not None:
        raise NotImplementedError("qwen3_5_gated_attention is declared for the mask-free "
                                  "causal branch; this call supplied an attention mask")
    out = _run_site(
        state, SITE_ATTENTION,
        {"q": query, "k": key, "v": value, "gate": gate, "o_weight": self.o_proj.weight},
        ("q", "k", "v", "gate"),
        lambda i: (production_gated_attention(self, i["q"], i["k"], i["v"], i["gate"],
                                              attention_mask, input_shape, **kwargs)
                   if _is_production(kernel, SITE_ATTENTION)
                   else kernel(i["q"], i["k"], i["v"], i["gate"], i["o_weight"])))
    return out, None


def _mlp_forward(self, x):
    state = getattr(self, _TAG + "state")
    state["counters"].hit(SITE_MLP)
    kernel = state["kernels"].get(SITE_MLP)
    return _run_site(
        state, SITE_MLP,
        {"x": x, "gate_weight": self.gate_proj.weight, "up_weight": self.up_proj.weight,
         "down_weight": self.down_proj.weight},
        ("x",),
        lambda i: (production_swiglu_mlp(self, i["x"]) if _is_production(kernel, SITE_MLP)
                   else kernel(i["x"], i["gate_weight"], i["up_weight"], i["down_weight"])))


def _layer_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                   position_ids=None, past_key_values=None, **kwargs):
    """``Qwen3_5DecoderLayer.forward`` with the post-attention add + norm as a site.

    Same operations in the same order: input norm native, the token mixer
    through its own (possibly adapted) forward, the residual site, the MLP, and
    the closing ``residual + branch`` as the original writes it.
    """
    state = getattr(self, _TAG + "state")
    counters, kernels = state["counters"], state["kernels"]
    if past_key_values is not None:
        raise NotImplementedError("the Qwen3.5 decoder adapter is a training boundary")
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    if self.block_type == "linear_attention":
        branch = self.linear_attn(hidden_states=hidden_states, cache_params=past_key_values,
                                  attention_mask=attention_mask, **kwargs)
    else:
        branch, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values,
                                   position_embeddings=position_embeddings, **kwargs)
    kernel = kernels.get(SITE_RESIDUAL)
    counters.hit(SITE_RESIDUAL)
    norm = self.post_attention_layernorm
    normed, summed = _run_site(
        state, SITE_RESIDUAL,
        {"branch": branch, "residual": residual, "weight": norm.weight, "eps": float(norm.eps)},
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


_GDN_SITES = (SITE_GDN_IN, SITE_GDR, SITE_GDN_OUT)
_ATTN_SITES = (SITE_QKV, SITE_ATTENTION)


def install_sites(layer, kernels: dict, counters: SiteCounters,
                  layer_idx: int | None = None) -> dict[str, tuple[str, ...]]:
    """Install every requested site's adapter on one decoder layer; return local paths.

    Sites the layer's kind does not have are skipped (a whole-model patch asks
    every layer for every site).
    """
    paths: dict[str, tuple[str, ...]] = {}
    requested = set(kernels)
    if requested & set(_GDN_SITES) and getattr(layer, "linear_attn", None) is not None:
        _install(layer.linear_attn, _gdn_forward, kernels, counters, layer_idx)
        for site in requested & set(_GDN_SITES):
            paths[site] = ("linear_attn",)
    if requested & set(_ATTN_SITES) and getattr(layer, "self_attn", None) is not None:
        _install(layer.self_attn, _attention_forward, kernels, counters, layer_idx)
        for site in requested & set(_ATTN_SITES):
            paths[site] = ("self_attn",)
    if SITE_RESIDUAL in requested:
        _install(layer, _layer_forward, kernels, counters, layer_idx)
        paths[SITE_RESIDUAL] = ("post_attention_layernorm",)
    if SITE_MLP in requested:
        _install(layer.mlp, _mlp_forward, kernels, counters, layer_idx)
        paths[SITE_MLP] = ("mlp",)
    return paths


# ── the whole model ──────────────────────────────────────────────────────────


def expected_counts(layer_types: Sequence[str]) -> dict[str, int]:
    """What one forward of a model with these layer types invokes, per site:
    24 / 8 / 32 for Qwen3.5-4B (each site once per layer of a kind holding it)."""
    return {site: sum(1 for kind in layer_types if site in SITES_BY_BLOCK[kind])
            for site in SITES}


def _layer_types(model) -> tuple[str, ...]:
    return tuple(model.config.layer_types[: model.config.num_hidden_layers])


def patch_model(model, kernels, counters: SiteCounters | None = None,
                *, expected_layers: int | None = None):
    """Install every requested site on a built ``Qwen3_5ForCausalLM``.

    Returns ``(provenance, counters, carrier)``; ``carrier`` is always ``None``
    (no cross-layer fusion in this step), kept for the Qwen3-0.6B shape. A
    requested site that reaches fewer modules than the layer types say raises:
    a provider labelled "patched" that is partly native would report native
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
    types_ = _layer_types(model)
    paths: dict[str, list[str]] = {}
    for index, layer in enumerate(layers):
        if layer.block_type != types_[index]:
            raise ValueError(f"layer {index} is {layer.block_type}, the config says "
                             f"{types_[index]}")
        for site, local in install_sites(layer, selected, counters, index).items():
            paths.setdefault(site, []).extend(f"model.layers.{index}.{p}" for p in local)
    want = expected_counts(types_)
    for site in requested:
        found = len(paths.get(site, ()))
        if found != want[site]:
            raise ValueError(f"patch site {site} reached {found} modules, the layer types "
                             f"have {want[site]}")
    provenance = PatchProvenance(
        method="module_surgery", requested_sites=tuple(sorted(requested)),
        actual_sites=tuple(sorted(paths)), paths={k: tuple(v) for k, v in paths.items()})
    return provenance, counters, None


def module_patches() -> tuple[ModulePatch, ...]:
    """The declarative view of what this workload rebinds, for a report."""
    def _named(kind: str):
        return lambda module: type(module).__name__ == kind

    return (
        ModulePatch(SITE_GDN_IN, _named("Qwen3_5GatedDeltaNet"), lambda o, k: o,
                    sites=(SITE_GDR, SITE_GDN_OUT)),
        ModulePatch(SITE_QKV, _named("Qwen3_5Attention"), lambda o, k: o,
                    sites=(SITE_ATTENTION,)),
        ModulePatch(SITE_RESIDUAL, _named("Qwen3_5DecoderLayer"), lambda o, k: o),
        ModulePatch(SITE_MLP, _named("Qwen3_5MLP"), lambda o, k: o),
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
    "ADAPTER_GROUPS", "BLOCK_FULL", "BLOCK_GDN", "OBSERVED_SUITE", "PRODUCTION", "PatchedModel",
    "SiteCounters", "bound_pair_identity_kernels", "expected_counts", "install_sites",
    "live_sites", "module_patches", "patch_model", "qwen3_5_sites", "set_tap",
    "structural_identity_kernels", "supporting_sites",
]
