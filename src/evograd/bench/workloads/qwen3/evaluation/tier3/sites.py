"""Drop-in replacement sites inside a real ``Qwen3ForCausalLM``.

Four places in the model where a kernel can be swapped, and the machinery that
puts one there without the call site, the state dict, the parameters, or the
training loop noticing:

    qkv_norm_rope     projections, per-head Q/K RMSNorm, RoPE
    attention         causal GQA SDPA, then the output projection
    swiglu_mlp        the gated MLP block
    residual_rmsnorm  a residual add immediately followed by an RMSNorm

**Nothing is replaced.** Every adapter rebinds ``forward`` on the *existing*
module instance and leaves ``_parameters``, ``_buffers`` and ``_modules``
untouched. That is stronger than building a lookalike and copying weights
across: the parameters are not copied, they are the same objects, so
``state_dict`` keys and ordering, parameter count, dtype, device and
``requires_grad`` are identical by construction rather than by assertion. It
also makes patching reversible by simply rebuilding the workload, and it means
a mistake shows up as a wrong number rather than as a silently reinitialised
model.

Two ways to reach a kernel, deliberately kept apart:

* **production** -- the exact spelling the installed Transformers executes,
  called through native autograd. Patching every site this way changes the
  module structure and nothing else, so the result must be *bitwise* identical
  to the unmodified model. That is `structural identity`.
* **bound** -- the declared operator through ``opdecl.bind``, i.e. a
  ``torch.autograd.Function`` built from the declaration, which is the path a
  future evolved kernel takes. That is `bound-pair identity`, and it is gated by
  the declared tolerances rather than by bitwise equality.

An unpatched site always uses `production`, so patching one site never silently
changes another.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from evograd.bench.tier3_patch import ModulePatch, PatchProvenance, Site, SiteRegistry

#: What the four sites are called, and which declaration governs each.
SITE_QKV = "qkv_norm_rope"
SITE_ATTENTION = "attention"
SITE_MLP = "swiglu_mlp"
SITE_RESIDUAL = "residual_rmsnorm"

#: Which sites share one installed adapter. Patching any member of a group
#: installs that group's adapter, and the adapter runs *every* member's boundary
#: on the same call -- the unpatched ones through their production spelling.
#: ``qkv_norm_rope`` and ``attention`` are one Qwen3Attention adapter with two
#: switches, so patching either makes both live. Anything reading counts has to
#: know that, and this is the one place it is written down.
ADAPTER_GROUPS: tuple[tuple[str, ...], ...] = (
    (SITE_QKV, SITE_ATTENTION),
    (SITE_MLP,),
    (SITE_RESIDUAL,),
)


def live_sites(requested) -> tuple[str, ...]:
    """Every site that becomes a live boundary when ``requested`` is patched.

    A superset of ``requested``: the sites that merely come along because they
    share an adapter with something that was asked for.
    """
    wanted = set(requested)
    live: set[str] = set()
    for group in ADAPTER_GROUPS:
        if wanted.intersection(group):
            live.update(group)
    return tuple(sorted(live))


def supporting_sites(requested) -> tuple[str, ...]:
    """The live sites that were carried along rather than asked for."""
    return tuple(sorted(set(live_sites(requested)) - set(requested)))


#: Attribute prefix for everything an adapter attaches to a live module. Plain
#: attributes, never parameters or buffers, so ``state_dict`` cannot see them.
_TAG = "_evograd_qwen3_"

#: "no key given", distinct from a real key of ``None`` (the final norm has no layer).
_UNSET = object()


# ── invocation counting ──────────────────────────────────────────────────────


@dataclass
class SiteCounters:
    """How many times each site actually ran, per forward and cumulatively.

    Declared counts are a claim about the architecture; these are the
    observation. A patch that matches 28 modules but runs 27 of them is a
    different defect from one that matches 27, and only counting both finds it.
    """

    counts: dict[str, int] = field(default_factory=dict)

    def hit(self, site: str) -> None:
        self.counts[site] = self.counts.get(site, 0) + 1

    def reset(self) -> None:
        self.counts.clear()

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


# ── the residual carrier ─────────────────────────────────────────────────────


class ResidualCarrier:
    """Hands one decoder layer's un-added branch to the next fusion site.

    ``fused_add_rms_norm`` is the only site here that is not one module. Its two
    halves live in different ones: the add at the end of decoder layer *i*, the
    norm at the start of layer *i+1* (or in ``model.norm`` for the last). Fusing
    them means the add must not have happened yet when the norm runs, so the
    layer stops performing it and puts the pair here instead.

    What flows between layers in the model loop therefore becomes the *branch*
    output rather than the residual stream. Nothing else reads it -- the loop
    only passes it on, and ``model.norm`` takes its inputs from the carrier --
    so the model's output is unchanged. ``output_hidden_states`` is the one
    caller that would read it, and the adapter refuses that mode rather than
    reporting a tensor that no longer means what its name says.
    """

    def __init__(self) -> None:
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def put(self, branch: torch.Tensor, residual: torch.Tensor) -> None:
        if self.pending is not None:
            raise RuntimeError(
                "residual carrier already holds a pending branch; a decoder "
                "layer ran twice without its fusion site consuming the first"
            )
        self.pending = (branch, residual)

    def take(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending is None:
            raise RuntimeError(
                "residual carrier is empty; a fusion site ran before the "
                "decoder layer that feeds it"
            )
        pair, self.pending = self.pending, None
        return pair

    def reset(self) -> None:
        self.pending = None


# ── the production spellings, as the model calls them ────────────────────────


def production_qkv_norm_rope(module, hidden_states, cos, sin):
    """Exactly ``Qwen3Attention.forward``'s first five lines, on ``module``.

    Not a re-derivation: the same submodule calls in the same order, so
    replacing the structure changes no arithmetic. ``apply_rotary_pos_emb`` is
    imported from the installed Transformers rather than reimplemented.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, value_states


def production_attention(module, query, key, value, attention_mask, input_shape, **kwargs):
    """Exactly ``Qwen3Attention.forward``'s attention interface and output projection."""
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output), attn_weights


def production_swiglu_mlp(module, x):
    """Exactly ``Qwen3MLP.forward``."""
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


def production_residual_rmsnorm(norm_call, branch, residual):
    """The add and the norm, in the order and spelling the model runs them.

    ``residual + branch``, not the other way round: floating addition is
    commutative and exact, but writing it the same way as
    ``Qwen3DecoderLayer.forward`` removes the question. ``norm_call`` is the
    original ``Qwen3RMSNorm`` computation -- for ``model.norm``, whose forward
    this file rebinds, that is the bound method captured before rebinding, not
    the module, or the production path would call itself.
    """
    summed = residual + branch
    return norm_call(summed), summed


# ── the registry ─────────────────────────────────────────────────────────────


def _observed(op_name: str):
    """The calibrated observed configurations for one operator, if it has any."""
    from evograd.ops import get_op

    try:
        return tuple(get_op(op_name).benchmark_workloads(suite="qwen3_0_6b_observed"))
    except Exception:  # pragma: no cover - a declaration without the suite
        return ()


def build_registry() -> SiteRegistry:
    """Qwen3's four sites, with their calibrated shapes attached to preflight.

    The preflight workloads are the whole point of attaching them here: they are
    the configurations this model actually presents, and until the tolerance
    calibration landed they could not pass. A candidate is now gated on them
    before tier 3 will time it.
    """
    return SiteRegistry(
        name="qwen3_0_6b",
        sites=(
            Site(SITE_QKV, "qwen3_qkv_norm_rope", production_qkv_norm_rope,
                 preflight=_observed("qwen3_qkv_norm_rope")),
            Site(SITE_ATTENTION, "qwen3_attention", production_attention,
                 preflight=_observed("qwen3_attention")),
            Site(SITE_MLP, "qwen3_swiglu_mlp", production_swiglu_mlp,
                 preflight=_observed("qwen3_swiglu_mlp")),
            Site(SITE_RESIDUAL, "fused_add_rms_norm", production_residual_rmsnorm,
                 preflight=_observed("fused_add_rms_norm")),
        ),
    )


#: Built lazily so importing this module never imports the operator registry.
_REGISTRY: SiteRegistry | None = None


def qwen3_sites() -> SiteRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_registry()
    return _REGISTRY


# ── kernel selection ─────────────────────────────────────────────────────────


def _is_production(kernel, site: str) -> bool:
    return kernel is qwen3_sites().require(site).default


def structural_identity_kernels(registry: SiteRegistry | None = None):
    """Every site patched with the production spelling it already had.

    The adapters are installed and the module structure changes; the arithmetic
    does not, because each site calls the same submodules in the same order
    through native autograd. So the result must be **bitwise** identical to the
    unmodified model, and anything else is a defect in the restructure rather
    than a tolerance question.

    Distinct from the bound-pair control below and from tier 3's generic
    ``identity_control_kernels``, whose backward recomputes the forward.
    """
    from evograd.bench.tier3_patch import KernelSet, KernelSource, patch

    registry = registry or qwen3_sites()
    kernels = KernelSet(registry=registry)
    for site in registry.sites:
        kernels = patch(
            kernels, site.name, site.default,
            source=KernelSource(site=site.name, op_name=site.op, module=None,
                                origin="structural_identity"),
        )
    return kernels


def bound_pair_identity_kernels(ops, sites: tuple[str, ...] | None = None,
                                registry: SiteRegistry | None = None):
    """Every site patched with the declared pair, through ``bind``.

    The deployment path an evolved kernel will take: ``opdecl.bind`` wraps the
    pair in a ``torch.autograd.Function`` and the rank adapter reshapes around
    it. Gated by the declared tolerances, not by bitwise equality, because the
    declared reference and the production spelling are different computations.

    The pair is ``eager_pair_for``, whose backward recomputes the forward, so
    what this measures is a *ceiling* on the plumbing rather than its cost.
    """
    from evograd.bench.tier3_patch import identity_control_kernels

    return identity_control_kernels(ops, sites, registry=registry or qwen3_sites())


# ── the composite attention adapter ──────────────────────────────────────────


def _probe_outputs(state, outputs):
    """Alias each differentiable output so its *external* gradient is readable.

    A hook on the tensor an operator returned sees every consumer's
    contribution, and one of those consumers can be the operator itself: the
    residual fusion normalizes the same ``summed`` it hands back, so a hook
    there reads the boundary gradient plus the operator's own internal one. The
    two are not separable after the fact.

    An alias separates them. ``view_as`` is identity in value, dtype and
    storage -- no copy, no numerical change -- but it is a distinct graph node,
    and only the *downstream* consumers see it. A hook on the alias is therefore
    exactly the gradient the model delivers into that boundary. The aliases
    exist only while a validator is attached; nothing measured ever sees one.
    """
    listener = state.get("tap")
    if listener is None or not getattr(listener, "probes", False):
        return outputs, None
    single = not isinstance(outputs, tuple)
    got = (outputs,) if single else outputs
    upstream: dict[int, torch.Tensor] = {}
    aliased = []
    for index, tensor in enumerate(got):
        if torch.is_tensor(tensor) and tensor.requires_grad:
            alias = tensor.view_as(tensor)
            alias.register_hook(
                lambda grad, _i=index: upstream.__setitem__(_i, grad.detach())
            )
            aliased.append(alias)
        else:
            aliased.append(tensor)
    return (aliased[0] if single else tuple(aliased)), upstream


def _probe_inputs(state, mapping: dict, names: tuple[str, ...]):
    """Alias the activations an operator is about to consume.

    ``_probe_outputs`` reasoning, the other way round. A hook on an input tensor
    reads what *every* consumer of it contributed, and consumers are shared:
    layer 0's ``hidden_states`` feeds both ``input_layernorm`` and the
    post-attention fusion's residual, so a hook there is the sum of two
    operators' emissions and belongs to neither. The operator consumes the alias
    instead, and the alias has exactly one consumer -- this call.

    Only activations are aliased; a parameter is read off its module by the
    production spelling and cannot be substituted. That is sound here because
    every projection and norm weight belongs to exactly one site invocation,
    which the validator re-checks rather than assumes.
    """
    listener = state.get("tap")
    if listener is None or not getattr(listener, "probes", False):
        return mapping, None
    emitted: dict[str, torch.Tensor] = {}
    probed = dict(mapping)
    for name in names:
        value = mapping.get(name)
        if torch.is_tensor(value) and value.requires_grad:
            alias = value.view_as(value)
            alias.register_hook(
                lambda grad, _n=name: emitted.__setitem__(_n, grad.detach())
            )
            probed[name] = alias
    return probed, emitted


def _tap(state, site: str, inputs: dict, outputs, key=_UNSET, emitted=None):
    """Hand one boundary's live tensors to a validator, if one is listening.

    ``None`` in every measured run -- the taps exist so the identity check can
    compare what the operator was actually given and returned, on the model's
    own tensors, rather than on a reconstruction of them. It fires on the
    production branch and on a candidate's alike: a validator that only ever saw
    the production path would be checking the one provider that cannot be wrong.
    """
    listener = state.get("tap")
    if listener is None:
        return outputs
    outputs, upstream = _probe_outputs(state, outputs)
    site_key = state.get("layer_idx") if key is _UNSET else key
    if getattr(listener, "probes", False):
        listener(site, site_key, inputs, outputs,
                 {"upstream": upstream, "emitted": emitted or {}})
    else:
        listener(site, site_key, inputs, outputs)
    return outputs


def _attention_forward(self, hidden_states, position_embeddings=None,
                       attention_mask=None, past_key_values=None, **kwargs):
    """One adapter, two independently selectable sites, one module replaced.

    The external signature is the installed Transformers 5.16.1 one and the
    return is its ``(attn_output, attn_weights)`` pair, so ``Qwen3DecoderLayer``
    calls this exactly as it called the original. Two sequential patches on the
    same module would have meant two wrappers, two sets of provenance, and an
    ordering question; one adapter holding two switches has none of that.
    """
    state = getattr(self, _TAG + "state")
    counters = state["counters"]
    if past_key_values is not None:
        raise NotImplementedError(
            "the Qwen3 tier-3 attention adapter is a training-step boundary; "
            "cache-enabled execution is not supported"
        )
    cos, sin = position_embeddings
    input_shape = hidden_states.shape[:-1]

    qkv_kernel = state["kernels"].get(SITE_QKV)
    counters.hit(SITE_QKV)
    qkv_inputs = {
        "x": hidden_states, "q_weight": self.q_proj.weight,
        "k_weight": self.k_proj.weight, "v_weight": self.v_proj.weight,
        "q_norm_weight": self.q_norm.weight, "k_norm_weight": self.k_norm.weight,
        "cos": cos, "sin": sin, "eps": self.q_norm.variance_epsilon,
    }
    qkv_inputs, qkv_emitted = _probe_inputs(state, qkv_inputs, ("x",))
    if qkv_kernel is None or _is_production(qkv_kernel, SITE_QKV):
        query, key, value = _tap(
            state, SITE_QKV, qkv_inputs,
            production_qkv_norm_rope(self, qkv_inputs["x"], cos, sin),
            emitted=qkv_emitted,
        )
    else:
        # The declared contract's cos/sin are [1, T, D]: one row of positions,
        # broadcast over the batch. The model's are [B, T, D] with identical
        # rows -- `position_ids` has no padding here -- so taking the first row
        # is exact rather than an approximation, and it is checked once.
        qkv_inputs["cos"] = _shared_positions(state, cos)
        qkv_inputs["sin"] = _shared_positions(state, sin)
        query, key, value = _tap(state, SITE_QKV, qkv_inputs, qkv_kernel(
            qkv_inputs["x"],
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            self.q_norm.weight, self.k_norm.weight,
            qkv_inputs["cos"], qkv_inputs["sin"], self.q_norm.variance_epsilon,
        ), emitted=qkv_emitted)

    attn_kernel = state["kernels"].get(SITE_ATTENTION)
    counters.hit(SITE_ATTENTION)
    attn_inputs = {"q": query, "k": key, "v": value, "o_weight": self.o_proj.weight}
    attn_inputs, attn_emitted = _probe_inputs(state, attn_inputs, ("q", "k", "v"))
    if attn_kernel is None or _is_production(attn_kernel, SITE_ATTENTION):
        out, weights = production_attention(
            self, attn_inputs["q"], attn_inputs["k"], attn_inputs["v"],
            attention_mask, input_shape, **kwargs
        )
        return _tap(state, SITE_ATTENTION, attn_inputs, out,
                    emitted=attn_emitted), weights
    if attention_mask is not None:
        raise NotImplementedError(
            "qwen3_attention is declared for the mask-free causal branch "
            "(attn_mask=None, is_causal=True), which is what the canonical "
            "workload takes; this call supplied an explicit attention mask"
        )
    # The three outputs of the first site feed the second directly: no detach,
    # no recomputation, no contiguous() the contract does not ask for.
    return _tap(state, SITE_ATTENTION, attn_inputs,
                attn_kernel(attn_inputs["q"], attn_inputs["k"], attn_inputs["v"],
                            self.o_proj.weight), emitted=attn_emitted), None


def _shared_positions(state, table: torch.Tensor) -> torch.Tensor:
    """``[B, T, D]`` rotary tables reduced to the declared ``[1, T, D]``.

    Checked once per adapter rather than per call: every row of ``cos`` comes
    from the same ``position_ids`` when there is no padding, so they are equal,
    and if they ever are not this raises instead of silently dropping a row.
    """
    if table.shape[0] == 1:
        return table
    if not state.get("positions_checked"):
        if not bool(torch.equal(table[0], table[1])):
            raise NotImplementedError(
                "the rotary tables differ across the batch, so the declared "
                "[1, T, D] cos/sin cannot represent them; qwen3_qkv_norm_rope "
                "is declared for the unpadded canonical workload"
            )
        state["positions_checked"] = True
    return table[:1]


def install_attention_adapter(module, kernels: dict[str, Callable],
                              counters: SiteCounters, layer_idx: int | None = None):
    """Rebind ``forward`` on a live ``Qwen3Attention``. Nothing is copied."""
    setattr(module, _TAG + "state", {
        "kernels": dict(kernels), "counters": counters, "layer_idx": layer_idx,
    })
    module.forward = types.MethodType(_attention_forward, module)
    return module


# ── the MLP adapter ──────────────────────────────────────────────────────────


def _mlp_forward(self, x):
    state = getattr(self, _TAG + "state")
    state["counters"].hit(SITE_MLP)
    kernel = state["kernels"].get(SITE_MLP)
    mlp_inputs = {"x": x, "gate_weight": self.gate_proj.weight,
                  "up_weight": self.up_proj.weight,
                  "down_weight": self.down_proj.weight}
    mlp_inputs, mlp_emitted = _probe_inputs(state, mlp_inputs, ("x",))
    if kernel is None or _is_production(kernel, SITE_MLP):
        return _tap(state, SITE_MLP, mlp_inputs,
                    production_swiglu_mlp(self, mlp_inputs["x"]),
                    emitted=mlp_emitted)
    return _tap(state, SITE_MLP, mlp_inputs, kernel(
        mlp_inputs["x"], self.gate_proj.weight, self.up_proj.weight,
        self.down_proj.weight), emitted=mlp_emitted)


def install_mlp_adapter(module, kernels: dict[str, Callable],
                        counters: SiteCounters, layer_idx: int | None = None):
    setattr(module, _TAG + "state", {
        "kernels": dict(kernels), "counters": counters, "layer_idx": layer_idx,
    })
    module.forward = types.MethodType(_mlp_forward, module)
    return module


# ── the residual/RMSNorm integration ─────────────────────────────────────────


def _fuse(kernel, site_kernel_is_production, branch, residual, norm_module,
          norm_call=None):
    """One fusion site, either spelling.

    ``norm_module`` supplies the parameters the declared operator takes;
    ``norm_call`` is how the production path computes the norm, defaulting to
    calling the module.
    """
    if site_kernel_is_production:
        return production_residual_rmsnorm(norm_call or norm_module, branch, residual)
    return kernel(branch, residual, norm_module.weight, norm_module.variance_epsilon)


def _fused_site(state, kernel, production, category, branch, residual,
                norm_module, norm_call=None):
    """One residual fusion, probed, executed, and handed to any validator.

    The three fusions are structurally different even though the operator is the
    same: the post-attention one lives inside a layer, the cross-layer one spans
    two, and the final one is ``model.norm``. A validator that checked only the
    first would miss exactly the wiring the other two exercise, so the category
    travels with the invocation.
    """
    inputs = {  # `r` is what fused_add_rms_norm calls the residual.
        "x": branch, "r": residual, "weight": norm_module.weight,
        "eps": norm_module.variance_epsilon,
    }
    inputs, emitted = _probe_inputs(state, inputs, ("x", "r"))
    outputs = _fuse(kernel, production, inputs["x"], inputs["r"], norm_module,
                    norm_call)
    return _tap(state, SITE_RESIDUAL, inputs, outputs,
                key=(state.get("layer_idx"), category), emitted=emitted)


def _decoder_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_values=None, use_cache=False,
                           position_embeddings=None, **kwargs):
    """A decoder layer that hands its trailing add to the next norm.

    Two of the layer's four residual/norm meetings are here. The attention one
    is wholly inside this function. The MLP one is not: its norm belongs to the
    *next* layer, so this returns the un-added MLP branch and leaves the pair in
    the carrier. Layer 0's ``input_layernorm`` has no preceding decoder add and
    stays unfused, which is why there are 56 sites and not 57.
    """
    state = getattr(self, _TAG + "state")
    counters, carrier = state["counters"], state["carrier"]
    kernel = state["kernels"].get(SITE_RESIDUAL)
    production = kernel is None or _is_production(kernel, SITE_RESIDUAL)
    if kwargs.get("output_hidden_states"):
        raise NotImplementedError(
            "the fused residual path changes what flows between decoder layers "
            "to the un-added branch, so output_hidden_states would report a "
            "tensor that is not the residual stream"
        )
    if past_key_values is not None or use_cache:
        raise NotImplementedError(
            "the Qwen3 tier-3 decoder adapter is a training-step boundary; "
            "cache-enabled execution is not supported"
        )

    if state["layer_idx"] == 0:
        # No decoder residual add precedes this norm. Unfused, as the model runs it.
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
    else:
        branch, prev_residual = carrier.take()
        counters.hit(SITE_RESIDUAL)
        normed, residual = _fused_site(
            state, kernel, production, "mlp_to_next_input", branch,
            prev_residual, self.input_layernorm,
        )

    attn_out, _ = self.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    counters.hit(SITE_RESIDUAL)
    post_normed, attn_residual = _fused_site(
        state, kernel, production, "post_attention", attn_out, residual,
        self.post_attention_layernorm,
    )
    mlp_out = self.mlp(post_normed)
    # The add that would close this layer is the next fusion site's first input.
    carrier.put(mlp_out, attn_residual)
    return mlp_out


def _final_norm_forward(self, hidden_states):
    """``model.norm``, fused with the last layer's MLP residual add.

    Ignores its argument by design: what the loop passed is the last layer's
    un-added branch, and the residual it belongs to is in the carrier. Taking
    both from there is what makes this the 56th fusion site rather than a norm
    applied to a sum computed twice.
    """
    state = getattr(self, _TAG + "state")
    kernel = state["kernels"].get(SITE_RESIDUAL)
    production = kernel is None or _is_production(kernel, SITE_RESIDUAL)
    branch, residual = state["carrier"].take()
    if branch is not hidden_states:
        raise RuntimeError(
            "the final norm received a tensor the last decoder layer did not "
            "produce; the model loop is not the one this adapter was built for"
        )
    state["counters"].hit(SITE_RESIDUAL)
    normed, _summed = _fused_site(
        state, kernel, production, "final_model_norm", branch, residual, self,
        state["norm_forward"],
    )
    return normed


def install_residual_adapters(model, kernels: dict[str, Callable], counters: SiteCounters):
    """Rebind every decoder layer and the final norm onto one shared carrier.

    Returns the module paths touched. The decoder layers and ``model.norm`` are
    one coordinated site: patching only some of them would leave a model whose
    residual stream is half carried and half added.
    """
    inner = model.model
    carrier = ResidualCarrier()
    paths: list[str] = []
    for index, layer in enumerate(inner.layers):
        setattr(layer, _TAG + "state", {
            "kernels": dict(kernels), "counters": counters,
            "carrier": carrier, "layer_idx": index,
        })
        layer.forward = types.MethodType(_decoder_layer_forward, layer)
        paths.append(f"model.layers.{index}")
    setattr(inner.norm, _TAG + "state", {
        "kernels": dict(kernels), "counters": counters, "carrier": carrier,
        # Captured before rebinding: the production path must reach the original
        # computation, not the adapter that replaced it.
        "norm_forward": inner.norm.forward,
    })
    inner.norm.forward = types.MethodType(_final_norm_forward, inner.norm)
    paths.append("model.norm")
    setattr(inner, _TAG + "carrier", carrier)
    return paths, carrier


# ── structural counts ────────────────────────────────────────────────────────


def expected_counts(layers: int) -> dict[str, int]:
    """What one canonical forward must invoke, per site.

    ``2 * layers`` residual fusions: one per decoder layer for the attention
    add, one per layer for the MLP add -- of which ``layers - 1`` are consumed
    by the next layer's ``input_layernorm`` and the last by ``model.norm``.
    Layer 0's ``input_layernorm`` is not one, and that is the whole of the
    difference between ``2 * layers`` and ``2 * layers + 1``.
    """
    return {
        SITE_QKV: layers,
        SITE_ATTENTION: layers,
        SITE_MLP: layers,
        SITE_RESIDUAL: 2 * layers,
    }


def patch_model(model, kernels, counters: SiteCounters | None = None,
                *, expected_layers: int | None = None):
    """Install every requested site on a built ``Qwen3ForCausalLM``.

    Returns ``(provenance, counters, carrier)``. A requested site that reaches
    zero modules, or fewer than the architecture has, raises: a provider
    labelled "patched" that is byte-identical to eager would report eager's
    numbers as the kernel's.
    """
    counters = counters or SiteCounters()
    registry = kernels.registry
    requested = tuple(kernels.patched)
    selected = {site: kernels.kernel_for(site) for site in requested}
    # The architecture's layer count, not the live list's. Counting the list
    # against itself can only ever agree, so a model whose layers were replaced
    # or truncated would patch "completely" and measure something else.
    declared = (
        expected_layers
        if expected_layers is not None
        else getattr(model.config, "num_hidden_layers", None)
    )
    layers = declared if declared is not None else len(model.model.layers)
    if len(model.model.layers) != layers:
        raise ValueError(
            f"the model has {len(model.model.layers)} decoder layers but its "
            f"configuration declares {layers}; a partially patched model would "
            "report the unpatched path's numbers as the kernel's"
        )
    paths: dict[str, tuple[str, ...]] = {}

    attention_group = next(g for g in ADAPTER_GROUPS if SITE_QKV in g)
    attention_sites = [s for s in attention_group if s in selected]
    if attention_sites:
        touched = []
        for index, layer in enumerate(model.model.layers):
            install_attention_adapter(layer.self_attn, selected, counters, index)
            touched.append(f"model.layers.{index}.self_attn")
        _require(len(touched), layers, attention_sites, "Qwen3Attention")
        for site in attention_sites:
            paths[site] = tuple(touched)

    if SITE_MLP in selected:
        touched = []
        for index, layer in enumerate(model.model.layers):
            install_mlp_adapter(layer.mlp, selected, counters, index)
            touched.append(f"model.layers.{index}.mlp")
        _require(len(touched), layers, [SITE_MLP], "Qwen3MLP")
        paths[SITE_MLP] = tuple(touched)

    carrier = None
    if SITE_RESIDUAL in selected:
        touched, carrier = install_residual_adapters(model, selected, counters)
        _require(len(touched), layers + 1, [SITE_RESIDUAL], "decoder layer / final norm")
        paths[SITE_RESIDUAL] = tuple(touched)

    provenance = PatchProvenance(
        method="module_surgery",
        requested_sites=tuple(sorted(requested)),
        actual_sites=tuple(sorted(paths)),
        paths=paths,
    )
    return provenance, counters, carrier


def _require(found: int, expected: int, sites, what: str) -> None:
    if found != expected:
        raise ValueError(
            f"patch sites {sorted(sites)} reached {found} {what} modules, "
            f"expected {expected}; a partially patched model would report the "
            "unpatched path's numbers as the kernel's"
        )


def module_patches() -> tuple[ModulePatch, ...]:
    """The declarative view of what this workload replaces, for the report.

    ``patch_model`` does the installing, because the residual site is not a
    submodule replacement at all -- it is a decoder-layer and model-loop
    integration. These describe the same set for anything that wants to read it.
    """
    def _named(kind: str):
        return lambda module: type(module).__name__ == kind

    return (
        ModulePatch(SITE_QKV, _named("Qwen3Attention"), lambda o, k: o,
                    sites=(SITE_ATTENTION,)),
        ModulePatch(SITE_MLP, _named("Qwen3MLP"), lambda o, k: o),
        ModulePatch(SITE_RESIDUAL, _named("Qwen3DecoderLayer"), lambda o, k: o),
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
        """Sites whose observed invocation count is not the declared one."""
        observed = self.observed()
        problems = []
        for site in self.provenance.actual_sites:
            want, got = self.expected.get(site), observed.get(site, 0)
            if want is not None and got != want:
                problems.append(f"{site}: expected {want} invocations, observed {got}")
        return problems
