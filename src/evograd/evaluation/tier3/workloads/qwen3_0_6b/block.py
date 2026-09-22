"""Qwen3-0.6B's decoder layer as a tier-3 block.

The block boundary is the original ``Qwen3DecoderLayer`` call: one
``hidden_states`` in, one ``hidden_states`` out, the rotary tables and position
ids as non-differentiable keyword arguments, exactly as the model passes them.
Four sites live inside it, each invoked once:

    qkv_norm_rope     the projections, per-head norms and RoPE
    attention         causal GQA SDPA and the output projection
    swiglu_mlp        the gated MLP
    residual_rmsnorm  the post-attention add followed by ``post_attention_layernorm``

The layer's own ``input_layernorm`` and its final residual add stay native and
inside the block, so the output and its backward keep the original contract.
The cross-layer fusion (this layer's MLP add with the *next* layer's input norm)
and the final ``model.norm`` fusion are outside this scope and are listed as
excluded rather than implied: a block report covers one of the model's fusion
sites, not all of them.

The production spellings and the attention/MLP adapters are the ones
:mod:`.sites` installs in the full model; the residual site is a
block-specific decoder forward that reuses ``_fused_site`` with no carrier,
because within one layer there is nothing to carry. ``patch_model`` is not
used: it expects the model loop and hands residual state across layers.

Two input sources, two claims. A **captured** case replays the verified
``layer14.pt`` artifact -- the model's own weights, arguments and upstream
gradient -- and can say what the model computed. A **config-derived** case
builds the block from the pinned architecture with seeded weights, inputs and
cotangents, needs no snapshot, no full model and no checkpoint, and can say
whether a kernel composition is correct and fast inside the block; it cannot
say anything about observed activation distributions.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from evograd.evaluation.tier3.block import (
    BlockCase,
    BlockInvocation,
    BuiltBlock,
    Installed,
    InvocationError,
    tensor_tree_hash,
)
from evograd.evaluation.tier3.patch import KernelSet, PatchProvenance, restrict

from .sites import (
    SITE_ATTENTION,
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    SiteCounters,
    _TAG,
    _fused_site,
    _is_production,
    bound_pair_identity_kernels,
    install_attention_adapter,
    install_mlp_adapter,
    live_sites,
    qwen3_sites,
    set_tap,
    structural_identity_kernels,
    supporting_sites,
)

ARCHITECTURE = "qwen3_0_6b"
BLOCK_KIND = "decoder_layer"
OBSERVED_SUITE = "qwen3_0_6b_observed"
BOUNDARY = ("Qwen3DecoderLayer: hidden_states in -> hidden_states out; input_layernorm "
            "and the final residual add stay native inside the block")
EXCLUDED = (
    "residual_rmsnorm(mlp_to_next_input): cross-layer fusion, covered by the model scope",
    "residual_rmsnorm(final_model_norm): model.norm is not in this block",
)
STATE_CONTRACT = {"use_cache": False, "gradient_checkpointing": False, "carried_state": [],
                  "reset": "zero_grad(set_to_none=True); fresh leaf hidden_states per repetition"}

#: What the block is called with, by path. Stated once; the executor never
#: guesses which element of a return is the output.
DIFFERENTIABLE_INPUTS = ("args[0]",)
OUTPUTS = ("result",)


def _revision() -> str:
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import TESTED_TRANSFORMERS

    try:
        import transformers
        live = transformers.__version__
    except Exception:  # pragma: no cover
        live = "unavailable"
    return (f"Qwen/Qwen3-0.6B config.json as pinned by levels.level4.spec.QWEN3_0_6B; "
            f"transformers {live} (tested {TESTED_TRANSFORMERS})")


def _dims(arch: Mapping[str, Any], batch: int, seq: int) -> dict[str, int]:
    return {"B": batch, "T": seq, "H": int(arch["hidden_size"]), "HQ": int(arch["num_attention_heads"]),
            "HK": int(arch["num_key_value_heads"]), "D": int(arch["head_dim"]),
            "I": int(arch["intermediate_size"])}


# ── the block-specific decoder forward ───────────────────────────────────────


def _block_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                         past_key_values=None, use_cache=False, position_embeddings=None,
                         **kwargs):
    """``Qwen3DecoderLayer.forward`` with the post-attention add+norm as a site.

    Same operations in the same order as the installed Transformers spelling:
    the input norm is native, attention runs through ``self.self_attn`` (whose
    own adapter may be installed), the post-attention residual add and norm go
    through the fusion site, the MLP through ``self.mlp``, and the final add is
    written ``residual + branch`` as the original does. No carrier: nothing
    leaves this layer un-added.
    """
    state = getattr(self, _TAG + "state")
    counters = state["counters"]
    kernel = state["kernels"].get(SITE_RESIDUAL)
    production = kernel is None or _is_production(kernel, SITE_RESIDUAL)
    if past_key_values is not None or use_cache:
        raise NotImplementedError(
            "the Qwen3 block adapter is a training boundary; cache-enabled execution "
            "is not supported"
        )
    residual = hidden_states
    normed = self.input_layernorm(hidden_states)
    attn_out, _ = self.self_attn(
        hidden_states=normed, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache,
        position_embeddings=position_embeddings, **kwargs,
    )
    counters.hit(SITE_RESIDUAL)
    post_normed, attn_residual = _fused_site(
        state, kernel, production, "post_attention", attn_out, residual,
        self.post_attention_layernorm,
    )
    mlp_out = self.mlp(post_normed)
    return attn_residual + mlp_out


# ── the adapter ──────────────────────────────────────────────────────────────


@dataclass
class Qwen3Block:
    """One Qwen3 decoder layer, captured or config-derived, as the executor sees it."""

    case: BlockCase
    arch: dict[str, Any]
    layer_index: int
    dtype: str
    #: The block's weights on the CPU, loaded into every build. For a captured
    #: case they are the model's; for a config case they were drawn once from
    #: the recipe's seed, so every build is the same block whatever the device.
    weights: dict[str, torch.Tensor]
    invocation_cpu: BlockInvocation
    #: Captured case only: what the full model produced at this boundary.
    capture: dict[str, Any] | None = None
    registry: Any = field(default_factory=qwen3_sites)
    observed_suite: str | None = OBSERVED_SUITE

    # -- build / prepare ----------------------------------------------------

    def build(self, *, device: str) -> BuiltBlock:
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.prepare import build_single_layer

        layer = build_single_layer(self.arch, self.layer_index, device=device, dtype=self.dtype)
        layer.load_state_dict({k: v.to(device) for k, v in self.weights.items()}, strict=True)
        return BuiltBlock(module=layer, parameters=dict(layer.named_parameters()),
                          buffers=dict(layer.named_buffers()))

    def prepare(self, built: BuiltBlock, *, device: str) -> BlockInvocation:
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.prepare import verify_layout

        moved = self.invocation_cpu.to(device)
        problems = verify_layout(self.invocation_cpu.args, moved.args, "$args")
        problems += verify_layout(self.invocation_cpu.kwargs, moved.kwargs, "$kwargs")
        if problems:
            raise InvocationError("the block's inputs changed layout on the move to "
                                  f"{device}: {problems[:3]}")
        return moved

    # -- install / counts / reset -------------------------------------------

    def install(self, built: BuiltBlock, kernels: KernelSet) -> Installed:
        layer = built.module
        counters = SiteCounters()
        requested = tuple(kernels.patched)
        selected = {site: kernels.kernel_for(site) for site in requested}
        paths: dict[str, tuple[str, ...]] = {}
        attention_sites = [s for s in (SITE_QKV, SITE_ATTENTION) if s in selected]
        if attention_sites:
            install_attention_adapter(layer.self_attn, selected, counters, self.layer_index)
            for site in attention_sites:
                paths[site] = ("self_attn",)
        if SITE_MLP in selected:
            install_mlp_adapter(layer.mlp, selected, counters, self.layer_index)
            paths[SITE_MLP] = ("mlp",)
        if SITE_RESIDUAL in selected:
            setattr(layer, _TAG + "state", {
                "kernels": dict(selected), "counters": counters, "layer_idx": self.layer_index,
            })
            layer.forward = types.MethodType(_block_layer_forward, layer)
            paths[SITE_RESIDUAL] = ("post_attention_layernorm",)
        provenance = PatchProvenance(
            method="module_surgery", requested_sites=tuple(sorted(requested)),
            actual_sites=tuple(sorted(paths)), paths=paths,
        )
        return Installed(provenance=provenance, counters=counters)

    def expected_invocations(self, kernels: KernelSet) -> dict[str, int]:
        """Once per live site: the sites asked for, plus the ones that share an
        adapter with them and therefore run through it too."""
        return {site: 1 for site in live_sites(kernels.patched)}

    def reset(self, built: BuiltBlock) -> None:
        # No carried state: the block ends with its own residual add. The
        # executor clears gradients and counters; the adapters keep nothing
        # between calls that a repetition could see.
        return None

    # -- checks the model package owns ---------------------------------------

    def local_checks(self, built: BuiltBlock, kernels: KernelSet,
                     invocation: BlockInvocation) -> dict[str, Any]:
        """Every live invocation shadow-checked against its declaration.

        The same per-invocation validator the model scope runs (outputs and
        emitted gradients against the declared reference, at the declared
        tolerance), on the block's own tensors; the plan expects each live site
        once. Untimed.
        """
        from evograd.benchmark import get_task

        from .boundary import BoundaryReport, SitePlan, declared_case_for, make_validator
        from .gate import _boundary_reason

        report = BoundaryReport()
        registry = self.registry

        def op_lookup(site: str):
            return get_task(registry.require(site).op)

        set_tap(built.module, make_validator(
            op_lookup, workload_case=lambda op: declared_case_for(op, self.dtype), report=report))
        try:
            built.module.zero_grad(set_to_none=True)
            fresh = invocation.fresh()
            result = built.module(*fresh.args, **fresh.kwargs)
            torch.autograd.backward(fresh.select_outputs(result), fresh.cotangents)
            report.finalize()
        finally:
            set_tap(built.module, None)
            built.module.zero_grad(set_to_none=True)
        patched = tuple(sorted(kernels.patched))
        plan = SitePlan(patched=patched, supporting=supporting_sites(patched),
                        expected=self.expected_invocations(kernels))
        summary = report.to_dict(plan=plan)
        summary["local_check_mode"] = "declared_tolerance_only"
        if not summary["ok"]:
            summary["reason"] = _boundary_reason(summary)
        return summary

    def capture_reference(self) -> dict[str, Any] | None:
        return self.capture

    def controls(self, kernels: KernelSet, ops: Mapping[str, Any],
                 names: tuple[str, ...]) -> dict[str, KernelSet]:
        """Trusted providers patched at exactly the candidate's sites."""
        sites = tuple(kernels.patched)
        out: dict[str, KernelSet] = {}
        for name in names:
            if name == "structural_identity":
                out[name] = restrict(structural_identity_kernels(self.registry), sites)
            elif name == "bound_pair_identity":
                out[name] = bound_pair_identity_kernels(dict(ops), sites, registry=self.registry)
            elif name == "trusted_torch_compile":
                from evograd.evaluation.tier3.providers import compiled_kernels
                out[name] = compiled_kernels(sites, self.registry)
            else:
                raise ValueError(f"unknown control {name!r}; known: structural_identity, "
                                 "bound_pair_identity, trusted_torch_compile")
        return out


# ── case construction ────────────────────────────────────────────────────────


def captured_case(artifact_path: str | Path, *, layer_index: int | None = None,
                  canonical: bool = True) -> Qwen3Block:
    """The verified layer artifact as a block case.

    ``canonical=True`` goes through ``load_canonical``, which checks both
    hashes and every identity field against the tracked snapshot and takes no
    argument that turns any of that off. ``False`` still verifies both hashes
    (a debug capture from a reduced model, as the tests use) but makes no claim
    to be the canonical execution, and the case says so in its source.
    """
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.artifact import (
        LayerArtifact, load_canonical)

    path = Path(artifact_path)
    if canonical:
        artifact = load_canonical(path, layer_index=layer_index)
    else:
        artifact = LayerArtifact.load(path)  # both hashes, always
    payload = artifact.payload
    identity = dict(payload["identity"])
    if layer_index is not None and identity["layer_index"] != layer_index:
        raise InvocationError(f"{path} captures layer {identity['layer_index']}, "
                              f"not the requested {layer_index}")
    arch = dict(payload["arch"])
    args = tuple(payload["args"])
    kwargs = dict(payload["kwargs"])
    hidden = args[0]
    dtype = str(payload["output"].dtype).removeprefix("torch.")
    invocation = BlockInvocation(
        args=args, kwargs=kwargs, differentiable_inputs=DIFFERENTIABLE_INPUTS,
        outputs=OUTPUTS, cotangents=(payload["grad_output"],), source_mode="captured",
    )
    weights = {k: v for k, v in payload["state_dict"].items()}
    case = BlockCase(
        architecture=ARCHITECTURE, architecture_revision=_revision(), block_kind=BLOCK_KIND,
        block_index=int(identity["layer_index"]), source_mode="captured",
        source={"artifact": str(path), "artifact_schema": payload.get("schema_version"),
                "content_hash": payload.get("content_hash"),
                "artifact_hash": payload.get("artifact_hash"),
                "canonical_identity_verified": bool(canonical), **identity},
        dims=_dims(arch, int(hidden.shape[0]), int(hidden.shape[1])), dtype=dtype,
        boundary=BOUNDARY, sites=dict(qwen3_sites().site_ops), excluded=EXCLUDED,
        state_contract=dict(STATE_CONTRACT),
        weights_hash=tensor_tree_hash(weights, label="weights"),
        inputs_hash=tensor_tree_hash({"args": args, "kwargs": kwargs}, label="inputs"),
        cotangents_hash=tensor_tree_hash(payload["grad_output"], label="cotangents"),
    )
    capture = {"outputs": {"result": payload["output"]},
               "input_grads": {"args[0]": payload["grad_input"]},
               "param_grads": dict(payload["param_grads"])}
    return Qwen3Block(case=case, arch=arch, layer_index=int(identity["layer_index"]),
                      dtype=dtype, weights=weights, invocation_cpu=invocation, capture=capture)


#: The config recipe's cotangent scale. A residual-stream upstream gradient is
#: small relative to the activations; ``1/sqrt(H)`` keeps the VJP in a sensible
#: bfloat16 range. A recipe choice, recorded in the case, not an observation.
COTANGENT_SCALE = "1/sqrt(H)"


def config_case(*, batch: int, seq: int, layer_index: int, dtype: str, seed: int = 0,
                arch_overrides: Mapping[str, Any] | None = None) -> Qwen3Block:
    """A block from the pinned architecture with seeded weights, inputs and cotangents.

    Needs no snapshot, no full model and no checkpoint. Everything random is
    drawn on the CPU from ``seed`` so the case is the same on every device;
    the rotary tables come from the model's own ``Qwen3RotaryEmbedding`` on
    ``position_ids = arange(T)``, which is what the full model would hand the
    layer. Weights are the layer's default initialisation under the same seed
    and are kept on the adapter so every build loads the same numbers.
    """
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.prepare import (
        build_config, build_single_layer)
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import DTYPES
    from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import QWEN3_0_6B

    arch = {**QWEN3_0_6B, **dict(arch_overrides or {})}
    if not 0 <= layer_index < int(arch["num_hidden_layers"]):
        raise ValueError(f"layer_index {layer_index} is outside the architecture's "
                         f"{arch['num_hidden_layers']} layers")
    torch_dtype = DTYPES[dtype]
    config = build_config(arch)
    hidden_size = int(arch["hidden_size"])

    with torch.random.fork_rng(devices=()):
        torch.manual_seed(seed)
        layer = build_single_layer(arch, layer_index, device="cpu", dtype=dtype)
    weights = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    del layer

    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, hidden_size, generator=generator).to(torch_dtype)
    cotangent = (torch.randn(batch, seq, hidden_size, generator=generator)
                 / math.sqrt(hidden_size)).to(torch_dtype)
    position_ids = torch.arange(seq, dtype=torch.int64).unsqueeze(0)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    rotary = Qwen3RotaryEmbedding(config=config)
    with torch.no_grad():
        cos, sin = rotary(hidden, position_ids)
    kwargs = {"attention_mask": None, "position_ids": position_ids, "past_key_values": None,
              "use_cache": False, "position_embeddings": (cos.detach(), sin.detach())}
    invocation = BlockInvocation(
        args=(hidden,), kwargs=kwargs, differentiable_inputs=DIFFERENTIABLE_INPUTS,
        outputs=OUTPUTS, cotangents=(cotangent,), source_mode="config",
    )
    recipe = {
        "recipe": "config-derived", "seed": seed,
        "weights": {"kind": "seeded_default_init", "seed": seed, "note":
                    "Qwen3DecoderLayer default initialisation under torch.manual_seed(seed) on CPU"},
        "inputs": {"hidden_states": "randn(B, T, H) from torch.Generator(seed)",
                   "position_ids": "arange(T)[None]",
                   "position_embeddings": "Qwen3RotaryEmbedding(config)(hidden_states, position_ids)",
                   "attention_mask": None, "use_cache": False},
        "cotangent": f"randn(B, T, H) * {COTANGENT_SCALE} from the same generator",
        "claim": "block-kernel correctness and performance; not observed activation distributions",
        "arch_overrides": dict(arch_overrides or {}),
    }
    case = BlockCase(
        architecture=ARCHITECTURE, architecture_revision=_revision(), block_kind=BLOCK_KIND,
        block_index=layer_index, source_mode="config", source=recipe,
        dims=_dims(arch, batch, seq), dtype=dtype, boundary=BOUNDARY,
        sites=dict(qwen3_sites().site_ops), excluded=EXCLUDED, state_contract=dict(STATE_CONTRACT),
        weights_hash=tensor_tree_hash(weights, label="weights"),
        inputs_hash=tensor_tree_hash({"args": invocation.args, "kwargs": kwargs}, label="inputs"),
        cotangents_hash=tensor_tree_hash(cotangent, label="cotangents"),
    )
    return Qwen3Block(case=case, arch=arch, layer_index=layer_index, dtype=dtype,
                      weights=weights, invocation_cpu=invocation, capture=None)


def from_args(args) -> Qwen3Block:
    """The block adapter the tier-3 CLI asked for, from parsed arguments."""
    source = getattr(args, "block_source", "config")
    layer_index = getattr(args, "layer_index", None)
    if source == "captured":
        artifact = getattr(args, "artifact", None)
        if artifact is None:
            raise ValueError("--block-source captured needs --artifact PATH")
        return captured_case(artifact, layer_index=layer_index,
                             canonical=not getattr(args, "no_canonical_check", False))
    if source == "config":
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import CANONICAL

        overrides = dict(getattr(args, "arch_overrides", None) or {})
        return config_case(
            batch=args.batch if getattr(args, "batch", None) is not None else CANONICAL.batch_size,
            seq=args.tokens if getattr(args, "tokens", None) is not None else CANONICAL.seq_len,
            layer_index=layer_index if layer_index is not None else 14,
            dtype=args.dtype if getattr(args, "dtype", None) is not None else CANONICAL.dtype,
            seed=int(getattr(args, "block_seed", 0) or 0),
            arch_overrides=overrides,
        )
    raise ValueError(f"unknown --block-source {source!r}; choose captured or config")


__all__ = [
    "ARCHITECTURE",
    "BLOCK_KIND",
    "BOUNDARY",
    "EXCLUDED",
    "OBSERVED_SUITE",
    "Qwen3Block",
    "captured_case",
    "config_case",
    "from_args",
]
