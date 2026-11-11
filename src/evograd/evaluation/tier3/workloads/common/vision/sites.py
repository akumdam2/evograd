"""Where a vision workload's Level-2 kernels go: forward rebinding, never new modules.

Module boundaries (``operators.bindings``) rebind their owner's forward. One
owner can hold several sites: a ``ConvNextV2Layer`` holds both
``dwconv_layernorm`` and ``convnext_mlp`` and runs natively whichever of the
two is not patched. A residual stream (``operators.residual_streams``) rebinds
its blocks and its final norm: each block adds and normalizes through the
kernel twice, after its attention and after its MLP, and hands the second
normalized sum to the next block (or the final norm) with the sum itself,
which is what flows between blocks as before.
"""

import types

from evograd.benchmark.topdown.common.vision.operators import bindings, residual_streams


def base_site(site):
    """``dwconv_layernorm_stage2`` -> ``dwconv_layernorm`` (the task's site)."""
    return site.split("_stage", 1)[0]


def _site(binding):
    if ".stages." in binding.path:
        return f"{binding.site}_stage{binding.path.split('.stages.')[1].split('.')[0]}"
    return binding.site


def _convnext_layer(chosen):
    dw, mlp = chosen.get("dwconv_layernorm"), chosen.get("convnext_mlp")

    def forward(self, features):
        h = dw[0](*dw[1]((features,), {})) if dw else \
            self.layernorm(self.dwconv(features).permute(0, 2, 3, 1))
        h = mlp[0](*mlp[1]((h,), {})) if mlp else self.pwconv2(self.grn(self.act(self.pwconv1(h))))
        return features + self.drop_path(h.permute(0, 3, 1, 2))

    return forward


def _module_call(chosen):
    ((fn, pack),) = chosen.values()

    def forward(self, x):
        return fn(*pack((x,), {}))

    return forward


def _vision_attention(chosen):
    ((fn, pack),) = chosen.values()

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None,
                **kwargs):
        return fn(*pack((hidden_states,), {"cu_seqlens": cu_seqlens,
                                           "position_embeddings": position_embeddings}))

    return forward


def _text_attention(chosen):
    ((fn, pack),) = chosen.values()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                output_attentions=False, use_cache=False, position_embeddings=None, **kwargs):
        if past_key_values is not None or use_cache:
            raise NotImplementedError("the text attention site is a training-step boundary; "
                                      "cache-enabled execution is not supported")
        return fn(*pack((hidden_states,), {"attention_mask": attention_mask,
                                           "position_embeddings": position_embeddings})), None

    return forward


ADAPTERS = {"ConvNextV2Layer": _convnext_layer, "Qwen2MLP": _module_call,
            "VisionMlp": _module_call, "VisionAttention": _vision_attention,
            "Qwen2VLAttention": _text_attention}


class Handoff:
    """The next norm's output, computed by a block with the sum that feeds it."""

    def __init__(self):
        self.pending = None

    def put(self, normed, summed):
        self.pending = (normed, summed)

    def take(self, summed):
        if self.pending is None or self.pending[1] is not summed:
            raise RuntimeError("a residual-stream norm received a tensor its block did not hand "
                               "over; the model loop is not the one this adapter was built for")
        normed, self.pending = self.pending[0], None
        return normed


def _vision_block(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """``Qwen2VLVisionBlock.forward`` with both adds fused into the norms after them."""
    fuse, handoff, first, after = self._evograd_residual
    if first:
        handoff.pending = None
    normed = self.norm1(hidden_states) if first else handoff.take(hidden_states)
    branch = self.attn(normed, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
                       **kwargs)
    normed, hidden_states = fuse(branch, hidden_states, self.norm2)
    normed, hidden_states = fuse(self.mlp(normed), hidden_states, after)
    handoff.put(normed, hidden_states)
    return hidden_states


def _decoder_layer(self, hidden_states, attention_mask=None, position_ids=None,
                   past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
    """``Qwen2VLDecoderLayer.forward`` with both adds fused into the norms after them."""
    fuse, handoff, first, after = self._evograd_residual
    if first:
        handoff.pending = None
    normed = self.input_layernorm(hidden_states) if first else handoff.take(hidden_states)
    branch, _ = self.self_attn(hidden_states=normed, attention_mask=attention_mask,
                               position_ids=position_ids, past_key_values=past_key_values,
                               use_cache=use_cache, position_embeddings=position_embeddings,
                               **kwargs)
    normed, hidden_states = fuse(branch, hidden_states, self.post_attention_layernorm)
    normed, hidden_states = fuse(self.mlp(normed), hidden_states, after)
    handoff.put(normed, hidden_states)
    return hidden_states


BLOCKS = {"vision_residual_layernorm": _vision_block, "text_residual_rmsnorm": _decoder_layer}


def _install_stream(stream, fn):
    """Rebind the stream's blocks and final norm; the paths of its boundaries' norms."""
    handoff = Handoff()

    def fuse(branch, residual, norm):
        return fn(branch, residual, *stream.norm_args(norm))

    norms = [norm for _, norm in stream.norms]
    for i, (_, block) in enumerate(stream.blocks):
        block._evograd_residual = (fuse, handoff, i == 0, norms[2 * i + 2])
        block.forward = types.MethodType(BLOCKS[stream.site], block)
    norms[-1].forward = types.MethodType(lambda self, x: handoff.take(x), norms[-1])
    return [path for path, _ in stream.norms[1:]]


def install(model, kernels):
    """Replace forward methods only; retain parameter names, identity and ties."""
    from evograd.evaluation.tier3.patch import PatchProvenance

    paths, owners = {}, {}
    for binding in bindings(model):
        site = _site(binding)
        if site in kernels.patched:
            owners.setdefault(binding.path, (binding.owner, {}))[1][binding.site] = (
                kernels.kernel_for(site), binding.pack)
            paths.setdefault(site, []).append(binding.path)
    for owner, chosen in owners.values():
        owner.forward = types.MethodType(ADAPTERS[type(owner).__name__](chosen), owner)
    for stream in residual_streams(model):
        if stream.site in kernels.patched:
            paths[stream.site] = _install_stream(stream, kernels.kernel_for(stream.site))
    if set(paths) != set(kernels.patched):
        raise ValueError(f"unmatched requested sites: {set(kernels.patched)-set(paths)}")
    return PatchProvenance(
        "module_surgery",
        tuple(kernels.patched),
        tuple(sorted(paths)),
        {key: tuple(value) for key, value in paths.items()},
    )


def site_dims(spec, site):
    """The dims ``site`` runs at, from the architecture and the execution configuration."""
    arch, base = spec.arch, base_site(site)
    if spec.model == "convnextv2_tiny":
        stage = int(site.rsplit("_stage", 1)[1])
        width, side = arch["hidden_sizes"][stage], spec.image_size // (4 * 2**stage)
        if base == "dwconv_layernorm":
            return dict(B=spec.batch_size, C=width, Y=side, X=side, K=7)
        return dict(B=spec.batch_size, Y=side, X=side, H=width, I=4 * width)
    vision, hidden = arch["vision_config"], arch["hidden_size"]
    embed, head = vision["embed_dim"], hidden // arch["num_attention_heads"]
    tokens = spec.batch_size * (spec.image_size // vision["patch_size"]) ** 2
    text = dict(B=spec.batch_size, T=spec.seq_len, H=hidden)
    return {
        "text_mlp": {**text, "I": arch["intermediate_size"]},
        "text_attention_block": {**text, "KV": arch["num_key_value_heads"] * head, "D": head},
        "text_residual_rmsnorm": text,
        "vision_mlp": dict(T=tokens, H=embed, I=int(embed * vision.get("mlp_ratio", 4))),
        "vision_attention": dict(T=tokens, H=embed, QKV=3 * embed,
                                 D=embed // vision["num_heads"]),
        "vision_residual_layernorm": dict(T=tokens, H=embed),
    }[base]


def preflight_cases(spec, site):
    """Use matching observed layouts, otherwise explicit configuration-derived shapes.

    A custom batch/image/text size must never reuse the canonical site's grid.
    These fallback cases are for verification, not observed timing suites; a
    matching snapshot that predates the site's task has none of its own.
    """
    from evograd.benchmark.topdown import has_snapshot, load_snapshot
    from evograd.benchmark.topdown.common.vision.contracts import observed
    from evograd.opdecl import Workload, Provenance

    dims = site_dims(spec, site)
    if has_snapshot(spec.model) and load_snapshot(spec.model)["workload_id"] == spec.workload_id:
        harvested = observed(spec.model, spec.model + "_" + base_site(site))
        if harvested:
            cases = tuple(case for case in harvested if case.dims == dims)
            if not cases:
                raise ValueError(f"{site}: matching snapshot has no matching specialization")
            return cases
    return (
        Workload(
            dims=dims,
            dtype=spec.dtype,
            provenance=Provenance(
                model=spec.model,
                component=site,
                source="specification",
                note="Derived from pinned architecture plus current execution configuration; not a harvested timing case",
            ),
        ),
    )
