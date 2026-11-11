"""Executable Level-2 contracts; measured grids require a harvested snapshot."""

import torch
from evograd.opdecl import Active, Inactive, Workload, Provenance, declare_op
from evograd.benchmark.topdown import has_snapshot, load_snapshot

PREFIX = "evograd.benchmark.topdown.common.generated.model:"
#: Site -> its tensor arguments, in call order; every one is Active.
SHAPES = {
    "dit_modulation": (("c", "[B, H]"), ("weight", "[J, H]"), ("bias", "[J]")),
    "dit_mlp": (("x", "[B, T, H]"), ("w1", "[I, H]"), ("b1", "[I]"),
                ("w2", "[H, I]"), ("b2", "[H]")),
    "dit_attention": (("x", "[B, T, H]"), ("qkv_weight", "[Q, H]"), ("qkv_bias", "[Q]"),
                      ("proj_weight", "[H, H]"), ("proj_bias", "[H]")),
    "dit_adaln_residual": (("x", "[B, T, H]"), ("gate", "[B, H]"), ("branch", "[B, T, H]"),
                           ("shift", "[B, H]"), ("scale", "[B, H]")),
    "dlrm_interaction": (("dense", "[B, D]"), ("sparse", "[B, S, D]")),
}
#: Site -> its scalar arguments after the tensors, with the published model's value.
SCALARS = {"dit_attention": (("heads", 6),)}
#: Site -> the arguments that are model parameters.
PARAMETERS = {
    "dit_modulation": ("weight", "bias"),
    "dit_mlp": ("w1", "b1", "w2", "b2"),
    "dit_attention": ("qkv_weight", "qkv_bias", "proj_weight", "proj_bias"),
    "dit_adaln_residual": (),
    "dlrm_interaction": (),
}
#: Site -> its outputs, in return order.
OUTPUTS = {
    "dit_modulation": (("out", "[B, J]"),),
    "dit_mlp": (("out", "[B, T, H]"),),
    "dit_attention": (("out", "[B, T, H]"),),
    "dit_adaln_residual": (("modulated", "[B, T, H]"), ("summed", "[B, T, H]")),
    "dlrm_interaction": (("out", "[B, O]"),),
}
DIMS = {"dit_modulation": dict(B=2, H=12, J=72),
        "dit_mlp": dict(B=2, T=7, H=12, I=48),
        "dit_attention": dict(B=2, T=7, H=12, Q=36),
        "dit_adaln_residual": dict(B=2, T=7, H=12),
        "dlrm_interaction": dict(B=3, S=4, D=8, O=18)}
FAMILY = {"dit_modulation": "normalization", "dit_mlp": "mlp", "dit_attention": "attention",
          "dit_adaln_residual": "normalization", "dlrm_interaction": "interaction"}
SEMANTICS = {
    "dit_modulation": "SiLU conditioning projection to six adaLN-Zero shift, scale, and gate vectors",
    "dit_mlp": "Biased linear, tanh-approximate GELU, biased linear",
    "dit_attention": (
        "Biased qkv projection of x to [B, T, Q] (Q = 3H), viewed as [B, T, 3, heads, H/heads] "
        "and permuted to q, k, v of [B, heads, T, H/heads]; non-causal scaled dot-product "
        "attention (scale 1/sqrt(H/heads), no mask, no dropout); heads merged back to "
        "[B, T, H]; biased output projection"),
    "dit_adaln_residual": (
        "summed = x + gate[:, None] * branch; modulated = LayerNorm(summed) over H without "
        "affine parameters (eps 1e-6) * (1 + scale[:, None]) + shift[:, None]; returns "
        "(modulated, summed) IN THAT ORDER. x is the residual stream (the model lays it out "
        "channel-first, strides [T*H, 1, T]); gate, shift and scale are column slices of the "
        "modulation output, so their row stride is that output's width. summed is the next "
        "call's x; the model's add gives it x's strides"),
    "dlrm_interaction": "Dense vector and strict triangular pairwise dot products",
}
BACKWARD = {
    "dit_adaln_residual": (
        "The backward receives (dmodulated, dsummed) and returns dx, dgate, dbranch, dshift, "
        "dscale IN THIS ORDER. Both paths reach summed: its gradient is dsummed plus the "
        "LayerNorm backward of dmodulated * (1 + scale[:, None]); dx is that gradient, dbranch "
        "it times gate[:, None], dgate its product with branch summed over T, dshift and "
        "dscale dmodulated (times the normalized sum for dscale) summed over T. dsummed is "
        "zero where the sum has no other consumer (the last block's MLP residual)"),
}


def _inputs(torch_module, op, workload, device="cuda"):
    from evograd.opdecl.activity import bind_shape
    gen = torch.Generator(device="cpu").manual_seed(17)
    free = dict(getattr(workload.provenance, "free", None) or {})
    result = {}
    for arg in op.args:
        if isinstance(arg, Inactive):
            result[arg.name] = free.get(arg.name, arg.default)
            continue
        shape = bind_shape(arg.shape, workload.dims)
        result[arg.name] = torch.randn(shape, generator=gen).to(device=device, dtype=getattr(torch, workload.dtype))
    for out in op.outputs:
        result[out.grad_name] = torch.randn(bind_shape(out.shape, workload.dims), generator=gen).to(
            device=device, dtype=getattr(torch, workload.dtype))
    return result


def observed(model, name):
    """``name``'s configurations in ``model``'s tracked snapshot, with their recorded
    strides and static parameters (none without a snapshot)."""
    if not has_snapshot(model):
        return ()
    return tuple(Workload(dims=c["dims"], dtype=c["dtype"],
                          provenance=Provenance(model=model,
                                                component=(c.get("provenance") or {}).get("component",
                                                                                          name),
                                                free=dict((c.get("provenance") or {}).get("free") or {}),
                                                source="execution",
                                                note=f"Observed {c['config_id']}",
                                                strides=tuple((k, tuple(v))
                                                              for k, v in c["strides"].items()),
                                                layout="strided"))
                 for c in load_snapshot(model)["tasks"].get(name, {}).get("configurations", ()))


def declaration(model, site):
    name = model + "_" + site
    cases = observed(model, name)
    outputs = tuple(Active(n, s) for n, s in OUTPUTS[site])
    return declare_op(name=name, level=2, family=FAMILY[site],
                      forward=PREFIX + site, dims=tuple(DIMS[site]),
                      args=(tuple(Active(n, s) for n, s in SHAPES[site])
                            + tuple(Inactive(n, default=v) for n, v in SCALARS.get(site, ()))),
                      output=outputs[0] if len(outputs) == 1 else outputs,
                      parameter_args=PARAMETERS[site],
                      forward_semantics=SEMANTICS[site],
                      backward_semantics=BACKWARD.get(
                          site, "Autograd gradients for every active input and parameter"),
                      coverage=cases, benchmark=cases,
                      benchmark_suites={model + "_observed": cases},
                      make_inputs=_inputs)
