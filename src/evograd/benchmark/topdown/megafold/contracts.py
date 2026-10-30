"""MegaFold's two mixed-precision Level-2 contracts.

MegaFold trains with float32 parameters and a bfloat16 autocast around the
trunk (input embedding, templates, MSA module, pairformer); the diffusion
module, the distogram and confidence heads and the loss run with autocast off,
in float32 (``disable_*_casting`` in ``configs/megafold_1x1.yaml``). Two of its
fused boundaries therefore compute something no precision-uniform contract
describes, and each is declared here exactly as MegaFold's own PyTorch code
computes it, in both regions:

``megafold_layernorm_linear``
    ``LayernormLinear``'s torch path, ``F.linear(F.layer_norm(x, ...),
    linear_weight.T)``. Autocast runs the LayerNorm in float32 (casting a
    bfloat16 ``x`` up first) and the GEMM in bfloat16 (casting the weight
    down), so at this boundary ``x`` is float32 and ``linear_weight`` arrives
    in the GEMM's dtype, which is the case dtype: ``bfloat16`` under autocast,
    ``float32`` outside it. Instances with a linear bias (only the triangle
    multiplication's ``combined`` projection) are not this contract.

``megafold_evoattention``
    ``Attend``'s default path: ``q * scale``, ``q @ k^T``, ``+ attn_bias``,
    ``masked_fill`` of dropped keys with the dtype's most negative value, a
    float32 softmax cast back, ``@ v`` -- every step but the softmax in the
    case dtype. ``q, k, v`` keep MegaFold's layout, viewed as
    ``[B, S, N, H, D]``; the bias is one ``[B, 1, H, N, N]`` tensor shared
    across ``S`` (triangle attention repeats it over the rows) and ``key_mask``
    is MegaFold's boolean key mask, materialized (MegaFold broadcasts it over
    the rows).

Cases come from the tracked harvest snapshot (one per observed configuration,
layouts included); a machine without it sees only the small correctness
fixtures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.benchmark.topdown import has_snapshot, load_snapshot
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op

WORKLOAD = "megafold"
LAYER_NORM_EPS = 1e-5


def megafold_layernorm_linear_forward(x, weight, bias, linear_weight, eps=LAYER_NORM_EPS):
    """``F.linear(F.layer_norm(x, (K,), weight, bias, eps), linear_weight.T)`` as autocast runs it.

    The LayerNorm in float32; the GEMM in ``linear_weight``'s dtype, the
    normalized rows cast to it the way autocast casts them.
    """
    y_hat = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return F.linear(y_hat.to(linear_weight.dtype), linear_weight.T)


def megafold_evoattention_forward(q, k, v, key_mask, pair_bias):
    """``Attend``'s default path on ``[B, S, N, H, D]`` views of MegaFold's tensors."""
    b, s, n, h, d = q.shape

    def heads(t):  # [B, S, N, H, D] -> [(B S), H, N, D]: MegaFold's own view of the same memory
        return t.flatten(0, 1).permute(0, 2, 1, 3)

    qh, kh, vh = heads(q * d ** -0.5), heads(k), heads(v)
    sim = torch.einsum("b h i d, b h j d -> b h i j", qh, kh)
    sim = (sim.view(b, s, h, n, n) + pair_bias).view(b * s, h, n, n)
    sim = sim.masked_fill(~key_mask.reshape(b * s, 1, 1, n), -torch.finfo(sim.dtype).max)
    attn = sim.float().softmax(dim=-1).to(q.dtype)
    out = torch.einsum("b h i j, b h j d -> b h i d", attn, vh)
    return out.permute(0, 2, 1, 3).reshape(b, s, n, h, d)


def _observed(name: str, args: tuple[str, ...]) -> tuple[Workload, ...]:
    """One case per observed configuration of ``name``, with its recorded layout."""
    if not has_snapshot(WORKLOAD):
        return ()
    cases = []
    for c in load_snapshot(WORKLOAD)["tasks"].get(name, {}).get("configurations", ()):
        prov = c.get("provenance") or {}
        shapes = {arg: tuple(meta["shape"]) for arg, meta in zip(args, c["inputs"])}
        strides = tuple((arg, tuple(stride)) for arg, stride in sorted((c.get("strides") or {}).items())
                        if tuple(stride) != torch.empty(shapes[arg], device="meta").stride())
        cases.append(Workload(dims=c["dims"], dtype=c["dtype"],
                              provenance=Provenance(model=WORKLOAD, component=prov.get("component", name),
                                                    source="execution", strides=strides,
                                                    free=dict(prov.get("free") or {}),
                                                    note=f"Observed {c['config_id']}")))
    return tuple(cases)


def _layernorm_linear_inputs(torch_, op, workload, device="cuda"):
    m, k, n = (workload.dims[d] for d in ("M", "K", "N"))
    gen = torch.Generator(device="cpu").manual_seed((m * 131 + k) * 131 + n)
    dtype = getattr(torch, workload.dtype)
    return {"x": torch.randn((m, k), generator=gen).to(device),
            "weight": (1.0 + 0.1 * torch.randn((k,), generator=gen)).to(device),
            "bias": (0.1 * torch.randn((k,), generator=gen)).to(device),
            "linear_weight": (torch.randn((k, n), generator=gen) * k ** -0.5).to(device=device, dtype=dtype),
            "eps": LAYER_NORM_EPS,
            "dout": torch.randn((m, n), generator=gen).to(device=device, dtype=dtype)}


def _evoattention_inputs(torch_, op, workload, device="cuda"):
    b, s, h, n, d = (workload.dims[x] for x in ("B", "S", "H", "N", "D"))
    gen = torch.Generator(device="cpu").manual_seed((((b * 131 + s) * 131 + h) * 131 + n) * 131 + d)
    dtype = getattr(torch, workload.dtype)

    def rand(*shape):
        return torch.randn(shape, generator=gen).to(device=device, dtype=dtype)

    keep = torch.rand((b, s, n), generator=gen) > 0.1
    keep[..., 0] = True  # a query row with every key dropped is not a case MegaFold presents
    return {"q": rand(b, s, n, h, d), "k": rand(b, s, n, h, d), "v": rand(b, s, n, h, d),
            "key_mask": keep.to(device), "pair_bias": rand(b, 1, h, n, n), "do": rand(b, s, n, h, d)}


def _fixtures(dims_list):
    return tuple(Workload(dims=dims, dtype=dtype,
                          provenance=Provenance(model=WORKLOAD, component="fixture", source="handpicked",
                                                scaled=True, note="Reduced correctness fixture"))
                 for dims in dims_list for dtype in ("float32", "bfloat16"))


LAYERNORM_LINEAR = "megafold_layernorm_linear"
EVOATTENTION = "megafold_evoattention"


def layernorm_linear_op():
    observed = _observed(LAYERNORM_LINEAR, ("x", "weight", "bias", "linear_weight"))
    return declare_op(
        name=LAYERNORM_LINEAR, level=2, family="gemm",
        forward=f"{__name__}:megafold_layernorm_linear_forward",
        dims=("M", "K", "N"),
        args=(Active("x", "[M, K]", dtype="float32"),
              Active("weight", "[K]", dtype="float32", note="LayerNorm gamma"),
              Active("bias", "[K]", dtype="float32", note="LayerNorm beta"),
              Active("linear_weight", "[K, N]", note="in the GEMM's dtype (the case dtype)"),
              Inactive("eps", default=LAYER_NORM_EPS)),
        output=Active("out", "[M, N]"),
        parameter_args=("weight", "bias", "linear_weight"),
        forward_semantics=(
            "MegaFold's LayernormLinear under its training precision: y_hat = layer_norm(x, weight, "
            "bias, eps) over the last axis, computed in float32 from float32 x; out = "
            "y_hat.to(case dtype) @ linear_weight with float32 accumulation, out [M, N] in the case "
            "dtype. The case dtype is the GEMM's: bfloat16 under MegaFold's autocast (linear_weight "
            "arrives already cast to it), float32 where autocast is off. x, weight, bias are float32."),
        backward_semantics=(
            "Return (dx, dweight, dbias, dlinear_weight), each in its input's dtype, as autograd "
            "differentiates the forward above: the GEMM backward in the case dtype (dout and "
            "dlinear_weight are in it), the LayerNorm backward in float32."),
        correctness=_fixtures((dict(M=64, K=64, N=128), dict(M=129, K=128, N=16))),
        coverage=observed, benchmark=observed, benchmark_suites={f"{WORKLOAD}_observed": observed},
        tolerances={"float32": (1e-4, 1e-4), "bfloat16": (2e-2, 2e-2)},
        make_inputs=_layernorm_linear_inputs)


def evoattention_op():
    observed = _observed(EVOATTENTION, ("q", "k", "v", "key_mask", "pair_bias"))
    return declare_op(
        name=EVOATTENTION, level=2, family="attention",
        forward=f"{__name__}:megafold_evoattention_forward",
        dims=("B", "S", "H", "N", "D"),
        args=(Active("q", "[B, S, N, H, D]"), Active("k", "[B, S, N, H, D]"),
              Active("v", "[B, S, N, H, D]"),
              Inactive("key_mask", "[B, S, N]", dtype="bool", note="True = attend to this key"),
              Active("pair_bias", "[B, 1, H, N, N]", grad="d_pair_bias",
                     note="shared across S; d_pair_bias sums over S")),
        output=Active("o", "[B, S, N, H, D]"),
        parameter_args=(),
        forward_semantics=(
            "MegaFold's Attend default path, per (B, S, head), in the case dtype except the softmax: "
            "q is scaled by D**-0.5; scores = q @ k^T over the key axis N; scores += pair_bias "
            "(broadcast over S); keys with key_mask False are set to the dtype's most negative finite "
            "value (masked_fill, not an additive mask); probabilities = softmax over keys computed in "
            "float32, cast to the case dtype; o = probabilities @ v. Inputs and output are "
            "[B, S, N, H, D] (layouts as MegaFold presents them, possibly strided)."),
        backward_semantics=(
            "Return (dq, dk, dv, d_pair_bias): dq/dk/dv [B, S, N, H, D] in the case dtype, "
            "d_pair_bias [B, 1, H, N, N] summed over S, as autograd differentiates the forward above."),
        correctness=_fixtures((dict(B=1, S=1, H=4, N=23, D=8), dict(B=1, S=3, H=4, N=64, D=16))),
        coverage=observed, benchmark=observed, benchmark_suites={f"{WORKLOAD}_observed": observed},
        tolerances={"float32": (1e-4, 1e-4), "bfloat16": (4e-2, 4e-2)},
        tolerance_multipliers={"d_pair_bias": (2.0, 1.0)},
        make_inputs=_evoattention_inputs)
