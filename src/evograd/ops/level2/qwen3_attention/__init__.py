"""Operator declaration: Qwen3's causal grouped-query attention plus output projection.

The second task derived from the observed Qwen3-0.6B training step. The harvest
found one SDPA configuration running 28 times, once per decoder layer, and one
``o_proj`` configuration running alongside it; together they are the half of
``Qwen3Attention`` that follows the projections.

**Boundary, stated once and unambiguously.** This task is

    q, k, v
      -> F.scaled_dot_product_attention(attn_mask=None, dropout_p=0.0,
                                        is_causal=True, scale=1/sqrt(D),
                                        enable_gqa=True)
      -> transpose heads and tokens, restore [B, T, HQ*D]
      -> F.linear(..., o_weight)
      -> out [B, T, H]

and it deliberately **excludes** ``q_proj``, ``k_proj``, ``v_proj``, the Q/K
RMSNorm over the head dimension, and the rotary embedding. Those are a separate
boundary with a separate cost, and they will become ``qwen3_qkv_norm_rope``.
q, k and v arrive here already projected, already normalized and already
rotated, which is exactly the state the observed SDPA call received them in.

**Layout is part of the contract.** The model reaches this call by projecting
into ``[B, T, heads, D]`` and transposing, so q, k and v are non-contiguous with
head-major strides. ``make_qwen3_attention_inputs`` reproduces that by building
token-major tensors and transposing them, rather than allocating contiguous
substitutes that would benchmark a different access pattern.
"""

import math

from evograd.bench.workloads import load_snapshot as _load_snapshot
from evograd.bench.workloads import load_snapshot_task as _snapshot_task
from evograd.opdecl import Active, Provenance, Workload, declare_op
from evograd.opdecl.tolerance import ReductionScaledAtol

#: The harvested workload these dims came from; also the ``Provenance``
#: model key, so the declaration and the snapshot cannot disagree about
#: which run they describe.
_WORKLOAD = "qwen3_0_6b"

_SNAPSHOT = _load_snapshot(_WORKLOAD)

#: The harvested record this declaration is derived from: the SDPA
#: configuration, plus the output projection that completes the boundary.
HARVEST = _snapshot_task(_WORKLOAD, "qwen3_attention")

#: 28 -- once per decoder layer, for both halves of the boundary.
FREQUENCY = HARVEST["frequency"]

PROVENANCE_CHAIN = (
    f"canonical workload {_SNAPSHOT['workload_id']}",
    f"harvest manifest {_SNAPSHOT['manifest_hash']}",
    f"{_SNAPSHOT['representative_layer']['module_path']}.self_attn",
    "verified standalone Qwen3DecoderLayer replay",
    f"scaled_dot_product_attention, harvested configuration {HARVEST['config_id']}",
    f"o_proj, harvested configuration "
    f"{HARVEST['supporting']['output_projection']['config_id']}",
    "qwen3_attention",
)

_Q, _K, _V = (entry["shape"] for entry in HARVEST["input_shapes"])
_BATCH, _HQ, _SEQ, _HEAD_DIM = _Q
_HKV = _K[1]
_OUT_PROJ = HARVEST["supporting"]["output_projection"]
_HIDDEN, _Q_FANOUT = _OUT_PROJ["params"]["weight"]["shape"]

#: The observed strides, kept so a test can assert the generated inputs match
#: the layout the model actually presented.
OBSERVED_STRIDES = {
    "q": tuple(HARVEST["input_shapes"][0]["stride"]),
    "k": tuple(HARVEST["input_shapes"][1]["stride"]),
    "v": tuple(HARVEST["input_shapes"][2]["stride"]),
}

assert _K == _V, HARVEST
assert _Q_FANOUT == _HQ * _HEAD_DIM, HARVEST
assert HARVEST["attrs"]["is_causal"] is True, HARVEST
assert HARVEST["attrs"]["enable_gqa"] is True, HARVEST
assert HARVEST["attrs"]["dropout_p"] == 0.0, HARVEST
assert HARVEST["attrs"]["attn_mask_provided"] is False, HARVEST
assert abs(HARVEST["attrs"]["scale"] - 1.0 / math.sqrt(_HEAD_DIM)) < 1e-12, HARVEST
assert _OUT_PROJ["attrs"]["bias"] is False, HARVEST

_DIMS = ("B", "T", "HQ", "HK", "D", "QO", "H")

_OBSERVED = Provenance(
    model="qwen3_0_6b",
    component="causal_gqa_attention",
    free={"batch": _BATCH, "seq": _SEQ},
    source="hf_config",
)

_SHRUNK = Provenance(
    model="qwen3_0_6b",
    component="causal_gqa_attention",
    free={},
    source="handpicked",
    scaled=True,
    note=(
        "head count, head dimension, sequence length and hidden size reduced "
        "from Qwen3-0.6B's 16/8 heads, 128-wide heads and 2048 tokens so the "
        "correctness cases run on CPU; the 2:1 grouped-query ratio, the causal "
        "mask, the absence of a bias and the head-major input layout are all "
        "preserved"
    ),
)

_BENCHMARK = (
    Workload(
        dims={
            "B": _BATCH,
            "T": _SEQ,
            "HQ": _HQ,
            "HK": _HKV,
            "D": _HEAD_DIM,
            "QO": _Q_FANOUT,
            "H": _HIDDEN,
        },
        dtype="bfloat16",
        provenance=_OBSERVED,
    ),
)

_CORRECTNESS = tuple(
    Workload(
        dims={
            "B": batch,
            "T": tokens,
            "HQ": heads,
            "HK": kv_heads,
            "D": head_dim,
            "QO": heads * head_dim,
            "H": hidden,
        },
        dtype=dtype,
        provenance=_SHRUNK,
    )
    for batch, tokens, heads, kv_heads, head_dim, hidden, dtype in (
        (1, 16, 4, 2, 8, 24, "float32"),
        (2, 32, 4, 2, 16, 48, "float32"),
        (2, 32, 4, 2, 16, 48, "bfloat16"),
        # A 4:1 group ratio, so a kernel that assumed 2:1 fails here.
        (1, 24, 8, 2, 16, 32, "bfloat16"),
    )
)


#: The bf16 correctness case the declared multipliers were measured on.
_ANCHOR = {"B": 2, "T": 32, "HQ": 4, "HK": 2, "D": 16, "QO": 64, "H": 48}

_REDUCTION_SCALED = ReductionScaledAtol(
    anchor_dims=_ANCHOR,
    reduction_dims={},
    result_dims={
        "out": ("B", "T", "H"),
        "dq": ("B", "HQ", "T", "D"),
        "dk": ("B", "HK", "T", "D"),
        "dv": ("B", "HK", "T", "D"),
        "do_weight": ("H", "QO"),
    },
    gain=2.5,
)


def make_qwen3_attention_inputs(torch, op, workload, device="cuda"):
    """Head-major q/k/v with the observed non-contiguous strides.

    Built token-major and transposed, which is how the model produces them --
    ``self.q_proj(x).view(B, T, -1, D).transpose(1, 2)``. Allocating contiguous
    tensors instead would silently benchmark a different memory access pattern
    than the one the observed call had.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["HQ"] * 1009 + dims["D"] * 101
    )

    def head_major(heads):
        token_major = torch.randn(
            (dims["B"], dims["T"], heads, dims["D"]), device=device, dtype=dtype
        )
        return token_major.transpose(1, 2)

    q = head_major(dims["HQ"])
    k = head_major(dims["HK"])
    v = head_major(dims["HK"])
    o_weight = (
        torch.randn((dims["H"], dims["QO"]), device=device, dtype=torch.float32)
        * dims["QO"] ** -0.5
    ).to(dtype)
    dout = torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype)
    return {"q": q, "k": k, "v": v, "o_weight": o_weight, "dout": dout}


op = declare_op(
    name="qwen3_attention",
    level=2,
    family="attention",
    forward="evograd.ops.level2.qwen3_attention.forward_ref:qwen3_attention_forward_ref",
    # The eager baseline is timed through the SDPA branch the model runs. The
    # declared forward materializes a [B, HQ, T, T] score matrix -- 512 MiB at
    # the observed shape -- which the real execution never allocates, so timing
    # against it would compare every candidate to a strawman.
    runtime_forward=(
        "evograd.ops.level2.qwen3_attention.forward_ref:"
        "qwen3_attention_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("o_weight", "[H, QO]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("o_weight",),
    forward_semantics=(
        "Causal grouped-query attention followed by the output projection. "
        "Expand k and v from HK to HQ heads by repeating each KV head HQ/HK "
        "times; scores = q @ k_expanded^T * (1/sqrt(D)); mask strictly-future "
        "positions to -inf; softmax over the last axis in float32 and cast back "
        "to q's dtype; attn = weights @ v_expanded, shape [B, HQ, T, D]; "
        "transpose heads and tokens, make contiguous and reshape to [B, T, QO] "
        "where QO = HQ*D; out = merged @ o_weight^T, shape [B, T, H]. "
        "There is no attention mask tensor, no dropout, no bias and no KV "
        "cache. Do not call F.scaled_dot_product_attention, F.linear, "
        "torch.matmul, @, F.softmax, or autograd in the generated math."
    ),
    backward_semantics=(
        "Return gradients for q, k, v and o_weight IN THIS ORDER. With "
        "merged = attn.transpose(1,2).reshape(B, T, QO): "
        "do_weight = dout^T @ merged summed over B and T (shape [H, QO]); "
        "dmerged = dout @ o_weight, reshaped and transposed back to "
        "[B, HQ, T, D]; then the standard attention backward -- "
        "dv_expanded = weights^T @ dmerged, dweights = dmerged @ v_expanded^T, "
        "dscores = weights * (dweights - sum(dweights * weights, dim=-1, "
        "keepdim=True)) with future positions zeroed, "
        "dq = (dscores @ k_expanded) / sqrt(D) and "
        "dk_expanded = (dscores^T @ q) / sqrt(D). Finally sum dk_expanded and "
        "dv_expanded over each group of HQ/HK query heads to get dk and dv at "
        "HK heads. Accumulate every reduction and matmul in float32 before "
        "casting each gradient to its input's dtype."
    ),
    extra_constraints=(
        "Observed from a real Qwen3-0.6B training step, not chosen. The "
        "boundary starts after the q_proj/k_proj/v_proj projections, the Q/K "
        "head-dimension "
        "RMSNorms and the rotary embedding, and ends after o_proj -- those "
        "earlier stages are a separate task and must not be recomputed here. "
        "q, k and v are non-contiguous with head-major strides "
        "(q: [B*T*HQ*D, D, HQ*D, 1]) because the model transposes them out of "
        "[B, T, heads, D]; a kernel may make them contiguous internally but the "
        "declared inputs are not. HQ must be divisible by HK. o_weight has no "
        "bias. All floating tensors are CUDA tensors."
    ),
    grad_order=("dq", "dk", "dv", "do_weight"),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("q", "k", "v", "o_weight"),
    # Measured, not chosen. `evograd.bench.workloads.qwen3.levels.level2.attention calibrate`
    # compares the declared dense float32-softmax forward against
    # `runtime_forward` -- the SDPA branch the model runs -- on every
    # correctness workload and on the canonical [2, 16, 2048, 128] invocation,
    # and reports the smallest base `t` for which `allclose(atol=ma*t, rtol=t)`
    # accepts each result.
    #
    # float32: worst 4.0e-07 across every case, so the repository's ordinary
    # float32 pair clears it by ~50x.
    #
    # bfloat16, worst over all cases: out 5.8e-03, dq 4.6e-03, dk 5.8e-03,
    # dv 1.5e-08 (SDPA and the dense spelling compute dv identically),
    # do_weight 2.9e-02. Base 1e-02 therefore leaves ~1.7x on the binding
    # non-multiplied result.
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (1e-2, 1e-2)},
    # Only `do_weight` needs one. It reduces over all B*T tokens and cancels,
    # which is exactly what an atol multiplier is for; every other result
    # measured a minimum multiplier of 1.00 at base 1e-2 and so has none.
    #
    #   result       measured min ma at t=1e-2   declared
    #   do_weight                3.54              5.4
    # do_weight was 5.4 and needed 1.58x more at the observed shape than the
    # element-count term alone supplies. Raised to 6.5, which costs the
    # correctness grid a 1.2x looser gate for this one gradient (its measured
    # margin there goes 1.47x -> 1.77x) and buys 1.29x at the observed shape.
    tolerance_multipliers={"do_weight": (6.5, 1.0)},
    # Deliberately *no* reduction term. do_weight does contract over 4096
    # tokens, but the measured exponent of its required atol against reduction
    # length is 0.032 -- flat -- because attention's output is a softmax-weighted
    # average whose scale does not grow with the sum. Declaring a sqrt(N) term
    # here would loosen the gate about tenfold for no measured reason. Only the
    # element-count term applies.
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_qwen3_attention_inputs,
)
