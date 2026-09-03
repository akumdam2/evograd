"""Operator declaration: Qwen3's q/k/v projection, per-head RMSNorm and RoPE.

The third task derived from the observed Qwen3-0.6B training step, and the first
with more than one output. It is the *prefix* of ``Qwen3Attention``: everything
that happens before the attention itself.

**Boundary, stated once and unambiguously.**

    normalized hidden_states
      -> separate q_proj, k_proj, v_proj
      -> reshape by heads
      -> per-head Q/K RMSNorm over the head dimension
      -> transpose to head-major
      -> apply RoPE to q and k
      -> return (q, k, v)

``F.scaled_dot_product_attention`` and ``o_proj`` are **not** part of it -- those
are ``qwen3_attention``, and the two tasks meet exactly where this one's outputs
become that one's inputs. The residual RMSNorm that produces ``x`` is also
outside; it belongs to a later ``residual_rmsnorm`` task.

``cos`` and ``sin`` are Inactive: the rotary embedding module computes them once
per step from position ids and shares them across all 28 layers, so within this
boundary they are tables, not activations, and they receive no gradient.
"""

from evograd.bench.workloads.qwen3.harvest.snapshot import load as _load_snapshot
from evograd.bench.workloads.qwen3.harvest.snapshot import task as _snapshot_task
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.opdecl.tolerance import ReductionScaledAtol

_SNAPSHOT = _load_snapshot()

#: The harvested record this declaration is derived from: the RoPE application,
#: whose two outputs *are* q and k, plus the projections and head norms that
#: feed it and the SDPA call that consumes all three.
HARVEST = _snapshot_task("qwen3_qkv_norm_rope")

#: 28 -- once per decoder layer, for every configuration in the boundary.
FREQUENCY = HARVEST["frequency"]

_SUP = HARVEST["supporting"]

PROVENANCE_CHAIN = (
    f"canonical workload {_SNAPSHOT['workload_id']}",
    f"harvest manifest {_SNAPSHOT['manifest_hash']}",
    f"{_SNAPSHOT['representative_layer']['module_path']}.self_attn",
    "verified standalone Qwen3DecoderLayer replay",
    f"q_proj {_SUP['q_projection']['config_id']}, "
    f"k_proj/v_proj {_SUP['kv_projection']['config_id']}",
    f"q_norm {_SUP['q_norm']['config_id']}, k_norm {_SUP['k_norm']['config_id']}",
    f"apply_rotary_pos_emb {HARVEST['config_id']}",
    "qwen3_qkv_norm_rope",
)

# Every dimension is read off the harvest rather than written down twice.
_Q_OUT, _K_OUT = HARVEST["output_shapes"][0], HARVEST["output_shapes"][1]
_BATCH, _HQ, _SEQ, _HEAD_DIM = _Q_OUT["shape"]
_HKV = _K_OUT["shape"][1]
_HIDDEN = _SUP["q_projection"]["input_shapes"][0]["shape"][-1]
_Q_FANOUT = _SUP["q_projection"]["params"]["weight"]["shape"][0]
_KV_FANOUT = _SUP["kv_projection"]["params"]["weight"]["shape"][0]
_EPS = _SUP["q_norm"]["attrs"]["eps"]
#: v never passes through RoPE, so its observed layout comes from the SDPA call
#: that consumes it rather than from a record of its own.
_V_OBSERVED = _SUP["consumer"]["input_shapes"][2]

#: The observed output strides, so a test can assert the generated and produced
#: layouts are the ones the model actually had.
OBSERVED_STRIDES = {
    "q": tuple(_Q_OUT["stride"]),
    "k": tuple(_K_OUT["stride"]),
    "v": tuple(_V_OBSERVED["stride"]),
}

assert _Q_FANOUT == _HQ * _HEAD_DIM, HARVEST
assert _KV_FANOUT == _HKV * _HEAD_DIM, HARVEST
assert _V_OBSERVED["shape"] == _K_OUT["shape"], HARVEST
assert _SUP["k_norm"]["attrs"]["eps"] == _EPS, HARVEST
assert _SUP["q_norm"]["attrs"]["normalized_size"] == _HEAD_DIM, HARVEST
assert _SUP["q_projection"]["attrs"]["bias"] is False, HARVEST
assert _SUP["kv_projection"]["attrs"]["bias"] is False, HARVEST
assert HARVEST["attrs"]["unsqueeze_dim"] == 1, HARVEST

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_OBSERVED = Provenance(
    model="qwen3_0_6b",
    component="qkv_norm_rope",
    free={"batch": _BATCH, "seq": _SEQ},
    source="hf_config",
)

_SHRUNK = Provenance(
    model="qwen3_0_6b",
    component="qkv_norm_rope",
    free={},
    source="handpicked",
    scaled=True,
    note=(
        "hidden size, head count, head dimension and sequence length reduced "
        "from Qwen3-0.6B's 1024/16-8 heads/128-wide heads/2048 tokens so the "
        "correctness cases run on CPU; the grouped-query ratio, the absence of "
        "projection biases, the per-head norm width and the head-major output "
        "layout are all preserved"
    ),
)

_BENCHMARK = (
    Workload(
        dims={
            "B": _BATCH,
            "T": _SEQ,
            "H": _HIDDEN,
            "HQ": _HQ,
            "HK": _HKV,
            "D": _HEAD_DIM,
            "QO": _Q_FANOUT,
            "KVO": _KV_FANOUT,
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
            "H": hidden,
            "HQ": heads,
            "HK": kv_heads,
            "D": head_dim,
            "QO": heads * head_dim,
            "KVO": kv_heads * head_dim,
        },
        dtype=dtype,
        provenance=_SHRUNK,
    )
    for batch, tokens, hidden, heads, kv_heads, head_dim, dtype in (
        (1, 16, 32, 4, 2, 8, "float32"),
        (2, 32, 64, 4, 2, 16, "float32"),
        (2, 32, 64, 4, 2, 16, "bfloat16"),
        # 4:1 grouping, so a kernel that assumed 2:1 fails here.
        (1, 24, 64, 8, 2, 16, "bfloat16"),
    )
)


#: The bf16 correctness case the declared multipliers were measured on.
_ANCHOR = {"B": 2, "T": 32, "H": 64, "HQ": 4, "HK": 2, "D": 16, "QO": 64, "KVO": 32}

#: The per-head norm weights are the longest contractions in the operator: a
#: single [D] vector accumulating every token of every query head. Their
#: reduction therefore includes the head axis, which is why they cannot share
#: the projections' term.
_REDUCTION_SCALED = ReductionScaledAtol(
    anchor_dims=_ANCHOR,
    reduction_dims={
        "dq_weight": ("B", "T"),
        "dk_weight": ("B", "T"),
        "dv_weight": ("B", "T"),
        "dq_norm_weight": ("B", "T", "HQ"),
        "dk_norm_weight": ("B", "T", "HK"),
    },
    result_dims={
        "q": ("B", "HQ", "T", "D"),
        "k": ("B", "HK", "T", "D"),
        "v": ("B", "HK", "T", "D"),
        "dx": ("B", "T", "H"),
        "dq_weight": ("QO", "H"),
        "dk_weight": ("KVO", "H"),
        "dv_weight": ("KVO", "H"),
        "dq_norm_weight": ("D",),
        "dk_norm_weight": ("D",),
    },
    gain=2.0,
)


def make_qwen3_qkv_norm_rope_inputs(torch, op, workload, device="cuda"):
    """Inputs at the magnitudes the observed call had, and real rotary tables.

    ``cos``/``sin`` are built from an inverse-frequency schedule rather than
    drawn at random: a random pair is not a rotation, and the reference's
    accuracy -- and therefore the calibrated tolerance -- would then describe a
    computation the model never performs.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["H"] * 1009 + dims["D"]
    )
    hidden, head_dim = dims["H"], dims["D"]

    def projection(out_features):
        return (
            torch.randn((out_features, hidden), device=device, dtype=torch.float32)
            * hidden**-0.5
        ).to(dtype)

    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (
        1000000.0
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None, :, :]
    return {
        "x": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"]),
        "k_weight": projection(dims["KVO"]),
        "v_weight": projection(dims["KVO"]),
        # Learned norm scales start at 1 in a Qwen3 checkpoint and stay near it.
        "q_norm_weight": (
            1.0 + 0.02 * torch.randn((head_dim,), device=device, dtype=torch.float32)
        ).to(dtype),
        "k_norm_weight": (
            1.0 + 0.02 * torch.randn((head_dim,), device=device, dtype=torch.float32)
        ).to(dtype),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": 1e-6,
        "dq": torch.randn(
            (dims["B"], dims["HQ"], dims["T"], head_dim), device=device, dtype=dtype
        ),
        "dk": torch.randn(
            (dims["B"], dims["HK"], dims["T"], head_dim), device=device, dtype=dtype
        ),
        "dv": torch.randn(
            (dims["B"], dims["HK"], dims["T"], head_dim), device=device, dtype=dtype
        ),
    }


op = declare_op(
    name="qwen3_qkv_norm_rope",
    level=2,
    family="attention",
    forward=(
        "evograd.ops.level2.qwen3_qkv_norm_rope.forward_ref:"
        "qwen3_qkv_norm_rope_forward_ref"
    ),
    # Timed through what the model runs: RMSNorm scaled after the cast back, and
    # RoPE applied in the model dtype. The declared forward keeps both in
    # float32, which is more accurate and correspondingly slower.
    runtime_forward=(
        "evograd.ops.level2.qwen3_qkv_norm_rope.forward_ref:"
        "qwen3_qkv_norm_rope_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("q_norm_weight", "[D]"),
        Active("k_norm_weight", "[D]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
        Inactive("eps", None, default=_EPS),
    ),
    output=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
    ),
    parameter_args=(
        "q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight",
    ),
    forward_semantics=(
        "Project the normalized residual stream three ways, normalize the query "
        "and key heads, and rotate them. q_flat = x @ q_weight^T (shape "
        "[B, T, QO]); reshape to [B, T, HQ, D]; per-head RMSNorm over the last "
        "axis with variance computed in float32: "
        "h * rsqrt(mean(h^2, -1) + eps) * q_norm_weight; transpose to "
        "[B, HQ, T, D]. The same for k with k_weight and k_norm_weight at HK "
        "heads. v = x @ v_weight^T reshaped to [B, T, HK, D] and transposed, "
        "with no normalization. Then RoPE on q and k only: with cos and sin "
        "unsqueezed at dim 1 and rotate_half(t) = cat((-t[..., D/2:], "
        "t[..., :D/2]), -1), t_out = t*cos + rotate_half(t)*sin. Return "
        "(q, k, v) IN THAT ORDER. There are no projection biases and no "
        "attention here. Do not call F.linear, torch.matmul, @, F.rms_norm, or "
        "autograd in the generated math."
    ),
    backward_semantics=(
        "Return gradients for x, q_weight, k_weight, v_weight, q_norm_weight "
        "and k_norm_weight IN THIS ORDER. The backward receives "
        "output_grads = (dq, dk, dv), one per output, and every one of them "
        "contributes to dx. cos and sin are inactive and get no gradient. "
        "Unrotate first: for q, dq_norm = dq*cos - rotate_half(dq*sin) using "
        "the same rotate_half, because rotate_half is its own negative inverse; "
        "likewise for k. Then the RMSNorm backward per head with "
        "r = rsqrt(mean(h^2,-1)+eps): d(norm_weight) = sum over B and T of "
        "dnorm * h * r; dh = norm_weight * r * (dnorm - h * "
        "mean(dnorm * norm_weight * h, -1) * r^2). Finally the three "
        "projections: dq_weight = dq_flat^T @ x summed over B and T, and "
        "dx = dq_flat @ q_weight + dk_flat @ k_weight + dv_flat @ v_weight. "
        "Accumulate every reduction and matmul in float32 before casting each "
        "gradient to its input's dtype."
    ),
    extra_constraints=(
        "Observed from a real Qwen3-0.6B training step, not chosen. This is the "
        "prefix of Qwen3Attention: it ends with (q, k, v) ready for "
        "scaled_dot_product_attention, and it does not contain SDPA or o_proj -- "
        "those are the qwen3_attention task. The residual RMSNorm that produces "
        "x is also outside it. cos and sin are shared tables computed once per "
        "step, so they are inputs and receive no gradient. The outputs are "
        "non-contiguous head-major views "
        "(q stride [B*T*HQ*D, D, HQ*D, 1]) because the model reaches them by "
        "view-then-transpose; a kernel may work in any internal layout but must "
        "return that one. HQ must be divisible by HK, and both projection "
        "fan-outs must be multiples of D. No biases."
    ),
    grad_order=(
        "dx",
        "dq_weight",
        "dk_weight",
        "dv_weight",
        "dq_norm_weight",
        "dk_norm_weight",
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    # Measured, not chosen. `evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope calibrate`
    # compares the declared float32 reference against `runtime_forward` -- the
    # spelling the model runs, and therefore the smallest disagreement any
    # correct implementation can have with the oracle -- on every correctness
    # workload and on the canonical invocation, and reports the smallest base
    # `t` for which `allclose(atol=ma*t, rtol=t)` accepts each result.
    #
    # float32: worst 2.2e-06 across every case and every result, which the
    # repository's ordinary float32 pair clears by ~9x. `v` and `dv_weight`
    # measure exactly 0.0 in every case, at both dtypes: the value path has no
    # RMSNorm and no rotation, so the two spellings are the same computation.
    #
    # bfloat16, worst over all cases and both populations:
    #   q 1.18e-02   k 9.17e-03   v 0
    #   dx 2.03e-02  dq_weight 5.68e-02  dk_weight 5.93e-02  dv_weight 0
    #   dq_norm_weight 9.17e-02  dk_norm_weight 3.94e-02
    #
    # The base is set by the binding *forward* requirement -- `q` at 1.18e-02 --
    # rather than by the gradients, so the outputs are gated by the base alone
    # and only the reductions carry multipliers. 2e-02 leaves 1.69x on `q`.
    # Going lower would mean putting a multiplier on a forward output, which
    # hides the very number a candidate is primarily judged on.
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (2e-2, 2e-2)},
    # Each multiplier is the measured minimum at base 2e-2 times a 1.5 safety
    # margin, rounded up to one decimal. The weight gradients need the largest
    # because they reduce over all B*T tokens with cancelling signs; `q`, `k`,
    # `v` and `dv_weight` measured 1.00 and so have none.
    #
    #   result            measured min ma at t=2e-2   declared
    #   dx                          1.02                1.6
    #   dq_weight                   4.67                7.1
    #   dk_weight                   3.56                5.4
    #   dq_norm_weight              4.91                7.4
    #   dk_norm_weight              2.75                4.2
    tolerance_multipliers={
        "dx": (1.6, 1.0),
        "dq_weight": (7.1, 1.0),
        "dk_weight": (5.4, 1.0),
        "dq_norm_weight": (7.4, 1.0),
        "dk_norm_weight": (4.2, 1.0),
    },
    # Measured on a GH200 with `calibrate inventory`. At the observed shape the
    # four weight gradients needed 8.7x to 13.7x more absolute tolerance than
    # the constants allow -- they contract over 4096 tokens where the grid's
    # longest is 64 -- and `q`, `k` and `dx`, which have no token reduction,
    # needed 1.33x to 1.45x from element count alone. The hook supplies both
    # terms and is the identity at and below the anchor.
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_qwen3_qkv_norm_rope_inputs,
)
