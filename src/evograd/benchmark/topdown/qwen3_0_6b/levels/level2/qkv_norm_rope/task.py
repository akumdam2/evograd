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

from evograd.benchmark.topdown import load_snapshot as _load_snapshot
from evograd.benchmark.topdown import load_snapshot_task as _snapshot_task
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op

#: The harvested workload these dims came from; also the ``Provenance``
#: model key, so the declaration and the snapshot cannot disagree about
#: which run they describe.
_WORKLOAD = "qwen3_0_6b"

_SNAPSHOT = _load_snapshot(_WORKLOAD)

#: The harvested record this declaration is derived from: the RoPE application,
#: whose two outputs *are* q and k, plus the projections and head norms that
#: feed it and the SDPA call that consumes all three.
HARVEST = _snapshot_task(_WORKLOAD, "qwen3_qkv_norm_rope")

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


def make_qwen3_qkv_norm_rope_inputs(torch, op, workload, device="cuda"):
    """Inputs at the magnitudes the observed call had, and real rotary tables.

    ``cos``/``sin`` are built from an inverse-frequency schedule rather than
    drawn at random: a random pair is not a rotation, and the case would then
    describe a computation the model never performs.
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
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.qkv_norm_rope.reference:"
        "qwen3_qkv_norm_rope_forward_ref"
    ),
    # Timed through what the model runs: RMSNorm scaled after the cast back, and
    # RoPE applied in the model dtype. The declared forward keeps both in
    # float32, which is more accurate and correspondingly slower.
    runtime_forward=(
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.qkv_norm_rope.reference:"
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
        "r = rsqrt(mean(h^2,-1)+eps) and xhat = h * r: d(norm_weight) = sum "
        "over B, T and the heads of dnorm * xhat; with g = dnorm * "
        "norm_weight, dh = r * (g - xhat * mean(g * xhat, -1)). Finally the three "
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
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    make_inputs=make_qwen3_qkv_norm_rope_inputs,
)
