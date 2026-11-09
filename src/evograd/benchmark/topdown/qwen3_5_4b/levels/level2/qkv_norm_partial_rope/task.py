"""Operator declaration: gated attention's q/gate/k/v projections, (1 + w) per-head
norms and partial RoPE (site ``qkv_norm_partial_rope``).

Four outputs: ``q``, ``k`` and ``v`` ready for SDPA, and the output ``gate`` that
``qwen3_5_gated_attention`` applies after it. ``cos``/``sin`` are the model's
rotary tables (64 wide for a 256-wide head, interleaved MRoPE on text positions)
and receive no gradient.
"""

from evograd.opdecl import Active, Inactive, declare_op

from ..task_tools import CONFIG, OBSERVED_SUITE, observed, seed_of, token_major


_DIMS = ("B", "T", "H", "HQ", "HK", "D", "R", "QG", "KVO", "QO")


def model_strides(dims):
    """``dv`` as the block delivers it: a ``[B, HK, T, D]`` view of a token-major
    buffer -- what the default SDPA backend's backward returns for v on the GH200
    (measured on the GH200). dq and dk arrive contiguous."""
    return (("dv", token_major(dims["B"], dims["T"], dims["HK"], dims["D"])),)


_BENCHMARK = observed("qkv_norm_partial_rope", strides=model_strides)


def make_qkv_norm_partial_rope_inputs(torch, op, workload, device="cuda"):
    """Projections at unit output scale, zero-centred norm weights (as the model
    stores them) and real rotary tables at the model's theta (1D positions: the
    text-only MRoPE tables agree with them to ~6e-8)."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 14))
    hidden, head_dim, rotary = dims["H"], dims["D"], dims["R"]

    def projection(out_features):
        return (torch.randn((out_features, hidden), device=device) * hidden ** -0.5).to(dtype)

    inv_freq = 1.0 / (CONFIG.rope_theta ** (
        torch.arange(0, rotary, 2, device=device, dtype=torch.float32) / rotary))
    angles = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None] * inv_freq[None]
    table = torch.cat((angles, angles), dim=-1)[None]
    b, t = dims["B"], dims["T"]
    return {
        "x": torch.randn((b, t, hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QG"]),
        "k_weight": projection(dims["KVO"]),
        "v_weight": projection(dims["KVO"]),
        "q_norm_weight": (0.02 * torch.randn((head_dim,), device=device)).to(dtype),
        "k_norm_weight": (0.02 * torch.randn((head_dim,), device=device)).to(dtype),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": 1e-6,
        "dq": torch.randn((b, dims["HQ"], t, head_dim), device=device, dtype=dtype),
        "dk": torch.randn((b, dims["HK"], t, head_dim), device=device, dtype=dtype),
        "dv": torch.randn((b, dims["HK"], t, head_dim), device=device, dtype=dtype),
        "dgate": torch.randn((b, t, dims["QO"]), device=device, dtype=dtype),
    }


op = declare_op(
    name="qwen3_5_qkv_norm_partial_rope",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.qkv_norm_partial_rope."
            "reference:qkv_norm_partial_rope_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.qkv_norm_partial_rope."
                    "reference:qkv_norm_partial_rope_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QG, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("q_norm_weight", "[D]"),
        Active("k_norm_weight", "[D]"),
        Inactive("cos", "[1, T, R]"),
        Inactive("sin", "[1, T, R]"),
        Inactive("eps", default=1e-6),
    ),
    output=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("gate", "[B, T, QO]"),
    ),
    parameter_args=("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight"),
    forward_semantics=(
        "qg = x @ q_weight^T viewed as [B, T, HQ, 2D]; per head the first D "
        "channels are the query and the next D the gate (interleaved per head, "
        "not two halves of QG). gate = the gate channels reshaped to [B, T, QO]. "
        "The query and k = (x @ k_weight^T) viewed [B, T, HK, D] are each "
        "normalized per head in float32 with a zero-centred weight: "
        "h * rsqrt(mean(h^2, -1) + eps) * (1 + w), cast to x's dtype, then "
        "transposed to [B, heads, T, D]. v = x @ v_weight^T viewed [B, T, HK, D] "
        "and transposed. RoPE on q and k over the first R channels only "
        "(rotate_half on that slice, cos/sin unsqueezed at dim 1); channels R..D "
        "pass through. Return (q, k, v, gate) IN THAT ORDER in x's dtype; q and k "
        "contiguous, v a transposed view, gate contiguous. No biases. Do not call "
        "F.linear, @, torch.matmul, F.rms_norm or autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dq, dk, dv, dgate) and returns dx, dq_weight, "
        "dk_weight, dv_weight, dq_norm_weight, dk_norm_weight IN THIS ORDER. "
        "Unrotate the first R channels of dq and dk (dt_rot*cos - "
        "rotate_half(dt_rot*sin)); then the (1 + w) RMSNorm backward per head "
        "(with r = rsqrt(mean(h^2, -1) + eps) and xhat = h * r: d(norm_weight) = "
        "sum over B, T and the heads of dnorm * xhat; dh = r * (g - xhat * "
        "mean(g * xhat, -1)) with g = dnorm * (1 + w)); re-interleave dquery and "
        "dgate into the [B, T, HQ, 2D] layout; finally dq_weight = dqg^T @ x, "
        "dk_weight, dv_weight likewise, and dx = dqg @ q_weight + dk_flat @ "
        "k_weight + dv_flat @ v_weight. Accumulate in float32; each gradient in "
        "its input's dtype. cos and sin get no gradient."
    ),
    extra_constraints=(
        "HQ divisible by HK; R even and at most D. The per-head [query | gate] "
        "interleave in q_weight is the model's and must be honoured exactly."
    ),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    make_inputs=make_qkv_norm_partial_rope_inputs,
)
