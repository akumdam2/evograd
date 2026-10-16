"""Operator declaration: OLMoE's q/k/v projections, full-width q/k RMSNorms and
RoPE (site ``qkv_norm_rope``).

Three outputs: ``q``, ``k`` and ``v`` ready for SDPA, in head-major layout.
``cos``/``sin`` are the model's rotary tables over the full 128-wide head
(theta 10,000) and receive no gradient. The norm weights are ``[QO]`` and
``[KVO]`` -- one scale per projection channel -- which is what makes this
boundary OLMoE's own rather than Qwen3's per-head one.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op

from ..task_tools import CONFIG, OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Not calibrated: the repository's standard float32/bfloat16 defaults until the
#: first GPU session's inventory (``task_tools.PROVISIONAL``).
PROVISIONAL_TOLERANCES = True

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_BENCHMARK = observed("qkv_norm_rope")


def _dims(b, t, h, hq, hk, d):
    return {"B": b, "T": t, "H": h, "HQ": hq, "HK": hk, "D": d, "QO": hq * d, "KVO": hk * d}


_CORRECTNESS = tuple(
    Workload(dims=_dims(*shape), dtype=dtype, provenance=shrunk("qkv_norm_rope"))
    for shape, dtype in (
        # Multi-head, as OLMoE-1B-7B's 16/16.
        ((1, 12, 32, 4, 4, 8), "float32"),
        # Grouped, so a kernel that assumed HQ == HK is caught.
        ((2, 20, 64, 4, 2, 16), "float32"),
        ((2, 24, 64, 8, 8, 8), "bfloat16"),
    )
)


def make_qkv_norm_rope_inputs(torch, op, workload, device="cuda"):
    """Projections at unit output scale, norm weights centred on one (as the
    model initializes them) and real rotary tables at the model's theta."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 21))
    hidden, head_dim = dims["H"], dims["D"]
    b, t = dims["B"], dims["T"]

    def projection(out_features):
        return (torch.randn((out_features, hidden), device=device) * hidden ** -0.5).to(dtype)

    def norm_weight(width):
        return (1.0 + 0.1 * torch.randn((width,), device=device)).to(dtype)

    inv_freq = 1.0 / (CONFIG.rope_theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.arange(t, device=device, dtype=torch.float32)[:, None] * inv_freq[None]
    table = torch.cat((angles, angles), dim=-1)[None]
    return {
        "x": torch.randn((b, t, hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"]),
        "k_weight": projection(dims["KVO"]),
        "v_weight": projection(dims["KVO"]),
        "q_norm_weight": norm_weight(dims["QO"]),
        "k_norm_weight": norm_weight(dims["KVO"]),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": 1e-5,
        "dq": torch.randn((b, dims["HQ"], t, head_dim), device=device, dtype=dtype),
        "dk": torch.randn((b, dims["HK"], t, head_dim), device=device, dtype=dtype),
        "dv": torch.randn((b, dims["HK"], t, head_dim), device=device, dtype=dtype),
    }


op = declare_op(
    name="olmoe_qkv_norm_rope",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.qkv_norm_rope.reference:"
            "qkv_norm_rope_forward_ref",
    runtime_forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.qkv_norm_rope."
                    "reference:qkv_norm_rope_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("q_norm_weight", "[QO]"),
        Active("k_norm_weight", "[KVO]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
        Inactive("eps", default=1e-5),
    ),
    output=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
    ),
    parameter_args=("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight"),
    forward_semantics=(
        "qp = x @ q_weight^T ([B, T, QO]) and kp = x @ k_weight^T ([B, T, KVO]) are "
        "each RMS-normalized over their LAST axis as a whole -- the full projection "
        "width, not per head -- in float32: h * rsqrt(mean(h^2, -1) + eps) * w, "
        "with w = q_norm_weight ([QO]) or k_norm_weight ([KVO]), cast to x's dtype. "
        "v = x @ v_weight^T. Each is then viewed [B, T, heads, D] and transposed to "
        "[B, heads, T, D] (HQ heads for q, HK for k and v). RoPE on q and k over "
        "the whole head: t * cos + rotate_half(t) * sin with cos/sin unsqueezed at "
        "dim 1, rotate_half(t) = cat(-t[..., D/2:], t[..., :D/2]). Return (q, k, v) "
        "IN THAT ORDER in x's dtype; q and k contiguous, v a transposed view. No "
        "biases, no clamp. Do not call F.linear, @, torch.matmul, F.rms_norm or "
        "autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dq, dk, dv) and returns dx, dq_weight, dk_weight, "
        "dv_weight, dq_norm_weight, dk_norm_weight IN THIS ORDER. Unrotate dq and dk "
        "(dt_rot * cos - rotate_half(dt_rot * sin)), transpose back to [B, T, heads, D] "
        "and flatten to [B, T, QO] / [B, T, KVO]; then the RMSNorm backward over the "
        "full width (dw = sum over B and T of dnorm * h * r; dh = r * (g - xhat * "
        "mean(g * xhat, -1)) with g = dnorm * w, r = rsqrt(mean(h^2) + eps)); finally "
        "dq_weight = dqp^T @ x, dk_weight = dkp^T @ x, dv_weight = dvp^T @ x (summed "
        "over B and T) and dx = dqp @ q_weight + dkp @ k_weight + dvp @ v_weight. "
        "Accumulate in float32; each gradient in its input's dtype. cos and sin get "
        "no gradient."
    ),
    extra_constraints=(
        "HQ divisible by HK; QO = HQ * D and KVO = HK * D; cos and sin are [1, T, D] "
        "over the whole head. The norm is over QO (or KVO) channels, not over D: a "
        "per-head normalization is a different operator and must not pass."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=PROVISIONAL,
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    make_inputs=make_qkv_norm_rope_inputs,
)
