"""Operator declaration: OLMoE's whole self-attention module (site ``attention_block``).

``OlmoeAttention.forward`` from its input to its output: the q/k/v projections,
the full-width q/k RMSNorms, RoPE, causal multi-head SDPA and ``o_proj``. One
output, ``out [B, T, H]``, whose gradient is the one the residual add hands
back. ``cos``/``sin`` are the model's rotary tables over the full 128-wide
head (theta 10,000) and receive no gradient. The norm weights are ``[QO]`` and
``[KVO]`` -- one scale per projection channel -- which is what makes this
boundary OLMoE's own rather than Qwen3's per-head one. OLMoE-1B-7B has 16
query heads over 16 KV heads; the contract is multi-head (``HQ == HK``).
"""

from evograd.opdecl import Active, Inactive, declare_op

from ..task_tools import CONFIG, OBSERVED_SUITE, observed, seed_of


_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_BENCHMARK = observed("attention_block")


def make_attention_block_inputs(torch, op, workload, device="cuda"):
    """Projections at unit output scale, norm weights centred on one (as the
    model initializes them) and real rotary tables at the model's theta."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 21))
    hidden, head_dim, t = dims["H"], dims["D"], dims["T"]

    def projection(out_features, in_features):
        return (torch.randn((out_features, in_features), device=device)
                * in_features ** -0.5).to(dtype)

    def norm_weight(width):
        return (1.0 + 0.1 * torch.randn((width,), device=device)).to(dtype)

    inv_freq = 1.0 / (CONFIG.rope_theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.arange(t, device=device, dtype=torch.float32)[:, None] * inv_freq[None]
    table = torch.cat((angles, angles), dim=-1)[None]
    return {
        "x": torch.randn((dims["B"], t, hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"], hidden),
        "k_weight": projection(dims["KVO"], hidden),
        "v_weight": projection(dims["KVO"], hidden),
        "q_norm_weight": norm_weight(dims["QO"]),
        "k_norm_weight": norm_weight(dims["KVO"]),
        "o_weight": projection(hidden, dims["QO"]),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": 1e-5,
        "dout": torch.randn((dims["B"], t, hidden), device=device, dtype=dtype),
    }


op = declare_op(
    name="olmoe_attention_block",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.attention_block.reference:"
            "attention_block_forward_ref",
    runtime_forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.attention_block."
                    "reference:attention_block_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("q_norm_weight", "[QO]"),
        Active("k_norm_weight", "[KVO]"),
        Active("o_weight", "[H, QO]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight",
                    "o_weight"),
    forward_semantics=(
        "qp = x @ q_weight^T ([B, T, QO]) and kp = x @ k_weight^T ([B, T, KVO]) are "
        "each RMS-normalized over their LAST axis as a whole -- the full projection "
        "width, not per head -- in float32: h * rsqrt(mean(h^2, -1) + eps) * w, "
        "with w = q_norm_weight ([QO]) or k_norm_weight ([KVO]). vp = x @ v_weight^T. "
        "Each is viewed [B, T, heads, D] and transposed to [B, heads, T, D] (HQ heads "
        "for q, HK = HQ for k and v). RoPE on q and k over the whole head: t * cos + "
        "rotate_half(t) * sin with cos/sin unsqueezed at dim 1, rotate_half(t) = "
        "cat(-t[..., D/2:], t[..., :D/2]). Causal attention with scale D^-0.5: scores "
        "= q @ k^T * D^-0.5 with strictly future positions masked, softmax in float32, "
        "a = weights @ v, [B, HQ, T, D]. Transpose a to [B, T, HQ, D], flatten to "
        "[B, T, QO] and project: out = a @ o_weight^T, [B, T, H] in x's dtype. No "
        "biases, no clamp, no mask tensor, no dropout. Do not call F.linear, @, "
        "torch.matmul, F.rms_norm, F.scaled_dot_product_attention or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "The backward receives dout and returns dx, dq_weight, dk_weight, dv_weight, "
        "dq_norm_weight, dk_norm_weight, do_weight IN THIS ORDER. do_weight = dout^T @ "
        "a_flat summed over B and T; da = dout @ o_weight reshaped back to [B, HQ, T, "
        "D]; the causal attention backward (with P the softmax weights: dv = P^T @ da, "
        "dP = da @ v^T, dS = P * (dP - sum(dP * P, -1)), dq = dS @ k * D^-0.5, dk = "
        "dS^T @ q * D^-0.5) gives the rotated dq and dk. Unrotate them (dt_rot * cos - "
        "rotate_half(dt_rot * sin)), transpose dq, dk and dv back to [B, T, heads, D] "
        "and flatten to [B, T, QO] / [B, T, KVO]; then the RMSNorm backward over the "
        "full width (dw = sum over B and T of dnorm * h * r; dh = r * (g - xhat * "
        "mean(g * xhat, -1)) with g = dnorm * w, r = rsqrt(mean(h^2) + eps), xhat = h * "
        "r); finally dq_weight = dqp^T @ x, dk_weight = dkp^T @ x, dv_weight = dvp^T @ "
        "x (summed over B and T) and dx = dqp @ q_weight + dkp @ k_weight + dvp @ "
        "v_weight. Accumulate in float32; each gradient in its input's dtype. cos and "
        "sin get no gradient."
    ),
    extra_constraints=(
        "HQ == HK (multi-head: OLMoE-1B-7B's 16 over 16); QO = HQ * D = KVO; cos and sin "
        "are [1, T, D] over the whole head. The norm is over QO (or KVO) channels, not "
        "over D: a per-head normalization is a different operator and must not pass."
    ),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    make_inputs=make_attention_block_inputs,
)
