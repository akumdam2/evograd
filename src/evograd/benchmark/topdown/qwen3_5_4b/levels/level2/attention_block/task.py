"""Operator declaration: the gated full-attention module (site ``attention_block``).

``Qwen3_5Attention`` from its input to its output: the q/gate, k and v
projections, the ``(1 + w)`` per-head q/k norms, partial RoPE (the first R of D
channels), causal grouped-query SDPA, the sigmoid output gate and o_proj. 16
query heads over 4 KV heads at D = 256 and R = 64 in Qwen3.5-4B. ``cos``/``sin``
are the model's rotary tables (interleaved MRoPE on text positions) and receive
no gradient.
"""

from evograd.opdecl import Active, Inactive, declare_op

from ..task_tools import CONFIG, OBSERVED_SUITE, observed, seed_of


_DIMS = ("B", "T", "H", "HQ", "HK", "D", "R", "QG", "KVO", "QO")

_BENCHMARK = observed("attention_block")


def make_attention_block_inputs(torch, op, workload, device="cuda"):
    """Projections at unit output scale, zero-centred norm weights (as the model
    stores them) and real rotary tables at the model's theta (1D positions: the
    text-only MRoPE tables agree with them to ~6e-8)."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 15))
    hidden, head_dim, rotary = dims["H"], dims["D"], dims["R"]

    def projection(out_features, in_features):
        return (torch.randn((out_features, in_features), device=device)
                * in_features ** -0.5).to(dtype)

    inv_freq = 1.0 / (CONFIG.rope_theta ** (
        torch.arange(0, rotary, 2, device=device, dtype=torch.float32) / rotary))
    angles = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None] * inv_freq[None]
    table = torch.cat((angles, angles), dim=-1)[None]
    b, t = dims["B"], dims["T"]
    return {
        "x": torch.randn((b, t, hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QG"], hidden),
        "k_weight": projection(dims["KVO"], hidden),
        "v_weight": projection(dims["KVO"], hidden),
        "q_norm_weight": (0.02 * torch.randn((head_dim,), device=device)).to(dtype),
        "k_norm_weight": (0.02 * torch.randn((head_dim,), device=device)).to(dtype),
        "o_weight": projection(hidden, dims["QO"]),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": 1e-6,
        "dy": torch.randn((b, t, hidden), device=device, dtype=dtype),
    }


op = declare_op(
    name="qwen3_5_attention_block",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.attention_block.reference:"
            "attention_block_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.attention_block."
                    "reference:attention_block_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QG, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("q_norm_weight", "[D]"),
        Active("k_norm_weight", "[D]"),
        Active("o_weight", "[H, QO]"),
        Inactive("cos", "[1, T, R]"),
        Inactive("sin", "[1, T, R]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[B, T, H]"),
    parameter_args=("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight",
                    "o_weight"),
    forward_semantics=(
        "qg = x @ q_weight^T viewed as [B, T, HQ, 2D]; per head the first D "
        "channels are the query and the next D the gate (interleaved per head, "
        "not two halves of QG). gate = the gate channels reshaped to [B, T, QO]. "
        "The query and k = (x @ k_weight^T) viewed [B, T, HK, D] are each "
        "normalized per head in float32 with a zero-centred weight: "
        "h * rsqrt(mean(h^2, -1) + eps) * (1 + w), cast to x's dtype, then "
        "transposed to [B, heads, T, D]. v = x @ v_weight^T viewed [B, T, HK, D] "
        "and transposed. RoPE on q and k over the first R channels only "
        "(rotate_half on that slice, cos/sin broadcast over the batch and the "
        "heads); channels R..D pass through. Causal grouped-query attention with "
        "scale D^-0.5: each KV head serves HQ/HK consecutive query heads; scores "
        "= q @ k^T * D^-0.5 with strictly future positions masked, softmax in "
        "float32, a = weights @ v, transposed to [B, T, HQ, D] and flattened to "
        "[B, T, QO]. y = (a * sigmoid(gate)) @ o_weight^T, [B, T, H] in x's "
        "dtype. No biases, no mask tensor, no dropout. Do not call F.linear, @, "
        "torch.matmul, F.rms_norm, F.scaled_dot_product_attention or autograd "
        "in the generated math."
    ),
    backward_semantics=(
        "The backward receives dy and returns dx, dq_weight, dk_weight, "
        "dv_weight, dq_norm_weight, dk_norm_weight, do_weight IN THIS ORDER. "
        "With s = sigmoid(gate): do_weight = dy^T @ (a * s) summed over B and T; "
        "dag = dy @ o_weight; dgate = dag * a * s * (1 - s); da = dag * s, "
        "reshaped to [B, HQ, T, D]. The causal attention backward, with P the "
        "softmax weights: dP = da @ v^T, dS = P * (dP - rowsum(dP * P)), dq = "
        "dS @ k * D^-0.5, dk = dS^T @ q * D^-0.5, dv = P^T @ da, dk and dv "
        "summed over each group of HQ/HK query heads. Unrotate the first R "
        "channels of dq and dk (d_rot*cos - rotate_half(d_rot*sin)). Then the "
        "(1 + w) RMSNorm backward per head (with r = rsqrt(mean(h^2, -1) + eps) "
        "and xhat = h * r: d(norm_weight) = sum over B, T and the heads of "
        "dnorm * xhat; dh = r * (g - xhat * mean(g * xhat, -1)) with g = dnorm "
        "* (1 + w)); re-interleave dquery and dgate into the [B, T, HQ, 2D] "
        "layout; finally dq_weight = dqg^T @ x, dk_weight, dv_weight likewise, "
        "and dx = dqg @ q_weight + dk_flat @ k_weight + dv_flat @ v_weight. "
        "Accumulate in float32; each gradient in its input's dtype. cos and sin "
        "get no gradient."
    ),
    extra_constraints=(
        "HQ divisible by HK; R even and at most D; D at most 256. The per-head "
        "[query | gate] interleave in q_weight is the model's and must be "
        "honoured exactly."
    ),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    make_inputs=make_attention_block_inputs,
)
