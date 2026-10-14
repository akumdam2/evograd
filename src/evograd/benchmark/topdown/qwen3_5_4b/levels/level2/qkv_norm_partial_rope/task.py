"""Operator declaration: gated attention's q/gate/k/v projections, (1 + w) per-head
norms and partial RoPE (site ``qkv_norm_partial_rope``).

Four outputs: ``q``, ``k`` and ``v`` ready for SDPA, and the output ``gate`` that
``qwen3_5_gated_attention`` applies after it. ``cos``/``sin`` are the model's
rotary tables (64 wide for a 256-wide head, interleaved MRoPE on text positions)
and receive no gradient.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op

from ..task_tools import (CONFIG, OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk,
                          token_major)

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "R", "QG", "KVO", "QO")


def model_strides(dims):
    """``dv`` as the block delivers it: a ``[B, HK, T, D]`` view of a token-major
    buffer -- what the default SDPA backend's backward returns for v on the GH200
    (measured on the GH200). dq and dk arrive contiguous."""
    return (("dv", token_major(dims["B"], dims["T"], dims["HK"], dims["D"])),)


_BENCHMARK = observed("qkv_norm_partial_rope", strides=model_strides)


def _dims(b, t, h, hq, hk, d, r):
    return {"B": b, "T": t, "H": h, "HQ": hq, "HK": hk, "D": d, "R": r,
            "QG": 2 * hq * d, "KVO": hk * d, "QO": hq * d}


_CORRECTNESS = tuple(
    Workload(dims=_dims(*shape), dtype=dtype, provenance=shrunk("qkv_norm_partial_rope"))
    for shape, dtype in (
        ((1, 12, 32, 4, 2, 16, 4), "float32"),
        ((2, 20, 64, 4, 2, 32, 8), "float32"),
        # 4:1 grouping, as Qwen3.5-4B's 16/4, and a rotary width of half the head.
        ((1, 16, 64, 8, 2, 16, 8), "float32"),
        ((2, 24, 64, 4, 1, 32, 8), "bfloat16"),
    )
) + (
    Workload(dims=_dims(2, 20, 64, 4, 2, 32, 8), dtype="float32",
             provenance=shrunk("qkv_norm_partial_rope",
                               model_strides(_dims(2, 20, 64, 4, 2, 32, 8)))),
)


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


# --- calibrated tolerances (scripts/qwen3_5_4b/calibration.py apply) ---
from evograd.opdecl.tolerance import ReductionScaledAtol  # noqa: E402

#: Calibrated for Qwen3.5-4B (run ``20261011_calib``; GH200, grid also on CPU):
#: ``op.forward`` against ``op.runtime_forward`` only -- never a library or
#: a candidate -- on captured, correctness, correctness[cpu], observed;
#: four repeats each (production was deterministic). The bfloat16 base
#: atol is the smallest 1-2-5 step at or above both 1.5x the forward
#: outputs' grid need and one bfloat16 ulp of their magnitude; rtol 2e-2.
#: Multipliers are per dtype (in the hook): 1.5x that dtype's grid need,
#: rounded up (none below 1). Each gain is the smallest that carries the
#: bfloat16 multiplier over the real-size cases with the same 1.5x.
#: ``need`` is the atol required at the base rtol, in units of the base atol.
#:
#:   result           fp32 grid bf16 grid real need fp32x bf16x  gain margin@real
#:   dk_norm_weight          0    0.778      18.5     1   1.2   1.5     1.58
#:   dk_weight               0     2.32      25.8     1   3.5   1.5     1.78
#:   dq_norm_weight          0     1.49      37.1     1   2.3   1.5     1.51
#:   dq_weight               0     2.34      28.4     1   3.6   1.5     1.57
#:   dv_weight               0        0    0.0075     1     1     0      133
#:   dx                      0    0.593      1.25     1     1     3     1.74
#:   gate                    0        0  0.000199     1     1     0 5.03e+03
#:   k                       0    0.143     0.666     1     1     0      1.5
#:   q                       0    0.265     0.904     1     1   1.5     1.69
#:   v                       0        0  0.000187     1     1     0 5.35e+03
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.02, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 2, "D": 32, "H": 64, "HK": 1, "HQ": 4, "KVO": 32, "QG": 256, "QO": 128, "R": 8, "T": 24},
    reduction_dims={"dk_norm_weight": ('B', 'HK', 'T'), "dk_weight": ('B', 'T'), "dq_norm_weight": ('B', 'HQ', 'T'), "dq_weight": ('B', 'T'), "dv_weight": ('B', 'T')},
    result_dims={"dk_norm_weight": ('D',), "dk_weight": ('KVO', 'H'), "dq_norm_weight": ('D',), "dq_weight": ('QG', 'H'), "dv_weight": ('KVO', 'H'), "dx": ('B', 'T', 'H'), "gate": ('B', 'T', 'QO'), "k": ('B', 'HK', 'T', 'D'), "q": ('B', 'HQ', 'T', 'D'), "v": ('B', 'HK', 'T', 'D')},
    gain=0.0,
    result_gain={"dk_norm_weight": 1.5, "dk_weight": 1.5, "dq_norm_weight": 1.5, "dq_weight": 1.5, "dx": 3, "q": 1.5},
    dtype_multipliers={"bfloat16": {"dk_norm_weight": 1.2, "dk_weight": 3.5, "dq_norm_weight": 2.3, "dq_weight": 3.6}},
)
# --- end calibrated tolerances ---


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
        "(d(norm_weight) = sum over rows of dnorm * h * r; dh = r * (g - xhat * "
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
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    make_inputs=make_qkv_norm_partial_rope_inputs,
)
