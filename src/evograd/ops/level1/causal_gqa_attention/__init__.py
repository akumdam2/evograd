"""Operator declaration: causal grouped-query scaled dot-product attention.

The one primitive a decoder-only training step runs that this benchmark had no
generic task for. Every other Qwen3-0.6B configuration mapped onto an operator
that already existed; this one did not, so it is added generically rather than
as a Qwen copy -- the shape family (batch, query heads, KV heads, tokens, head
dimension) describes every GQA decoder, and both Llama-3-8B and Qwen3-0.6B
grids are derived from their published configurations.

Attention alone. The output projection that follows it in a real layer is a
separate GEMM and belongs to the Level-2 ``qwen3_attention`` task; the two meet
exactly where this one's output becomes that one's input.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
)
from evograd.ops._common import (
    fixed_shape_suites,
    is_head_major_view,
    log_distance_weight,
    model_workloads,
    observed_workloads,
    regime_suites,
)

_DIMS = ("B", "HQ", "HK", "T", "D")


def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["T"])


#: Llama-3-8B's own GQA layer, swept over the standard token grid. Derived from
#: the published configuration, so the 32:8 head ratio and 128-wide heads cannot
#: quietly stop being Llama's.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "causal_gqa_sdpa",
    tuple({"batch": 1, "seq": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)

#: The observed Qwen3-0.6B configuration: 16 query heads over 8 KV heads, batch
#: 2 x sequence 2048, 28 invocations per step.
_QWEN3_OBSERVED = observed_workloads("qwen3_0_6b", "causal_gqa_attention")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, HQ=hq, HK=hk, T=t, D=d), dtype=dtype)
    for b, hq, hk, t, d in (
        (1, 4, 2, 16, 16),
        (2, 4, 2, 32, 32),
        # 4:1 grouping, so a kernel that assumed 2:1 fails here.
        (1, 8, 2, 24, 32),
    )
    for dtype in ("float32", "bfloat16")
)


def make_causal_gqa_attention_inputs(torch, op, workload, device="cuda"):
    """Head-major q/k/v, non-contiguous where the model's are.

    A decoder produces these by projecting into ``[B, T, heads, D]`` and
    transposing, so what SDPA receives is a non-contiguous head-major view. The
    observed Qwen3 configuration is generated that way; the derived Llama grid
    keeps the contiguous layout, so the two say which one they are measuring
    instead of both silently measuring the same substitute.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(dims["T"] * 100003 + dims["HQ"] * 1009 + dims["D"])
    observed = is_head_major_view(workload)

    def draw(heads):
        if observed:
            token_major = torch.randn(
                (dims["B"], dims["T"], heads, dims["D"]), device=device, dtype=dtype
            )
            return token_major.transpose(1, 2)
        return torch.randn(
            (dims["B"], heads, dims["T"], dims["D"]), device=device, dtype=dtype
        )

    return {
        "q": draw(dims["HQ"]),
        "k": draw(dims["HK"]),
        "v": draw(dims["HK"]),
        "do": draw(dims["HQ"]),
    }


op = declare_op(
    name="causal_gqa_attention",
    level=1,
    family="attention",
    forward=(
        "evograd.ops.level1.causal_gqa_attention.forward_ref:"
        "causal_gqa_attention_forward_ref"
    ),
    # The eager baseline is timed through the fused SDPA a training step runs.
    # The declared forward materializes a [B, HQ, T, T] score matrix the real
    # execution never allocates, so timing against it would compare every
    # candidate to a strawman.
    runtime_forward=(
        "evograd.ops.level1.causal_gqa_attention.forward_ref:"
        "causal_gqa_attention_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
    ),
    output=Active("o", "[B, HQ, T, D]"),
    parameter_args=(),
    forward_semantics=(
        "Causal grouped-query attention. Expand k and v from HK to HQ heads by "
        "repeating each KV head HQ/HK times; scores = q @ k_expanded^T * "
        "(1/sqrt(D)); mask strictly-future positions to -inf; softmax over the "
        "last axis in float32 and cast back to q's dtype; "
        "o = weights @ v_expanded, shape [B, HQ, T, D]. There is no attention "
        "mask tensor, no dropout and no KV cache. Do not call "
        "F.scaled_dot_product_attention, torch.matmul, @, F.softmax, or "
        "autograd in the generated math."
    ),
    backward_semantics=(
        "Return dq, dk, dv IN THIS ORDER. With weights as above: "
        "dv_expanded = weights^T @ do; dweights = do @ v_expanded^T; "
        "dscores = weights * (dweights - sum(dweights * weights, dim=-1, "
        "keepdim=True)), with strictly-future positions zeroed; "
        "dq = (dscores @ k_expanded) / sqrt(D) and dk_expanded = "
        "(dscores^T @ q) / sqrt(D). Sum dk_expanded and dv_expanded over each "
        "group of HQ/HK query heads to get dk and dv at HK heads. Accumulate "
        "every reduction and matmul in float32 before casting each gradient to "
        "its input's dtype."
    ),
    extra_constraints=(
        "HQ must be divisible by HK. q, k and v may be non-contiguous "
        "head-major views -- a decoder reaches them by transposing out of "
        "[B, T, heads, D] -- and the observed Qwen3 workloads are generated "
        "that way; a kernel may work in any internal layout. Attention only: "
        "the output projection is a separate GEMM."
    ),
    grad_order=("dq", "dk", "dv"),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK + _QWEN3_OBSERVED,
    benchmark=_BENCHMARK,
    benchmark_suites={
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
    },
    # Measured, not chosen. `bench.workloads.qwen3.levels.level1.mapping calibrate` compares the
    # declared dense forward against `runtime_forward` -- the fused SDPA the
    # model runs, and therefore the smallest disagreement any correct
    # implementation can have with the oracle -- on every correctness workload
    # and on the observed Qwen configuration.
    #
    # float32: worst 1.0e-06 across every case, which the repository's ordinary
    # float32 pair clears by ~20x. `dv` measures exactly 0.0 at both dtypes on
    # the small cases: SDPA and the dense spelling compute it identically there.
    #
    # bfloat16 worst: o 5.2e-03, dq 1.4e-02, dk 1.4e-02, dv 3.9e-03. The base is
    # set by the binding *forward* requirement so the output is gated by the base
    # alone and only the gradients carry multipliers; 1e-02 leaves 1.9x on `o`.
    tolerances={
        "float32": (2e-5, 2e-5),
        "bfloat16": (1e-2, 1e-2),
    },
    # The measured minimum atol multiplier at base 1e-2, times a 1.5 safety
    # margin, rounded up to one decimal. `o` and `dv` measured 1.00 and have
    # none.
    #
    #   result   measured min ma at t=1e-2   declared
    #   dq                 1.46                2.2
    #   dk                 1.48                2.3
    tolerance_multipliers={"dq": (2.2, 1.0), "dk": (2.3, 1.0)},
    memory_inputs=("q", "k", "v"),
    regime_feature=_regime_feature,
    regime_split=LLAMA_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_REGIME_SPLIT),
    make_inputs=make_causal_gqa_attention_inputs,
)
