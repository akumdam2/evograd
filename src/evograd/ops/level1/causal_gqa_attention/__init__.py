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

from evograd.opdecl import Active, declare_op
from evograd.ops._common import is_head_major_view

_DIMS = ("B", "HQ", "HK", "T", "D")


#: The observed Qwen3-0.6B configuration: 16 query heads over 8 KV heads, batch
#: 2 x sequence 2048, 28 invocations per step.


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
    memory_inputs=("q", "k", "v"),
    make_inputs=make_causal_gqa_attention_inputs,
)
