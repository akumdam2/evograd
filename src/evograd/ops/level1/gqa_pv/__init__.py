"""Operator declaration: grouped-query probability-value product.

The third primitive of the causal grouped-query attention decomposition
(scores -> causal softmax -> PV). The contract takes the masked probabilities
in the model's dtype and the values as the head-major view a decoder hands
attention, and returns the per-head attention output; the head merge and the
output projection that follow in a real layer are outside it. Derived, not
observed: no captured step calls this on its own.

Dtype policy: p and v in the workload dtype; o in that dtype with float32
accumulation; dp in p's dtype (the product with v^T accumulated in float32
and cast once); dv in v's dtype, the group reduction over the query heads
that share a KV head accumulated in float32.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import head_major_randn

_DIMS = ("B", "HQ", "HK", "T", "D")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, HQ=hq, HK=hk, T=t, D=d), dtype=dtype)
    for b, hq, hk, t, d in (
        (1, 4, 2, 16, 16),
        (2, 4, 2, 32, 32),
        (1, 8, 2, 24, 32),
    )
    for dtype in ("float32", "bfloat16")
) + (
    Workload(dims=dict(B=2, HQ=16, HK=8, T=2048, D=128), dtype="bfloat16"),
)


def make_gqa_pv_inputs(torch, op, workload, device="cuda"):
    """Causal probabilities (rows sum to one, future entries exactly zero),
    a head-major v view, and an upstream gradient for o."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(dims["T"] * 100003 + dims["HQ"] * 1009 + dims["D"] + 13)
    tokens = dims["T"]
    scores = torch.randn((dims["B"], dims["HQ"], tokens, tokens), device=device, dtype=torch.float32)
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=device).tril()
    p = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1).to(dtype)
    return {
        "p": p,
        "v": head_major_randn(torch, dims["B"], tokens, dims["HK"], dims["D"], device=device, dtype=dtype),
        "do": torch.randn((dims["B"], dims["HQ"], tokens, dims["D"]), device=device, dtype=dtype),
    }


op = declare_op(
    name="gqa_pv",
    level=1,
    family="attention",
    forward="evograd.ops.level1.gqa_pv.forward_ref:gqa_pv_forward_ref",
    runtime_forward="evograd.ops.level1.gqa_pv.forward_ref:gqa_pv_runtime",
    dims=_DIMS,
    args=(
        Active("p", "[B, HQ, T, T]"),
        Active("v", "[B, HK, T, D]"),
    ),
    output=Active("o", "[B, HQ, T, D]"),
    parameter_args=(),
    forward_semantics=(
        "Grouped-query probability-value product. Expand v from HK to HQ "
        "heads by repeating each KV head HQ/HK times (query head hq uses KV "
        "head hq // (HQ/HK)); o = p @ v_expanded, shape [B, HQ, T, D], in the "
        "workload dtype with the contraction accumulated in float32. p is "
        "causal: entries with key index > query index are exactly zero, so a "
        "kernel may skip strictly-upper tiles. Do not call torch.matmul, "
        "torch.bmm, @, F.scaled_dot_product_attention or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return dp, dv IN THIS ORDER. do has o's shape and dtype. "
        "dp = do @ v_expanded^T, shape [B, HQ, T, T], p's dtype (float32 "
        "accumulation, cast once; masked positions may hold any finite value "
        "the reference produces there -- the reference computes the full "
        "product); dv_expanded = p^T @ do; dv = dv_expanded summed over each "
        "group of HQ/HK query heads sharing a KV head, shape [B, HK, T, D], "
        "v's dtype, reduction accumulated in float32."
    ),
    extra_constraints=(
        "HQ must be divisible by HK. p is contiguous; v is a non-contiguous "
        "head-major view (strides [T*HK*D, D, HK*D, 1]) and dv must have v's "
        "shape (any strides). o is contiguous [B, HQ, T, D]; the head merge to "
        "[B, T, HQ*D] and the output projection are outside this contract."
    ),
    grad_order=("dp", "dv"),
    correctness=_CORRECTNESS,
    # PROVISIONAL until calibrated (oracle vs runtime_forward vs a
    # torch.compile control); see gqa_scaled_scores.
    tolerances={
        "float32": (2e-5, 2e-5),
        "bfloat16": (1e-2, 1e-2),
    },
    memory_inputs=("p", "v"),
    make_inputs=make_gqa_pv_inputs,
)
