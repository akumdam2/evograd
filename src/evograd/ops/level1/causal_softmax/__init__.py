"""Operator declaration: causal masked softmax over float32 scores.

The second of the three primitives the causal grouped-query attention
boundary decomposes into (scores -> causal softmax -> PV). The mathematics is
the row-wise ``softmax`` primitive with the causal mask folded in, but the
contract differs in two ways that matter to a kernel: the input is always
float32 scores, and the output is cast to bfloat16 because that is what the
value product of a bf16 model consumes. Like its two
siblings it is a derived contract, not an operator any captured step called on
its own; a model's case joins it as a derived case.

Dtype policy, stated once: ``s`` float32 in; ``p`` bfloat16 out; ``dp``
arrives in bfloat16; ``ds`` is float32. That is exactly the autograd of the
declared bf16 attention oracle, where the softmax weights are cast to the
model dtype before the value product. Every case is therefore a "bfloat16"
workload: the dtype names the model, and the intermediates are fixed by the
contract rather than by it.
"""

from evograd.opdecl import Active, Workload, declare_op

_DIMS = ("B", "HQ", "T")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, HQ=hq, T=t), dtype=dtype)
    for b, hq, t in (
        (1, 4, 16),
        (2, 4, 32),
        # a T that is not a multiple of any tile width
        (1, 8, 24),
    )
    for dtype in ("bfloat16",)
) + (
    # The full-size case of the decomposition: a target-size numerical check.
    Workload(dims=dict(B=2, HQ=16, T=2048), dtype="bfloat16"),
)


def make_causal_softmax_inputs(torch, op, workload, device="cuda"):
    """Unit-variance float32 scores (what scaled q.k of unit inputs produce)
    and a bfloat16 upstream gradient for the probabilities."""
    dims = workload.dims
    dtype = torch.bfloat16
    torch.manual_seed(dims["T"] * 100003 + dims["HQ"] * 1009 + 11)
    shape = (dims["B"], dims["HQ"], dims["T"], dims["T"])
    return {
        "s": torch.randn(shape, device=device, dtype=torch.float32),
        "dp": torch.randn(shape, device=device, dtype=dtype),
    }


op = declare_op(
    name="causal_softmax",
    level=1,
    family="reduction",
    forward="evograd.ops.level1.causal_softmax.forward_ref:causal_softmax_forward_ref",
    dims=_DIMS,
    args=(Active("s", "[B, HQ, T, T]", dtype="float32"),),
    output=Active("p", "[B, HQ, T, T]", dtype="bfloat16"),
    parameter_args=(),
    forward_semantics=(
        "Causal masked softmax over the last axis of float32 scores "
        "s [B, HQ, T, T]. Row m keeps keys n <= m; strictly-future keys "
        "(n > m) are masked to -inf and receive probability exactly 0. The "
        "softmax is computed in float32 (max-subtracted, exp, row sum) and the "
        "result p is cast to bfloat16 (always, whatever the model dtype). There is "
        "no mask tensor: the causal structure is implied by the row index. Do "
        "not call torch.softmax, F.softmax, .softmax(), "
        "F.scaled_dot_product_attention or autograd in the generated math."
    ),
    backward_semantics=(
        "Return ds, float32 [B, HQ, T, T]. dp arrives in bfloat16. With p_f "
        "the float32 (uncast) probabilities: ds = p_f * (dp - sum(dp * p_f, "
        "dim=-1, keepdim=True)); masked positions have p_f = 0 and therefore "
        "ds = 0. Compute the row reduction and the product in float32. You may "
        "save p in any dtype, or per-row statistics (max, log-sum-exp) and "
        "recompute p_f from s, as long as the result meets the tolerance."
    ),
    extra_constraints=(
        "s and p are contiguous. T is the same for rows and keys (square "
        "causal). B*HQ*T rows of length T; at the full-size case that is "
        "65536 rows of 2048, and materializing any extra [B, HQ, T, T] "
        "tensor costs 256-512 MiB of traffic per pass."
    ),
    grad_order=("ds",),
    correctness=_CORRECTNESS,
    # PROVISIONAL until calibrated against a torch.compile control and the
    # oracle's own repeat noise; see gqa_scaled_scores for where the report is.
    tolerances={
        "bfloat16": (1e-2, 1e-2),
    },
    memory_inputs=("s",),
    make_inputs=make_causal_softmax_inputs,
)
