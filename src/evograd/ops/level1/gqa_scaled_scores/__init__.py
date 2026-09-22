"""Operator declaration: scaled grouped-query attention scores.

The first of three primitives that the causal grouped-query attention
boundary decomposes into (scores -> causal softmax -> PV). Attention is what a
decoder runs; these three are how its mathematics splits into reusable
contracts whose cost can be measured and evolved one at a time. The
decomposition is a statement about the mathematics, not a claim that any
captured training step called this primitive on its own -- the observed
boundary is ``causal_gqa_attention`` -- so a model's cases join this contract
as *derived* cases (see ``benchmark.operator_suite.cases.gqa_scaled_scores``),
never as observed ones.

Intermediate dtypes are part of the contract. q and k are stored in the
model's dtype; the scores are float32 whatever that dtype is, because the
softmax that consumes them is defined on float32 scores and a bf16 score
tensor would lose the precision the real fused kernels keep in registers.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import head_major_randn

_DIMS = ("B", "HQ", "HK", "T", "D")

#: Small grids that prove an implementation right on CPU-sized shapes,
#: including a 4:1 grouping and a T that is not a tile multiple; then the
#: full-size case of the decomposition (T=2048, D=128, 16/8 heads) so the
#: gate includes a target-size numerical check and not only small ones.
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


def make_gqa_scaled_scores_inputs(torch, op, workload, device="cuda"):
    """Head-major q/k views, and a float32 upstream gradient for the scores."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(dims["T"] * 100003 + dims["HQ"] * 1009 + dims["D"] + 7)
    return {
        "q": head_major_randn(torch, dims["B"], dims["T"], dims["HQ"], dims["D"], device=device, dtype=dtype),
        "k": head_major_randn(torch, dims["B"], dims["T"], dims["HK"], dims["D"], device=device, dtype=dtype),
        "ds": torch.randn((dims["B"], dims["HQ"], dims["T"], dims["T"]), device=device, dtype=torch.float32),
    }


op = declare_op(
    name="gqa_scaled_scores",
    level=1,
    family="attention",
    forward=(
        "evograd.ops.level1.gqa_scaled_scores.forward_ref:"
        "gqa_scaled_scores_forward_ref"
    ),
    # The eager baseline is timed through the tensor-core spelling with a
    # float32 result; the oracle's float32-operand GEMMs are what defines the
    # contract, not what a training step would pay for.
    runtime_forward=(
        "evograd.ops.level1.gqa_scaled_scores.forward_ref:"
        "gqa_scaled_scores_runtime"
    ),
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
    ),
    output=Active("s", "[B, HQ, T, T]", dtype="float32"),
    parameter_args=(),
    forward_semantics=(
        "Scaled grouped-query attention scores. Expand k from HK to HQ heads "
        "by repeating each KV head HQ/HK times (query head hq uses KV head "
        "hq // (HQ/HK)); s = q @ k_expanded^T * (1/sqrt(D)), shape "
        "[B, HQ, T, T], ALWAYS float32 regardless of q's dtype, with the "
        "products accumulated in float32. Every position is computed (no "
        "causal mask here: masking belongs to the softmax that consumes s). "
        "Do not call torch.matmul, torch.bmm, @, F.scaled_dot_product_attention "
        "or autograd in the generated math."
    ),
    backward_semantics=(
        "Return dq, dk IN THIS ORDER. ds is float32 [B, HQ, T, T]. "
        "dq = (ds @ k_expanded) / sqrt(D), shape [B, HQ, T, D], q's dtype; "
        "dk_expanded = (ds^T @ q) / sqrt(D); dk = dk_expanded summed over "
        "each group of HQ/HK query heads that share a KV head, shape "
        "[B, HK, T, D], k's dtype. Accumulate every contraction and the "
        "group reduction in float32 before casting each gradient to its "
        "input's dtype."
    ),
    extra_constraints=(
        "HQ must be divisible by HK. q and k are non-contiguous head-major "
        "views (strides [T*heads*D, D, heads*D, 1]) because a decoder "
        "transposes them out of [B, T, heads, D]; a kernel may use any "
        "internal layout but must read the declared strides. The output s is "
        "float32 and contiguous. Scores only: the causal mask, the softmax "
        "and the value product are separate primitives."
    ),
    grad_order=("dq", "dk"),
    correctness=_CORRECTNESS,
    # Measured, not chosen (2026-09-16, GH200, before any search; report:
    # results/experiments/gqa_l1_context/20260915/evolve/calibration/calibration.json).
    # The declared oracle was compared against runtime_forward (bf16 tensor-core
    # GEMMs, ds cast to bf16) and torch.compile of the oracle on every
    # correctness case including the full-size one.
    #
    # float32: worst 8.4e-07 (accumulation order only); the ordinary pair.
    # bfloat16: s is float32 and needs only 8.9e-07, so the base is set at the
    # 2e-3 dtype floor -- tight enough that a score tensor rounded through
    # bf16 (relative 3.9e-3) is rejected, which is the point of the float32
    # output. dq and dk are what a bf16-operand backward needs at this
    # cotangent scale: minimal atol multipliers 46.4 and 50.9 at base 2e-3,
    # declared with a 1.5x margin.
    tolerances={
        "float32": (2e-5, 2e-5),
        "bfloat16": (2e-3, 2e-3),
    },
    tolerance_multipliers={"dq": (70.0, 1.0), "dk": (77.0, 1.0)},
    memory_inputs=("q", "k"),
    make_inputs=make_gqa_scaled_scores_inputs,
)
