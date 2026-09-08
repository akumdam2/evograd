"""Level-2 task: Llama-3-8B causal grouped-query attention plus output projection.

The same computation Qwen3-0.6B runs at this boundary, at Llama-3-8B's widths.
Same mathematics, different case -- and a case is not a contract, which is why
this is its own task key rather than a second suite on Qwen3's. An architecture
does not borrow another's task identity: a report row, a candidate program and
a calibration file all key off this name, and two models sharing one key would
make every one of them ambiguous.

The timed case is derived from the frozen Llama-3-8B configuration rather than
from a harvest snapshot: the shape is a property of the architecture, and
``tests/test_provenance`` re-derives it. Batch 2 x sequence 2048 is the
canonical Level-4 step for both models.
"""

from evograd.benchmark.topdown.common.level2_references import (
    attention_projection_forward_production as _production,
    attention_projection_forward_ref as _definition,
)
from evograd.opdecl import Active, Provenance, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.opdecl.tolerance import ReductionScaledAtol
from evograd.ops._common import model_workloads as _model_workloads

#: The workload these cases belong to; also the ``Provenance`` model key.
WORKLOAD = "llama_3_8b"

#: 32 -- once per decoder layer, for both halves of the boundary.
FREQUENCY = LLAMA_3_8B.layers

_DIMS = ("B", "T", "HQ", "HK", "D", "QO", "H")

#: The observed configuration, computed from the published model config.
_BENCHMARK = _model_workloads(
    LLAMA_3_8B,
    "causal_gqa_attention",
    ({"batch": 2, "seq": 2048},),
    ("bfloat16",),
)

_SHRUNK = Provenance(
    model="llama_3_8b",
    component="causal_gqa_attention",
    free={},
    source="handpicked",
    scaled=True,
    note=(
        "head count, head dimension, sequence length and hidden size reduced "
        "from Llama-3-8B's 32/8 heads, 128-wide heads and 2048 tokens so the "
        "correctness cases run on CPU; the 2:1 grouped-query ratio, the causal "
        "mask, the absence of a bias and the head-major input layout are all "
        "preserved"
    ),
)

_CORRECTNESS = tuple(
    Workload(
        dims={
            "B": batch,
            "T": tokens,
            "HQ": heads,
            "HK": kv_heads,
            "D": head_dim,
            "QO": heads * head_dim,
            "H": hidden,
        },
        dtype=dtype,
        provenance=_SHRUNK,
    )
    for batch, tokens, heads, kv_heads, head_dim, hidden, dtype in (
        (1, 16, 4, 2, 8, 24, "float32"),
        (2, 32, 4, 2, 16, 48, "float32"),
        (2, 32, 4, 2, 16, 48, "bfloat16"),
        # A 4:1 group ratio, so a kernel that assumed 2:1 fails here.
        (1, 24, 8, 2, 16, 32, "bfloat16"),
    )
)

_ANCHOR = {"B": 2, "T": 32, "HQ": 4, "HK": 2, "D": 16, "QO": 64, "H": 48}

_REDUCTION_SCALED = ReductionScaledAtol(
    anchor_dims=_ANCHOR,
    reduction_dims={},
    result_dims={
        "out": ("B", "T", "H"),
        "dq": ("B", "HQ", "T", "D"),
        "dk": ("B", "HK", "T", "D"),
        "dv": ("B", "HK", "T", "D"),
        "do_weight": ("H", "QO"),
    },
    gain=2.5,
)

def make_llama3_attention_inputs(torch, op, workload, device="cuda"):
    """Head-major q/k/v with the non-contiguous strides the model presents.

    Built token-major and transposed, which is how the model produces them --
    ``self.q_proj(x).view(B, T, -1, D).transpose(1, 2)``. Allocating contiguous
    tensors instead would silently benchmark a different memory access pattern
    than the one the observed call had.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["HQ"] * 1009 + dims["D"] * 101
    )

    def head_major(heads):
        token_major = torch.randn(
            (dims["B"], dims["T"], heads, dims["D"]), device=device, dtype=dtype
        )
        return token_major.transpose(1, 2)

    q = head_major(dims["HQ"])
    k = head_major(dims["HK"])
    v = head_major(dims["HK"])
    o_weight = (
        torch.randn((dims["H"], dims["QO"]), device=device, dtype=torch.float32)
        * dims["QO"] ** -0.5
    ).to(dtype)
    dout = torch.randn((dims["B"], dims["T"], dims["H"]), device=device, dtype=dtype)
    return {"q": q, "k": k, "v": v, "o_weight": o_weight, "dout": dout}

op = declare_op(
    name="llama3_attention",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.llama3_8b.levels.level2.attention.reference:llama3_attention_forward_ref",
    # The eager baseline is timed through the SDPA branch the model runs. The
    # declared forward materializes a [B, HQ, T, T] score matrix the real
    # execution never allocates, so timing against it would compare every
    # candidate to a strawman.
    runtime_forward=(
        "evograd.benchmark.topdown.llama3_8b.levels.level2.attention.reference:"
        "llama3_attention_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("o_weight", "[H, QO]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("o_weight",),
    forward_semantics="Causal grouped-query attention followed by the output projection. "
        "Expand k and v from HK to HQ heads by repeating each KV head HQ/HK "
        "times; scores = q @ k_expanded^T * (1/sqrt(D)); mask strictly-future "
        "positions to -inf; softmax over the last axis in float32 and cast back "
        "to q's dtype; attn = weights @ v_expanded, shape [B, HQ, T, D]; "
        "transpose heads and tokens, make contiguous and reshape to [B, T, QO] "
        "where QO = HQ*D; out = merged @ o_weight^T, shape [B, T, H]. "
        "There is no attention mask tensor, no dropout, no bias and no KV "
        "cache. Do not call F.scaled_dot_product_attention, F.linear, "
        "torch.matmul, @, F.softmax, or autograd in the generated math.",
    backward_semantics="Return gradients for q, k, v and o_weight IN THIS ORDER. With "
        "merged = attn.transpose(1,2).reshape(B, T, QO): "
        "do_weight = dout^T @ merged summed over B and T (shape [H, QO]); "
        "dmerged = dout @ o_weight, reshaped and transposed back to "
        "[B, HQ, T, D]; then the standard attention backward -- "
        "dv_expanded = weights^T @ dmerged, dweights = dmerged @ v_expanded^T, "
        "dscores = weights * (dweights - sum(dweights * weights, dim=-1, "
        "keepdim=True)) with future positions zeroed, "
        "dq = (dscores @ k_expanded) / sqrt(D) and "
        "dk_expanded = (dscores^T @ q) / sqrt(D). Finally sum dk_expanded and "
        "dv_expanded over each group of HQ/HK query heads to get dk and dv at "
        "HK heads. Accumulate every reduction and matmul in float32 before "
        "casting each gradient to its input's dtype.",
    extra_constraints=(
        "Derived from the published Llama-3-8B configuration, not chosen. The "
        "boundary starts after the q_proj/k_proj/v_proj projections and the "
        "rotary embedding and ends after o_proj -- those earlier stages are "
        "`llama3_qkv_rope` and must not be recomputed here. q, k and v are "
        "non-contiguous with head-major strides because the model transposes "
        "them out of [B, T, heads, D]; a kernel may make them contiguous "
        "internally but the declared inputs are not. HQ must be divisible by "
        "HK. o_weight has no bias. All floating tensors are CUDA tensors."
    ),
    grad_order=("dq", "dk", "dv", "do_weight"),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"llama_3_8b_observed": _BENCHMARK},
    memory_inputs=("q", "k", "v", "o_weight"),
    # The tolerances and the multiplier are Qwen3's measured values for the
    # same computation, carried over because the boundary's numerics are the
    # boundary's. They have not been recalibrated at Llama-3-8B's widths: doing
    # so needs the GPU calibration this workload has not yet run, and inventing
    # a number here would be worse than carrying a measured one.
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (1e-2, 1e-2)},
    tolerance_multipliers={"do_weight": (6.5, 1.0)},
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_llama3_attention_inputs,
)
