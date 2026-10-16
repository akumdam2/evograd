"""Operator declaration: causal attention and ``o_proj`` (site ``attention``).

The second half of ``OlmoeAttention``: what remains after ``olmoe_qkv_norm_rope``
has produced q, k and v. 16 query heads over 16 KV heads at D = 128 in
OLMoE-1B-7B -- multi-head attention, which the grouped-query contract covers as
a group of one.
"""

from evograd.opdecl import Active, Workload, declare_op

from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk, token_major

#: Not calibrated: see ``task_tools.PROVISIONAL``.
PROVISIONAL_TOLERANCES = True

_DIMS = ("B", "T", "HQ", "HK", "D", "QO", "H")


def model_strides(dims):
    """``v`` is ``v_proj(x).view(B, T, HK, D).transpose(1, 2)``: a head-major view."""
    return (("v", token_major(dims["B"], dims["T"], dims["HK"], dims["D"])),)


_BENCHMARK = observed("attention", strides=model_strides)

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, HQ=hq, HK=hk, D=d, QO=hq * d, H=h), dtype=dtype,
             provenance=shrunk("attention", model_strides(dict(B=b, T=t, HK=hk, D=d))))
    for b, t, hq, hk, d, h, dtype in (
        (1, 12, 4, 4, 16, 64, "float32"),
        (2, 20, 8, 2, 16, 64, "float32"),
        (2, 24, 4, 4, 32, 128, "bfloat16"),
    )
)


def make_attention_inputs(torch, op, workload, device="cuda"):
    """q and k contiguous (RoPE's concatenation returns them so); v drawn
    contiguous and re-laid out by the cases that state the model's layout."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 22))
    b, t, hq, hk, d = dims["B"], dims["T"], dims["HQ"], dims["HK"], dims["D"]
    return {
        "q": torch.randn((b, hq, t, d), device=device, dtype=dtype),
        "k": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "v": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "o_weight": (torch.randn((dims["H"], dims["QO"]), device=device)
                     * dims["QO"] ** -0.5).to(dtype),
        "dy": torch.randn((b, t, dims["H"]), device=device, dtype=dtype),
    }


op = declare_op(
    name="olmoe_attention",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.attention.reference:"
            "attention_forward_ref",
    runtime_forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.attention."
                    "reference:attention_forward_production",
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("o_weight", "[H, QO]"),
    ),
    output=Active("y", "[B, T, H]"),
    parameter_args=("o_weight",),
    forward_semantics=(
        "Causal (grouped-query) attention with scale D^-0.5: each KV head serves "
        "HQ/HK consecutive query heads (one each in OLMoE-1B-7B); scores = q @ k^T * "
        "D^-0.5 with strictly future positions masked, softmax in float32, a = "
        "weights @ v, [B, HQ, T, D]. Transpose to [B, T, HQ, D], flatten to "
        "[B, T, QO] and project: y = a @ o_weight^T, [B, T, H] in q's dtype. No mask "
        "tensor, no dropout. Do not call F.scaled_dot_product_attention, F.linear, "
        "@, torch.matmul or autograd in the generated math."
    ),
    backward_semantics=(
        "Return dq, dk, dv, do_weight IN THIS ORDER. da = dy @ o_weight reshaped back "
        "to [B, HQ, T, D]; do_weight = dy^T @ a summed over B and T; then the causal "
        "attention backward, summing dk and dv over each group of query heads. "
        "Accumulate in float32; each gradient in its input's dtype."
    ),
    extra_constraints=(
        "HQ divisible by HK; v may be a non-contiguous transposed view."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=PROVISIONAL,
    make_inputs=make_attention_inputs,
)
