"""Operator declaration: causal GQA attention, sigmoid output gate, o_proj
(site ``gated_attention``).

The second half of ``Qwen3_5Attention``: what remains after
``qwen3_5_qkv_norm_partial_rope`` has produced q, k, v and the gate. 16 query
heads over 4 KV heads at D = 256 in Qwen3.5-4B.
"""

from evograd.opdecl import Active, declare_op

from ..task_tools import OBSERVED_SUITE, observed, seed_of, token_major


_DIMS = ("B", "T", "HQ", "HK", "D", "QO", "H")


def model_strides(dims):
    """``v`` is ``v_proj(x).view(B, T, HK, D).transpose(1, 2)``: a head-major view."""
    return (("v", token_major(dims["B"], dims["T"], dims["HK"], dims["D"])),)


_BENCHMARK = observed("gated_attention", strides=model_strides)


def make_gated_attention_inputs(torch, op, workload, device="cuda"):
    """q and k contiguous (RoPE's concatenation returns them so); v drawn
    contiguous and re-laid out by the cases that state the model's layout."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 15))
    b, t, hq, hk, d = dims["B"], dims["T"], dims["HQ"], dims["HK"], dims["D"]
    return {
        "q": torch.randn((b, hq, t, d), device=device, dtype=dtype),
        "k": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "v": torch.randn((b, hk, t, d), device=device, dtype=dtype),
        "gate": torch.randn((b, t, dims["QO"]), device=device, dtype=dtype),
        "o_weight": (torch.randn((dims["H"], dims["QO"]), device=device)
                     * dims["QO"] ** -0.5).to(dtype),
        "dy": torch.randn((b, t, dims["H"]), device=device, dtype=dtype),
    }


op = declare_op(
    name="qwen3_5_gated_attention",
    level=2,
    family="attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_attention.reference:"
            "gated_attention_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_attention."
                    "reference:gated_attention_forward_production",
    dims=_DIMS,
    args=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
        Active("gate", "[B, T, QO]"),
        Active("o_weight", "[H, QO]"),
    ),
    output=Active("y", "[B, T, H]"),
    parameter_args=("o_weight",),
    forward_semantics=(
        "Causal grouped-query attention with scale D^-0.5: each KV head serves "
        "HQ/HK consecutive query heads; scores = q @ k^T * D^-0.5 with strictly "
        "future positions masked, softmax in float32, a = weights @ v, "
        "[B, HQ, T, D]. Transpose to [B, T, HQ, D], flatten to [B, T, QO], "
        "multiply by sigmoid(gate) elementwise and project: y = (a * "
        "sigmoid(gate)) @ o_weight^T, [B, T, H] in q's dtype. No mask tensor, no "
        "dropout. Do not call F.scaled_dot_product_attention, F.linear, @, "
        "torch.matmul or autograd in the generated math."
    ),
    backward_semantics=(
        "Return dq, dk, dv, dgate, do_weight IN THIS ORDER. dag = dy @ o_weight, "
        "do_weight = dy^T @ (a * sigmoid(gate)) summed over B and T; "
        "dgate = dag * a * s * (1 - s) with s = sigmoid(gate); da = dag * s, "
        "reshaped back to [B, HQ, T, D]; then the causal attention backward, "
        "summing dk and dv over each group of query heads. Accumulate in "
        "float32; each gradient in its input's dtype."
    ),
    extra_constraints=(
        "HQ divisible by HK; head dim D may be 256. v may be a non-contiguous "
        "transposed view."
    ),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    make_inputs=make_gated_attention_inputs,
)
