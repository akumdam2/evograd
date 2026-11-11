"""Operator declaration: Qwen3's whole self-attention module (site ``attention_block``).

The boundary is ``Qwen3Attention.forward`` on a training step, from the
normalized residual stream to the output projection:

    x [B, T, H]
      -> q_proj / k_proj / v_proj, viewed as heads
      -> per-head q_norm / k_norm (Qwen3RMSNorm over D)
      -> transpose to head-major, RoPE on q and k
      -> causal grouped-query SDPA (no mask, no dropout)
      -> merge heads, o_proj
      -> out [B, T, H]

Once per decoder layer, 28 calls per step. The residual RMSNorm that produces
``x`` and the residual add after ``out`` belong to ``fused_add_rms_norm``.
``cos``/``sin`` are Inactive: the rotary embedding computes them once per step
and every layer shares them.

The case is config-derived (``hf_config``: :meth:`ModelConfig.attention_block_dims`
of Qwen3-0.6B) at the batch and sequence of the harvested run, so it imports
before the snapshot holds this task; the capture binds every recorded call to
these shapes and the harvest counts the calls per step.
"""

from evograd.benchmark.topdown import load_snapshot as _load_snapshot
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.opdecl.models import QWEN3_0_6B

_WORKLOAD = "qwen3_0_6b"
_SNAPSHOT = _load_snapshot(_WORKLOAD)
_FREE = {"batch": _SNAPSHOT["batch_size"], "seq": _SNAPSHOT["seq_len"]}
_CASE_DIMS = QWEN3_0_6B.attention_block_dims(**_FREE)
_MODEL = _SNAPSHOT["model"]
assert (_CASE_DIMS["H"], _CASE_DIMS["HQ"], _CASE_DIMS["HK"], _CASE_DIMS["D"]) == (
    _MODEL["hidden_size"], _MODEL["num_attention_heads"], _MODEL["num_key_value_heads"],
    _MODEL["head_dim"]), _MODEL
#: ``rms_norm_eps`` as the harvested head norms recorded it.
_EPS = next(c["attrs"]["eps"] for c in _SNAPSHOT["level1"]["rmsnorm"]["configurations"]
            if c["provenance"]["component"] == "q_head_norm")

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_BENCHMARK = (
    Workload(
        dims=_CASE_DIMS,
        dtype="bfloat16",
        provenance=Provenance(model=_WORKLOAD, component="attention_block", free=_FREE,
                              source="hf_config"),
    ),
)


def make_qwen3_attention_block_inputs(torch, op, workload, device="cuda"):
    """Weights at an initialised model's scale, norm weights near 1, real rotary tables."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(dims["B"] * 1000003 + dims["T"] * 10007 + dims["H"] * 1009 + dims["D"])
    hidden, head_dim = dims["H"], dims["D"]

    def projection(out_features, in_features):
        return (torch.randn((out_features, in_features), device=device, dtype=torch.float32)
                * in_features**-0.5).to(dtype)

    def norm_weight():
        return (1.0 + 0.02 * torch.randn((head_dim,), device=device,
                                         dtype=torch.float32)).to(dtype)

    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (QWEN3_0_6B.rope_theta ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None, :, :]
    return {
        "x": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"], hidden),
        "k_weight": projection(dims["KVO"], hidden),
        "v_weight": projection(dims["KVO"], hidden),
        "o_weight": projection(hidden, dims["QO"]),
        "q_norm_weight": norm_weight(),
        "k_norm_weight": norm_weight(),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "eps": _EPS,
        "dout": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
    }


op = declare_op(
    name="qwen3_attention_block",
    level=2,
    family="attention",
    forward=(
        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2.attention_block.reference:"
        "qwen3_attention_block_forward"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("o_weight", "[H, QO]"),
        Active("q_norm_weight", "[D]"),
        Active("k_norm_weight", "[D]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
        Inactive("eps", None, default=_EPS),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("q_weight", "k_weight", "v_weight", "o_weight", "q_norm_weight",
                    "k_norm_weight"),
    forward_semantics=(
        "Qwen3's self-attention module, all in x's dtype except where stated. "
        "q = x @ q_weight^T viewed as [B, T, HQ, D]; per-head RMSNorm over D: h32 = "
        "float32(h), h32 * rsqrt(mean(h32^2, -1) + eps), cast back to x's dtype, then "
        "times q_norm_weight; transpose to [B, HQ, T, D]. k the same with k_weight and "
        "k_norm_weight at HK heads. v = x @ v_weight^T viewed as [B, T, HK, D] and "
        "transposed, not normalized. RoPE on q and k with cos/sin broadcast over batch "
        "and heads: t*cos + rotate_half(t)*sin, rotate_half(t) = cat(-t[..., D/2:], "
        "t[..., :D/2]). Causal attention, each KV head serving HQ/HK consecutive query "
        "heads: softmax(q k^T * D^-0.5 with future positions masked) @ v, softmax in "
        "float32. Transpose to [B, T, HQ, D], flatten to [B, T, QO] and project: out = "
        "a @ o_weight^T, [B, T, H]. No biases, no mask tensor, no dropout, no cache."
    ),
    backward_semantics=(
        "Return dx, dq_weight, dk_weight, dv_weight, do_weight, dq_norm_weight, "
        "dk_norm_weight IN THIS ORDER; cos, sin and eps get none. da = dout @ o_weight "
        "back to [B, HQ, T, D], do_weight = dout^T @ a over B and T; the causal "
        "attention backward gives dq, dk, dv (dk, dv summed over each query-head group); "
        "un-rotate dq and dk: dt*cos - rotate_half(dt*sin); the RMSNorm backward per "
        "head (xhat = h * r, r = rsqrt(mean(h^2)+eps), g = dn * norm_weight, dh = r * "
        "(g - xhat * mean(g * xhat, -1)), dnorm_weight = sum of dn * xhat over B, T and "
        "heads); then dx = dq_flat @ q_weight + dk_flat @ k_weight + dv_flat @ v_weight "
        "and each projection's weight gradient. Accumulate in float32; every gradient "
        "in its input's dtype."
    ),
    extra_constraints=(
        "The boundary is the whole Qwen3Attention module of a training step: x is the "
        "input_layernorm output, out goes to the residual add. cos and sin are the "
        "shared rotary tables' first row (the model's rows are identical). HQ is a "
        "multiple of HK and D <= 256. All tensors are contiguous CUDA tensors."
    ),
    grad_order=("dx", "dq_weight", "dk_weight", "dv_weight", "do_weight",
                "dq_norm_weight", "dk_norm_weight"),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"qwen3_0_6b_observed": _BENCHMARK},
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "o_weight", "q_norm_weight",
                   "k_norm_weight", "cos", "sin"),
    make_inputs=make_qwen3_attention_block_inputs,
)
