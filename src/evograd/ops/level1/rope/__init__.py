"""Operator declaration: rotary position embedding (RoPE).

One of the seven kernels the Liger paper reports, and one of only three that
``apply_liger_kernel_to_llama`` enables by default. Evograd had no declaration
for it before v1, so the position-encoding path of every LLM benchmark went
unmeasured.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import (
    LLAMA_3_8B,
    LLAMA_REGIME_SPLIT,
    LLAMA_TOKEN_SWEEP,
    MODELS,
)
from evograd.ops._common import (
    fixed_shape_suites,
    is_qwen3_observed,
    log_distance_weight,
    make_pair_baseline,
    model_workloads,
    qwen3_observed_workloads,
    regime_suites,
)

_DIMS = ("B", "n_heads", "T", "head_dim")


def _regime_feature(workload: Workload) -> float:
    return float(workload.dims["T"])


def make_rope_inputs(torch, op, workload, device="cuda"):
    """Build real rotary tables. The default would fill them with zeros.

    A zero ``cos``/``sin`` turns RoPE into the zero map. Nothing would crash and
    correctness would still pass, because the oracle sees the same zeros — the
    benchmark would simply be measuring a different, trivial function. This hook
    is the reason the declaration is trustworthy.
    """
    dims = workload.dims
    batch, heads = dims["B"], dims["n_heads"]
    tokens, head_dim = dims["T"], dims["head_dim"]
    if head_dim % 2:
        raise ValueError(f"rope needs an even head_dim, got {head_dim}")
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(tokens * 100003 + heads * 1009 + head_dim)

    # A model reaches RoPE by projecting into [B, T, heads, head_dim] and
    # transposing, so the tensor it rotates is a non-contiguous head-major view.
    # The observed Qwen3 configurations are reproduced that way; the historical
    # Llama grid keeps the contiguous layout it has always been measured at, so
    # its numbers stay comparable with earlier runs.
    observed = is_qwen3_observed(workload)

    def draw():
        if observed:
            token_major = torch.randn(
                (batch, tokens, heads, head_dim), device=device, dtype=dtype
            )
            return (token_major * 0.5).transpose(1, 2)
        return torch.randn((batch, heads, tokens, head_dim), device=device, dtype=dtype) * 0.5

    x = draw()
    # theta = 500000 is Llama-3's value and 1000000 is Qwen3's. Llama-2's 10000
    # produces a kernel that is self-consistent and completely wrong, which is
    # exactly the kind of error a benchmark should not be able to express by
    # accident -- so the base comes from the workload's own provenance rather
    # than from a constant.
    config = MODELS[workload.provenance.model] if workload.provenance else LLAMA_3_8B
    positions = torch.arange(tokens, device=device, dtype=torch.float32)[:, None]
    inv_freq = config.rope_theta ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    freqs = positions * inv_freq[None, :]
    # Duplicated halves, matching LlamaRotaryEmbedding and what Liger's kernel
    # assumes when it reads only the first head_dim // 2 entries of each row.
    angles = torch.cat((freqs, freqs), dim=-1)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)

    # The rotated output carries the input's layout, so its gradient does too.
    dy = draw()
    return {"x": x, "cos": cos, "sin": sin, "dy": dy}


def _liger_factory():
    from evograd.ops.level1.rope.liger import make_liger_rope_autograd_pair_fns

    return make_liger_rope_autograd_pair_fns()


# Both head counts a GQA layer rotates: the 32-head query tensor and the 8-head
# key tensor. They run the same kernel at different occupancy, so both belong in
# the timed grid.
_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "rope",
    tuple({"batch": 1, "seq": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
) + model_workloads(
    LLAMA_3_8B,
    "rope_kv",
    tuple({"batch": 1, "seq": tokens} for tokens in LLAMA_TOKEN_SWEEP),
    ("bfloat16",),
)

#: Both tensors one Qwen3-0.6B layer rotates: 16 query heads and 8 key heads,
#: at batch 2 x sequence 2048 x head_dim 128, 28 times per step. They come from
#: one harvested `apply_rotary_pos_emb` record, which rotates q and k together.
_QWEN3_OBSERVED = qwen3_observed_workloads("rope")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, n_heads=h, T=t, head_dim=d), dtype=dtype)
    for b, h, t, d in (
        (1, 4, 16, 64),
        (2, 8, 128, 128),
        # Non-power-of-two token count: the kernel launches one program per
        # token, so a ragged count exercises the tail path.
        (1, 32, 129, 128),
    )
    for dtype in ("float32", "float16", "bfloat16")
)

op = declare_op(
    name="rope",
    forward="evograd.ops.level1.rope.forward_ref:rope_forward_ref",
    # The eager baseline is timed through the exact Transformers spelling, which
    # rotates entirely in the model dtype. The declared forward upcasts to
    # float32 -- more accurate, and slower by casts the model never executes.
    runtime_forward="evograd.ops.level1.rope.forward_ref:rope_runtime_ref",
    level=1,
    family="positional",
    dims=_DIMS,
    args=(
        Active("x", "[B, n_heads, T, head_dim]"),
        Inactive(
            "cos",
            "[T, head_dim]",
            note="rotary cosine table; both halves duplicated, as LlamaRotaryEmbedding emits",
        ),
        Inactive("sin", "[T, head_dim]", note="rotary sine table, same layout as cos"),
    ),
    output=Active("y", "[B, n_heads, T, head_dim]"),
    parameter_args=(),
    forward_semantics=(
        "Apply half-rotated (GPT-NeoX / Llama) rotary embedding to x, laid out "
        "as [batch, heads, tokens, head_dim]. With x1, x2 the two halves of the "
        "head dimension: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]. "
        "cos and sin are [tokens, head_dim] with duplicated halves and broadcast "
        "over batch and heads. Evaluate the products in float32 and cast back to "
        "x's dtype. Do NOT use the interleaved (GPT-J) convention — it is a "
        "different function and will pass no correctness case. Do not call "
        "apply_rotary_pos_emb or any high-level rotary helper."
    ),
    backward_semantics=(
        "Return dx only; cos and sin are inactive tables and receive no "
        "gradient. The rotation is orthogonal, so the backward is the same "
        "rotation with the sine negated: "
        "dx = [dy1, dy2] * [cos, cos] + [dy2, -dy1] * [sin, sin]. "
        "dx has x's shape and dtype. Accumulate in float32."
    ),
    extra_constraints=(
        "Tensor layout notes:\n"
        "- x, dy, y: [batch, heads, tokens, head_dim], contiguous CUDA\n"
        "- cos, sin: [tokens, head_dim], same dtype as x, broadcast over batch and heads\n"
        "- head_dim must be even; the two halves of cos/sin are equal\n"
        "- This is memory-bound: one read and one write of x per element, plus a "
        "shared row of cos/sin per token. Fusing the two halves into one pass is "
        "the point."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK + _QWEN3_OBSERVED,
    benchmark=_BENCHMARK,
    benchmark_suites={
        "qwen3_0_6b_observed": _QWEN3_OBSERVED,
        **regime_suites(_BENCHMARK, _regime_feature, LLAMA_REGIME_SPLIT),
        **fixed_shape_suites(_BENCHMARK),
    },
    performance_baselines={
        "liger": make_pair_baseline(_liger_factory, ("x", "cos", "sin"))
    },
    # Verified against the new `runtime_forward` rather than re-derived.
    # `level1 calibrate --op rope` measures the oracle-vs-Transformers
    # disagreement at worst 6.0e-03 (bfloat16, the observed 16-head case),
    # 6.5e-04 (float16) and exactly 0.0 (float32) -- 13x, 77x and unbounded
    # margins against the pair below. They are left alone because the same pair
    # gates the reviewed Liger rope baseline and the historical Llama grid;
    # tightening them is a separate change with its own blast radius.
    tolerances={
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    },
    # cos/sin are recomputed per step by the model, not stored activations, so
    # they do not belong in the saved-memory budget.
    memory_inputs=("x",),
    regime_feature=_regime_feature,
    regime_split=LLAMA_REGIME_SPLIT,
    case_weight=log_distance_weight(_regime_feature, LLAMA_REGIME_SPLIT),
    make_inputs=make_rope_inputs,
)
