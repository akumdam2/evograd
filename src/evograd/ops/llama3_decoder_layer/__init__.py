"""Operator declaration: one Llama-3-8B decoder layer (benchmark level 3).

Levels 1 and 2 ask whether a kernel is fast in isolation. This asks whether the
choices survive composition: what the block saves for backward is now a
whole-layer decision, and a candidate is free to fuse across the norm /
projection / attention / MLP boundaries that the smaller tasks draw.

Correctness at this depth needs care, and the declaration handles it three ways:

* ``reference_dtype="float32"`` — the reference runs in float32 while the
  candidate runs in bfloat16, so the tolerance bounds the candidate's own error
  instead of the difference between two equally-rounded computations. Composing
  ten operators makes that distinction the difference between a meaningful gate
  and a fudge factor.
* float32 correctness cases at small dimensions. A bfloat16-only gate cannot
  separate a real bug — an RMSNorm that skips its float32 upcast, say — from
  ordinary rounding, because both land at the same magnitude.
* the upstream gradient is scaled by ``1/sqrt(batch*tokens)`` in ``make_inputs``.
  Relative error at this depth is flat in the token count, but absolute
  magnitudes grow like the square root of it, so an unscaled gradient would need
  a different tolerance at every workload. This is also what a real
  mean-over-tokens loss produces.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.ops._common import fixed_shape_suites, model_workloads

_DIMS = ("B", "T", "hidden", "head_dim", "q_out", "kv_out", "intermediate")

# Timed grid. The ceiling is set by the oracle, not the candidate: the reference
# builds an autograd graph for the whole layer, and at float32 that is the
# largest tensor set in the whole suite. 4096 tokens keeps one case inside a
# 40 GB card with the baseline resident alongside it; 8192 lives in untimed
# coverage, where no graph is built.
_TIMED_TOKENS = (1024, 2048, 4096)
_COVERAGE_TOKENS = (8192,)

_BENCHMARK = model_workloads(
    LLAMA_3_8B,
    "decoder_layer",
    tuple({"batch": 1, "seq": tokens} for tokens in _TIMED_TOKENS),
    ("bfloat16",),
)
_COVERAGE = _BENCHMARK + model_workloads(
    LLAMA_3_8B,
    "decoder_layer",
    tuple({"batch": 1, "seq": tokens} for tokens in _COVERAGE_TOKENS),
    ("bfloat16",),
)

# Correctness runs a proportionally scaled block: same head/group structure,
# same GQA ratio, small enough to run on CPU in a unit test. The ratios are what
# the algebra depends on; the absolute widths are not.
_SMALL = dict(B=1, T=32, hidden=128, head_dim=32, q_out=128, kv_out=32, intermediate=352)
_SMALL_2 = dict(B=2, T=17, hidden=64, head_dim=16, q_out=64, kv_out=16, intermediate=176)

_CORRECTNESS = (
    # float32 is the gate that can actually see an algebra error.
    Workload(dims=_SMALL, dtype="float32", atol=1e-4, rtol=1e-4),
    Workload(dims=_SMALL_2, dtype="float32", atol=1e-4, rtol=1e-4),
    # bfloat16 gates dtype and casting handling, against a float32 reference.
    Workload(dims=_SMALL, dtype="bfloat16"),
    Workload(dims=_SMALL_2, dtype="bfloat16"),
)


def make_llama3_decoder_layer_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    batch, tokens, hidden = dims["B"], dims["T"], dims["hidden"]
    head_dim, q_out, kv_out = dims["head_dim"], dims["q_out"], dims["kv_out"]
    intermediate = dims["intermediate"]

    # Structural invariants, checked loudly rather than mis-computed quietly —
    # the same treatment fused_moe_swiglu gives its G == 2*I relation.
    if q_out % head_dim or kv_out % head_dim:
        raise ValueError("q_out and kv_out must be whole multiples of head_dim")
    n_heads, n_kv_heads = q_out // head_dim, kv_out // head_dim
    if n_heads % n_kv_heads:
        raise ValueError(
            f"GQA needs n_heads divisible by n_kv_heads, got {n_heads} and {n_kv_heads}"
        )
    if q_out != hidden:
        raise ValueError("Llama-3 ties n_heads*head_dim to hidden")

    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(tokens * 100003 + hidden * 1009 + head_dim)

    def weight(out_features, in_features):
        # fan-in scaling. Ten unit-variance projections in sequence would
        # overflow bfloat16's range long before the layer finished.
        return (
            torch.randn((out_features, in_features), device=device, dtype=torch.float32)
            * in_features**-0.5
        ).to(dtype)

    def gain(width):
        # Norm gains sit near 1.0 in a trained model. A zero-mean gain would
        # cancel the signal path, which matters far more here than for a
        # standalone norm.
        return (
            1.0 + 0.1 * torch.randn((width,), device=device, dtype=torch.float32)
        ).to(dtype)

    positions = torch.arange(tokens, device=device, dtype=torch.float32)[:, None]
    inv_freq = LLAMA_3_8B.rope_theta ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    angles = torch.cat((positions * inv_freq[None, :],) * 2, dim=-1)

    return {
        "x": torch.randn((batch, tokens, hidden), device=device, dtype=dtype) * 0.5,
        "input_norm_weight": gain(hidden),
        "q_weight": weight(q_out, hidden),
        "k_weight": weight(kv_out, hidden),
        "v_weight": weight(kv_out, hidden),
        "o_weight": weight(hidden, q_out),
        "post_norm_weight": gain(hidden),
        "gate_weight": weight(intermediate, hidden),
        "up_weight": weight(intermediate, hidden),
        "down_weight": weight(hidden, intermediate),
        "cos": angles.cos().to(dtype),
        "sin": angles.sin().to(dtype),
        "eps": 1e-5,
        # Scaled by 1/sqrt(batch*tokens): see the module docstring. Without it
        # the gradient magnitudes grow with the token count and no single
        # tolerance covers the sweep.
        "dout": (
            torch.randn((batch, tokens, hidden), device=device, dtype=dtype)
            * 0.5
            * (batch * tokens) ** -0.5
        ),
    }


op = declare_op(
    name="llama3_decoder_layer",
    forward=(
        "evograd.ops.llama3_decoder_layer.forward_ref:"
        "llama3_decoder_layer_forward_ref"
    ),
    level=3,
    family="llm_block",
    reference_dtype="float32",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, hidden]"),
        Active("input_norm_weight", "[hidden]", note="RMSNorm gain before attention"),
        Active("q_weight", "[q_out, hidden]"),
        Active("k_weight", "[kv_out, hidden]", note="fewer heads than q under GQA"),
        Active("v_weight", "[kv_out, hidden]"),
        Active("o_weight", "[hidden, q_out]"),
        Active("post_norm_weight", "[hidden]", note="RMSNorm gain before the MLP"),
        Active("gate_weight", "[intermediate, hidden]"),
        Active("up_weight", "[intermediate, hidden]"),
        Active("down_weight", "[hidden, intermediate]"),
        Inactive("cos", "[T, head_dim]", note="rotary table, duplicated halves"),
        Inactive("sin", "[T, head_dim]", note="rotary table, duplicated halves"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("out", "[B, T, hidden]"),
    forward_semantics=(
        "One Llama-3 decoder layer over x of shape [batch, tokens, hidden]: "
        "RMSNorm -> Q/K/V projections -> half-rotated RoPE on Q and K -> causal "
        "grouped-query attention -> output projection -> residual add -> RMSNorm "
        "-> SwiGLU MLP (silu(gate) * up, then down) -> residual add. "
        "Projection weights are in nn.Linear [out, in] orientation. Head counts "
        "come from q_out // head_dim and kv_out // head_dim; each key/value head "
        "serves n_heads // n_kv_heads query heads. Compute RMSNorm statistics, "
        "the SiLU, and the attention softmax in float32. Attention is causal. "
        "This is the training forward pass: there is no KV cache, no dropout, "
        "and no attention-weight output. Do not call any high-level transformer "
        "or attention module in the generated math."
    ),
    backward_semantics=(
        "Return gradients for all ten Active inputs in declaration order: "
        "dx, dinput_norm_weight, dq_weight, dk_weight, dv_weight, do_weight, "
        "dpost_norm_weight, dgate_weight, dup_weight, ddown_weight. "
        "cos, sin and eps are inactive and receive no gradient. Both residual "
        "connections contribute to dx, so it accumulates three paths: the two "
        "residual adds and the norm inputs. Under GQA the key/value gradients "
        "must be summed over the query heads that share each kv head. Every "
        "gradient carries its input's dtype. Accumulate reductions in float32."
    ),
    extra_constraints=(
        "This block is where the saved-state choice becomes interesting: the "
        "forward may keep the attention output, the MLP intermediates, the norm "
        "statistics, or recompute any of them. Recomputation is cheap relative "
        "to the projections and the saved-memory term rewards it.\n"
        "- x, out, dout: [batch, tokens, hidden], contiguous CUDA\n"
        "- cos, sin: [tokens, head_dim], duplicated halves, broadcast over batch and heads\n"
        "- q_out == hidden for Llama-3; kv_out is smaller (GQA)\n"
        "- The reference uses scaled_dot_product_attention, whose efficient "
        "backends accumulate dq with atomics and are therefore not bitwise "
        "reproducible run to run. The declared tolerances leave room for it."
    ),
    correctness=_CORRECTNESS,
    coverage=_COVERAGE,
    benchmark=_BENCHMARK,
    benchmark_suites=fixed_shape_suites(_BENCHMARK),
    tolerances={
        # float32 gates the algebra.
        "float32": (1e-4, 1e-4),
        # bfloat16 against a float32 reference. atol carries the gate; rtol is
        # near-useless at block level because the gradient tensors span a huge
        # dynamic range and their near-zero entries carry absolute noise.
        "bfloat16": (5e-2, 2e-2),
    },
    tolerance_multipliers={
        # The three reductions that sum over the most terms.
        "dk_weight": (1.5, 1.0),
        "dv_weight": (1.5, 1.0),
        "ddown_weight": (1.5, 1.0),
    },
    # cos/sin are recomputed once per step and shared by every layer; eps is a
    # scalar. Neither is layer activation state, so neither belongs in the
    # memory budget.
    memory_inputs=(
        "x",
        "input_norm_weight",
        "q_weight",
        "k_weight",
        "v_weight",
        "o_weight",
        "post_norm_weight",
        "gate_weight",
        "up_weight",
        "down_weight",
    ),
    make_inputs=make_llama3_decoder_layer_inputs,
)
