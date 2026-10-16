"""Operator declaration: OLMoE's routed SwiGLU experts (site ``moe_experts``).

Every one of the 16 layers runs it after the router: 4096 tokens, 64 experts of
2048 -> 1024 -> 2048, 8 per token, so 32,768 token-expert pairs per layer. The
contract is the operator suite's ``fused_moe_swiglu`` boundary at this model's
widths and routing, under this model's own key, cases and provenance -- with
``top_k_index`` in the ``int64`` the model's ``torch.topk`` produces, and
``top_k_weights`` the raw softmax probabilities (``norm_topk_prob`` is false).
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op

from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Not calibrated: see ``task_tools.PROVISIONAL``. The weight gradients sum over
#: every token routed to an expert (~512 at the observed size) and are the
#: likeliest to need a reduction-scaled gate once measured.
PROVISIONAL_TOLERANCES = True

_DIMS = ("T", "H", "E", "G", "I", "K")

_BENCHMARK = observed("moe_experts")


def _dims(t, h, e, i, k):
    return {"T": t, "H": h, "E": e, "G": 2 * i, "I": i, "K": k}


_CORRECTNESS = tuple(
    Workload(dims=_dims(*shape), dtype=dtype, provenance=shrunk("moe_experts"))
    for shape, dtype in (
        ((16, 64, 4, 32, 2), "float32"),
        # 8 experts, 1 of 8 active: every expert sees a different token count.
        ((40, 32, 8, 16, 1), "float32"),
        ((48, 64, 8, 32, 4), "bfloat16"),
    )
)


def make_moe_experts_inputs(torch, op, workload, device="cuda"):
    """Unit activations, projections at ``fan_in^-0.5``, and routing drawn the
    way the model draws it: a softmax over random router logits, top-k, the
    chosen probabilities unnormalized -- so expert loads are uneven."""
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 23))
    t, h, e, g, i, k = (dims[n] for n in ("T", "H", "E", "G", "I", "K"))
    logits = torch.randn((t, e), device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    top_value, top_index = torch.topk(probs, k, dim=-1)
    return {
        "x": torch.randn((t, h), device=device, dtype=dtype),
        "gate_up_proj": (torch.randn((e, g, h), device=device) * h ** -0.5).to(dtype),
        "down_proj": (torch.randn((e, h, i), device=device) * i ** -0.5).to(dtype),
        "top_k_index": top_index.contiguous(),
        "top_k_weights": top_value.to(dtype).contiguous(),
        "dout": torch.randn((t, h), device=device, dtype=dtype),
    }


op = declare_op(
    name="olmoe_moe_experts",
    level=2,
    family="moe",
    forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.moe_experts.reference:"
            "moe_experts_forward_ref",
    runtime_forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.moe_experts."
                    "reference:moe_experts_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[T, H]"),
        Active("gate_up_proj", "[E, G, H]"),
        Active("down_proj", "[E, H, I]"),
        Inactive("top_k_index", "[T, K]", dtype="int64"),
        Active("top_k_weights", "[T, K]"),
    ),
    output=Active("out", "[T, H]"),
    parameter_args=("gate_up_proj", "down_proj"),
    forward_semantics=(
        "For each token t and each of its K routed experts e = top_k_index[t, j]: "
        "gu = x[t] @ gate_up_proj[e]^T ([G] = [2I]; the first I entries are the "
        "gate, the next I the up projection), hidden = silu(gate) * up computed in "
        "float32, y = hidden @ down_proj[e]^T ([H]). out[t] = sum over j of "
        "top_k_weights[t, j] * y, accumulated in float32 and cast to x's dtype. "
        "top_k_index is fixed routing metadata and non-differentiable; the "
        "weights are not renormalized. Accumulate every GEMM in float32. Do not "
        "call F.linear, torch.matmul, @, torch.einsum, F.silu or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return gradients for x, gate_up_proj, down_proj and top_k_weights IN THIS "
        "ORDER; none for top_k_index. For each token-expert pair with weight w: "
        "dy = w * dout[t]; dtop_k_weights[t, j] = sum(dout[t] * y); "
        "ddown_proj[e] += dy^T outer hidden; dhidden = dy @ down_proj[e]; dup = "
        "dhidden * silu(gate); dgate = dhidden * up * dsilu(gate) with dsilu(v) = "
        "sigmoid(v) * (1 + v * (1 - sigmoid(v))); dgate_up_proj[e] += [dgate | dup]^T "
        "outer x[t]; dx[t] += [dgate | dup] @ gate_up_proj[e]. Accumulate every "
        "reduction in float32 before casting each gradient to its input's dtype."
    ),
    extra_constraints=(
        "G = 2 * I. top_k_index is int64 in [0, E) with no sentinel; every token "
        "has exactly K distinct experts. Expert loads are uneven: a kernel may not "
        "assume T * K / E tokens per expert. All floating tensors are contiguous "
        "CUDA tensors in one dtype."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=PROVISIONAL,
    memory_inputs=("x", "gate_up_proj", "down_proj", "top_k_weights"),
    make_inputs=make_moe_experts_inputs,
)
