"""Level-2 task: Llama-3.2-1B's whole self-attention module.

``LlamaAttention.forward`` as one boundary, from the normalized residual stream
to the output projection:

    x -> q_proj, k_proj, v_proj -> head-major view -> RoPE on q and k
      -> causal grouped-query SDPA -> merge heads -> o_proj -> out

It is the module boundary the capture records (``x`` and ``cos``/``sin`` in,
the output and its cotangent out), so no intermediate tensor of the module has
to be reconstructed. There is no per-head query/key RMSNorm and no bias.
``cos``/``sin`` are Inactive: ``LlamaRotaryEmbedding`` computes them once per
step from the position ids and every layer reads the same tables.

The case is derived from the published configuration (``LLAMA_3_2_1B``) at the
canonical batch 2 x sequence 2048; ``tests/test_provenance`` re-derives it.
Once per decoder layer: 16 calls per step.
"""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.opdecl.models import LLAMA_3_2_1B
from evograd.ops._common import model_workloads as _model_workloads

#: The workload these cases belong to; also the ``Provenance`` model key.
WORKLOAD = "llama_3_2_1b"

#: 16 -- once per decoder layer.
FREQUENCY = LLAMA_3_2_1B.layers

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_REFERENCE = "evograd.benchmark.topdown.llama3_2_1b.levels.level2.attention_block.reference"

#: The observed configuration, computed from the published model config.
_BENCHMARK = _model_workloads(
    LLAMA_3_2_1B,
    "attention_block",
    ({"batch": 2, "seq": 2048},),
    ("bfloat16",),
)


def make_llama3_attention_block_inputs(torch, op, workload, device="cuda"):
    """Inputs at a real step's magnitudes, with real rotary tables.

    ``cos``/``sin`` come from Llama-3's inverse-frequency schedule (``rope_theta``
    500000) rather than random draws: a random pair is not a rotation.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["H"] * 1009 + dims["D"]
    )
    hidden, head_dim = dims["H"], dims["D"]

    def projection(out_features, in_features):
        return (
            torch.randn((out_features, in_features), device=device, dtype=torch.float32)
            * in_features**-0.5
        ).to(dtype)

    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (
        LLAMA_3_2_1B.rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None, :, :]
    return {
        "x": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"], hidden),
        "k_weight": projection(dims["KVO"], hidden),
        "v_weight": projection(dims["KVO"], hidden),
        "o_weight": projection(hidden, dims["QO"]),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "dout": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
    }


op = declare_op(
    name="llama3_attention_block",
    level=2,
    family="attention",
    forward=f"{_REFERENCE}:llama3_attention_block_forward_ref",
    # The module's own arithmetic: RoPE in the model dtype and the fused SDPA
    # call the model makes. The declared forward rotates in float32 and
    # materializes a [B, HQ, T, T] score matrix the model never allocates.
    runtime_forward=f"{_REFERENCE}:llama3_attention_block_forward_production",
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Active("o_weight", "[H, QO]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
    ),
    output=Active("out", "[B, T, H]"),
    parameter_args=("q_weight", "k_weight", "v_weight", "o_weight"),
    forward_semantics=(
        "Llama's whole self-attention module. Project: q = x @ q_weight^T "
        "([B, T, QO]), viewed as [B, T, HQ, D] and transposed to [B, HQ, T, D]; "
        "k and v the same with k_weight and v_weight at HK heads. There is NO "
        "per-head normalization. Rotate q and k (not v): with cos and sin "
        "unsqueezed at dim 1 and rotate_half(t) = cat((-t[..., D/2:], "
        "t[..., :D/2]), -1), t_rot = t*cos + rotate_half(t)*sin. Then causal "
        "grouped-query attention: expand k and v from HK to HQ heads by "
        "repeating each KV head HQ/HK times; scores = q_rot @ k_rot_expanded^T "
        "* (1/sqrt(D)); mask strictly-future positions to -inf; softmax over the "
        "last axis in float32 and cast back to x's dtype; attn = weights @ "
        "v_expanded ([B, HQ, T, D]). Transpose heads and tokens, make "
        "contiguous and reshape to [B, T, QO]; out = merged @ o_weight^T, shape "
        "[B, T, H]. No bias, no attention mask tensor, no dropout, no KV cache. "
        "Do not call F.linear, torch.matmul, @, F.scaled_dot_product_attention, "
        "F.softmax, or autograd in the generated math."
    ),
    backward_semantics=(
        "Return gradients for x, q_weight, k_weight, v_weight and o_weight IN "
        "THIS ORDER. cos and sin are inactive and get no gradient. Output "
        "projection: do_weight = dout^T @ merged summed over B and T ([H, QO]); "
        "dmerged = dout @ o_weight, viewed as [B, T, HQ, D] and transposed to "
        "dattn [B, HQ, T, D]. Attention, with P the causal softmax weights: "
        "dv_expanded = P^T @ dattn; dP = dattn @ v_expanded^T; "
        "dS = P * (dP - sum(dP * P, dim=-1, keepdim=True)), future positions "
        "zero; dq_rot = (dS @ k_rot_expanded) / sqrt(D); dk_rot_expanded = "
        "(dS^T @ q_rot) / sqrt(D); sum dk_rot_expanded and dv_expanded over each "
        "group of HQ/HK query heads to get dk_rot and dv at HK heads. Unrotate: "
        "dq = dq_rot*cos - rotate_half(dq_rot*sin), the same for k; dv is not "
        "rotated. Flatten each to [B, T, heads*D] (transpose(1, 2) then "
        "reshape) as dq_flat, dk_flat, dv_flat; dq_weight = dq_flat^T @ x summed "
        "over B and T, and likewise dk_weight and dv_weight; dx = dq_flat @ "
        "q_weight + dk_flat @ k_weight + dv_flat @ v_weight. Accumulate every "
        "reduction and matmul in float32 before casting each gradient to its "
        "input's dtype."
    ),
    extra_constraints=(
        "Derived from the published Llama-3.2-1B configuration, not chosen. The "
        "boundary is the whole LlamaAttention module: it starts at the "
        "normalized residual stream x and ends after o_proj. The RMSNorm that "
        "produces x and the residual add that consumes out are outside it. cos "
        "and sin are shared tables computed once per step, so they are inputs "
        "and receive no gradient. x and out are contiguous [B, T, H]. HQ must "
        "be divisible by HK, and the projection fan-outs are QO = HQ*D and "
        "KVO = HK*D. QO equals H for Llama-3.2-1B; do not rely on that. All "
        "floating tensors are CUDA tensors."
    ),
    grad_order=("dx", "dq_weight", "dk_weight", "dv_weight", "do_weight"),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={"llama_3_2_1b_observed": _BENCHMARK},
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "o_weight", "cos", "sin"),
    make_inputs=make_llama3_attention_block_inputs,
)
