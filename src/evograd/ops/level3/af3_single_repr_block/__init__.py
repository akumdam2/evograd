"""Operator declaration: AlphaFold3 single-representation block (level 3).

The protein-model counterpart to ``llama3_decoder_layer``. Between them the two
level-3 tasks cover the two architectures this benchmark's operators were
actually drawn from — Liger's LLM kernels and MegaFold's AlphaFold3 kernels — so
a candidate is measured on composition in both, not only in the transformer.

Scope is narrower than the name of a full pairformer block would suggest; see
``forward_ref`` for what is excluded and why.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import fixed_shape_suites, model_workloads

# D (head dim) is intentionally absent: it appears in no tensor shape, and
# dispatch rebuilds the dim dict from tensor shapes alone. The reference
# recovers it as E // H.
_DIMS = ("B", "S", "N", "H", "C", "E", "F")

def make_af3_single_repr_block_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    batch, n_seq, n_res = dims["B"], dims["S"], dims["N"]
    heads, channels = dims["H"], dims["C"]
    fan_out, inner = dims["E"], dims["F"]

    if fan_out % heads:
        raise ValueError(f"E={fan_out} must be a whole multiple of H={heads}")
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(((batch * 131 + n_seq) * 131 + n_res) * 131 + channels)

    def weight(fan_in, fan_out_):
        return (
            torch.randn((fan_in, fan_out_), device=device, dtype=torch.float32)
            * fan_in**-0.5
        ).to(dtype)

    def gain(width):
        return (
            1.0 + 0.1 * torch.randn((width,), device=device, dtype=torch.float32)
        ).to(dtype)

    # Genuine Bernoulli keep/drop mask converted to an additive form, matching
    # the standalone evoattention declaration. Key 0 is always kept: an
    # all-dropped query row would make the softmax produce NaN, which is a
    # property of the mask, not a kernel defect worth testing for.
    keep = torch.rand((batch, n_seq, 1, 1, n_res), device=device) > 0.5
    keep[..., 0] = True
    res_mask = torch.where(
        keep,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), -1e9, device=device, dtype=torch.float32),
    )

    return {
        "x": torch.randn((batch, n_seq, n_res, channels), device=device, dtype=dtype)
        * 0.5,
        "ln1_weight": gain(channels),
        "ln1_bias": (
            0.1 * torch.randn((channels,), device=device, dtype=torch.float32)
        ).to(dtype),
        "q_weight": weight(channels, fan_out),
        "k_weight": weight(channels, fan_out),
        "v_weight": weight(channels, fan_out),
        "res_mask": res_mask,
        # float32 regardless of the case dtype, as MegaFold keeps it.
        "pair_bias": torch.randn(
            (batch, 1, heads, n_res, n_res), device=device, dtype=torch.float32
        )
        * 0.5,
        "out_weight": weight(fan_out, channels),
        "ln2_weight": gain(channels),
        "ln2_bias": (
            0.1 * torch.randn((channels,), device=device, dtype=torch.float32)
        ).to(dtype),
        "gate_weight": weight(channels, inner),
        "up_weight": weight(channels, inner),
        "down_weight": weight(inner, channels),
        "eps": 1e-5,
        # Scaled by 1/sqrt(elements), for the same reason the Llama block does
        # it: absolute gradient magnitudes grow with the reduced axes, so an
        # unscaled upstream gradient would need a per-workload tolerance.
        "dout": (
            torch.randn((batch, n_seq, n_res, channels), device=device, dtype=dtype)
            * 0.5
            * (batch * n_seq * n_res) ** -0.5
        ),
    }


_BENCHMARK = model_workloads(
    ALPHAFOLD3,
    "single_repr_block",
    tuple(
        {"batch": 1, "n_seq": 1, "residues": residues}
        for residues in AF3_RESIDUE_SWEEP
    ),
    ("bfloat16",),
    scaled=True,
    note="MegaFold trains at batch 1 with activation checkpointing; n_seq 1 is the pair-bias attention setting",
)

_SMALL = dict(B=1, S=2, N=16, H=4, C=32, E=64, F=128)
_SMALL_2 = dict(B=1, S=1, N=23, H=2, C=16, E=32, F=64)
_CORRECTNESS = (
    Workload(dims=_SMALL, dtype="float32", atol=1e-4, rtol=1e-4),
    Workload(dims=_SMALL_2, dtype="float32", atol=1e-4, rtol=1e-4),
    Workload(dims=_SMALL, dtype="bfloat16"),
    Workload(dims=_SMALL_2, dtype="bfloat16"),
)

op = declare_op(
    name="af3_single_repr_block",
    forward=(
        "evograd.ops.level3.af3_single_repr_block.forward_ref:"
        "af3_single_repr_block_forward_ref"
    ),
    runtime_forward=(
        "evograd.ops.level3.af3_single_repr_block.forward_ref:"
        "af3_single_repr_block_runtime_ref"
    ),
    level=3,
    family="protein_block",
    reference_dtype="float32",
    dims=_DIMS,
    args=(
        Active("x", "[B, S, N, C]", note="single representation"),
        Active("ln1_weight", "[C]"),
        Active("ln1_bias", "[C]"),
        Active("q_weight", "[C, E]"),
        Active("k_weight", "[C, E]"),
        Active("v_weight", "[C, E]"),
        Inactive(
            "res_mask",
            "[B, S, 1, 1, N]",
            dtype="float32",
            note="additive per-key mask: 0 = keep, large-negative = drop",
        ),
        Active(
            "pair_bias",
            "[B, 1, H, N, N]",
            dtype="float32",
            grad="d_pair_bias",
            note="trainable pair bias; broadcast over the MSA (S) axis",
        ),
        Active("out_weight", "[E, C]"),
        Active("ln2_weight", "[C]"),
        Active("ln2_bias", "[C]"),
        Active("gate_weight", "[C, F]"),
        Active("up_weight", "[C, F]"),
        Active("down_weight", "[F, C]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("out", "[B, S, N, C]"),
    parameter_args=(
        "ln1_weight", "ln1_bias", "q_weight", "k_weight", "v_weight",
        "out_weight", "ln2_weight", "ln2_bias", "gate_weight", "up_weight",
        "down_weight",
    ),
    forward_semantics=(
        "One AlphaFold3 single-representation block over x of shape "
        "[batch, n_seq, n_res, channels]: LayerNorm -> Q/K/V projections to "
        "[.., heads, head_dim] -> attention scoring q @ k^T scaled by "
        "head_dim**-0.5, plus pair_bias and the additive res_mask, softmaxed "
        "over the key-residue axis in float32 -> weighted sum of v -> output "
        "projection -> residual add -> LayerNorm -> SwiGLU transition "
        "(silu(x @ gate) * (x @ up), then @ down) -> residual add. "
        "head_dim is E // H. pair_bias is shared across the MSA axis. "
        "Compute both LayerNorm statistics, the softmax, and the SiLU in "
        "float32. Do not call sdpa, flash_attn, F.layer_norm, or any high-level "
        "attention or transformer API in the generated math. "
        "This models the single-representation update only: the "
        "triangle-multiplicative pair update that a full pairformer block would "
        "also perform is out of scope."
    ),
    backward_semantics=(
        "Return gradients for the thirteen Active inputs in declaration order: "
        "dx, dln1_weight, dln1_bias, dq_weight, dk_weight, dv_weight, "
        "d_pair_bias, dout_weight, dln2_weight, dln2_bias, dgate_weight, "
        "dup_weight, ddown_weight. res_mask and eps are inactive and receive no "
        "gradient; if the extracted graph produces a d_res_mask, discard it. "
        "d_pair_bias has pair_bias's float32 dtype and shape "
        "[B, 1, H, N, N], which means it must reduce over the MSA (S) axis. "
        "Both residual connections feed dx. Accumulate every reduction in "
        "float32."
    ),
    extra_constraints=(
        "Tensor layout notes:\n"
        "- x, out, dout: [batch, n_seq, n_res, channels], contiguous CUDA\n"
        "- projections are [C, E] and [E, C]; the transition is [C, F] and [F, C]\n"
        "- E must be a whole multiple of H; F is 4*C\n"
        "- pair_bias and res_mask stay float32 even when x is bfloat16\n"
        "- attention contracts over the residue axis; the MSA axis is a batch "
        "axis for everything except d_pair_bias, which reduces over it\n"
        "- score memory is O(S*H*N^2), which is what makes the saved-state "
        "choice matter here: recomputing the softmax in the backward trades "
        "that for arithmetic."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites=fixed_shape_suites(_BENCHMARK),
    tolerances={
        "float32": (1e-4, 1e-4),
        "bfloat16": (5e-2, 2e-2),
    },
    tolerance_multipliers={
        # Reduces over the whole MSA axis, so its absolute error grows with S
        # while the other gradients' do not. Same treatment the standalone
        # evoattention declaration gives it.
        "d_pair_bias": (2.0, 1.0),
    },
    # res_mask is routing metadata, not activation state.
    memory_inputs=(
        "x",
        "ln1_weight",
        "ln1_bias",
        "q_weight",
        "k_weight",
        "v_weight",
        "pair_bias",
        "out_weight",
        "ln2_weight",
        "ln2_bias",
        "gate_weight",
        "up_weight",
        "down_weight",
    ),
    make_inputs=make_af3_single_repr_block_inputs,
)
