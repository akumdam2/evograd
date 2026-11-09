"""Operator declaration: Llama-3's q/k/v projection and RoPE.

The Llama-3 counterpart of ``qwen3_qkv_norm_rope``, and a separate declaration
for one reason: **Llama-3 has no per-head query/key RMSNorm.** Qwen3 normalizes
each head over the head dimension between the reshape and the transpose, with
two learned ``[D]`` weights; Llama does not. That is not a flag a shared kernel
could carry -- it changes the forward arithmetic, adds two gradients, and moves
the dominant reduction -- so the two boundaries are two operators.

**Boundary, stated once and unambiguously.**

    normalized hidden_states
      -> separate q_proj, k_proj, v_proj
      -> reshape by heads
      -> transpose to head-major
      -> apply RoPE to q and k
      -> return (q, k, v)

``F.scaled_dot_product_attention`` and ``o_proj`` are **not** part of it -- those
are the attention boundary, and the two tasks meet exactly where this one's
outputs become that one's inputs. The residual RMSNorm that produces ``x`` is
also outside it.

``cos`` and ``sin`` are Inactive: ``LlamaRotaryEmbedding`` computes them once per
step from position ids and shares them across all 16 layers, so within this
boundary they are tables, not activations, and they receive no gradient.

**A note on ``QO``.** For Llama-3, ``n_heads * head_dim == hidden``, so ``QO``
and ``H`` are both 2048 and ``q_proj`` is square. They remain separate declared
dims because they are separate concepts -- Qwen3's ``QO`` is 2048 against a
1024 hidden -- and a kernel that fuses them would be correct here and wrong
there.

**Provenance.** Every shape below is derived from ``LLAMA_3_2_1B`` in
:mod:`evograd.opdecl.models`, which is this workload's published
configuration. That is weaker than Qwen3's chain, which reads its dims out of a
harvest, and it is weaker on purpose rather than by oversight: the shapes here
are what the architecture *will* run, derived from the config, and they do not
become claims about an observation just because a harvest later appears next to
them. :data:`HARVESTED` is what says whether one has, and it gates the observed
suite alone.
"""

from evograd.benchmark.topdown import has_snapshot as _has_snapshot
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
# The workload pins Llama-3.2-1B. `opdecl.models` also carries LLAMA_3_8B,
# a different model.
from evograd.opdecl.models import LLAMA_3_2_1B

#: The workload these dims describe; also the ``Provenance`` model key.
_WORKLOAD = "llama_3_2_1b"

#: Whether a harvest for this workload has been run and tracked. Everything
#: below that says "observed" is a *derivation from the published config* until
#: this is true; the observed suite is attached only when it is.
HARVESTED = _has_snapshot(_WORKLOAD)


#: Once per decoder layer. Read off the published layer count rather than a
#: harvest, so this is what the canonical step *will* invoke, not what one did.
FREQUENCY = LLAMA_3_2_1B.layers

PROVENANCE_CHAIN = (
    f"published configuration {LLAMA_3_2_1B.source}",
    "LlamaAttention.forward, projection and rotary prefix",
    "llama3_qkv_rope",
)

#: The canonical run's free dimensions; the same batch and sequence Qwen3 uses,
#: so the two workloads' observed shapes can be read side by side.
_BATCH, _SEQ = 2, 2048

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_CONFIG_DIMS = LLAMA_3_2_1B.qkv_norm_rope_dims(batch=_BATCH, seq=_SEQ)

_CONFIGURED = Provenance(
    model=_WORKLOAD,
    component="qkv_norm_rope",
    free={"batch": _BATCH, "seq": _SEQ},
    source="hf_config",
    # The model reaches q/k/v by view-then-transpose, so what a kernel receives
    # and must return is the non-contiguous head-major view. Qwen3 reads this
    # off recorded strides; here it is a property of the spelling above, which
    # `forward_ref` reproduces.
    layout="head_major_view",
)


_BENCHMARK = (
    Workload(dims=dict(_CONFIG_DIMS), dtype="bfloat16", provenance=_CONFIGURED),
)


def make_llama3_qkv_rope_inputs(torch, op, workload, device="cuda"):
    """Inputs at the magnitudes a real step has, and real rotary tables.

    ``cos``/``sin`` are built from an inverse-frequency schedule rather than
    drawn at random: a random pair is not a rotation, and the case would then
    describe a computation the model never performs. ``rope_theta`` is
    Llama-3's 500000, not Llama-2's 10000.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(
        dims["B"] * 1000003 + dims["T"] * 10007 + dims["H"] * 1009 + dims["D"]
    )
    hidden, head_dim = dims["H"], dims["D"]

    def projection(out_features):
        return (
            torch.randn((out_features, hidden), device=device, dtype=torch.float32)
            * hidden**-0.5
        ).to(dtype)

    position = torch.arange(dims["T"], device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (
        LLAMA_3_2_1B.rope_theta
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = position * inv_freq[None, :]
    table = torch.cat((angles, angles), dim=-1)[None, :, :]
    return {
        "x": torch.randn((dims["B"], dims["T"], hidden), device=device, dtype=dtype),
        "q_weight": projection(dims["QO"]),
        "k_weight": projection(dims["KVO"]),
        "v_weight": projection(dims["KVO"]),
        "cos": table.cos().to(dtype),
        "sin": table.sin().to(dtype),
        "dq": torch.randn(
            (dims["B"], dims["HQ"], dims["T"], head_dim), device=device, dtype=dtype
        ),
        "dk": torch.randn(
            (dims["B"], dims["HK"], dims["T"], head_dim), device=device, dtype=dtype
        ),
        "dv": torch.randn(
            (dims["B"], dims["HK"], dims["T"], head_dim), device=device, dtype=dtype
        ),
    }


op = declare_op(
    name="llama3_qkv_rope",
    level=2,
    family="attention",
    forward=(
        "evograd.benchmark.topdown.llama3_2_1b.levels.level2.qkv_rope.reference:llama3_qkv_rope_forward_ref"
    ),
    # Timed through what the model runs: RoPE applied in the model dtype. The
    # declared forward rotates in float32, which is more accurate and
    # correspondingly slower.
    runtime_forward=(
        "evograd.benchmark.topdown.llama3_2_1b.levels.level2.qkv_rope.reference:"
        "llama3_qkv_rope_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, T, H]"),
        Active("q_weight", "[QO, H]"),
        Active("k_weight", "[KVO, H]"),
        Active("v_weight", "[KVO, H]"),
        Inactive("cos", "[1, T, D]"),
        Inactive("sin", "[1, T, D]"),
    ),
    output=(
        Active("q", "[B, HQ, T, D]"),
        Active("k", "[B, HK, T, D]"),
        Active("v", "[B, HK, T, D]"),
    ),
    parameter_args=("q_weight", "k_weight", "v_weight"),
    forward_semantics=(
        "Project the normalized residual stream three ways and rotate the "
        "query and key heads. q_flat = x @ q_weight^T (shape [B, T, QO]); "
        "reshape to [B, T, HQ, D]; transpose to [B, HQ, T, D]. The same for k "
        "with k_weight at HK heads, and for v with v_weight at HK heads. There "
        "is NO per-head normalization -- that is Qwen3's boundary, not this "
        "one. Then RoPE on q and k only: with cos and sin unsqueezed at dim 1 "
        "and rotate_half(t) = cat((-t[..., D/2:], t[..., :D/2]), -1), "
        "t_out = t*cos + rotate_half(t)*sin. v is returned unrotated. Return "
        "(q, k, v) IN THAT ORDER. There are no projection biases and no "
        "attention here. Do not call F.linear, torch.matmul, @, or autograd in "
        "the generated math."
    ),
    backward_semantics=(
        "Return gradients for x, q_weight, k_weight and v_weight IN THIS "
        "ORDER. The backward receives output_grads = (dq, dk, dv), one per "
        "output, and every one of them contributes to dx. cos and sin are "
        "inactive and get no gradient. Unrotate first: for q, "
        "dq_flat = dq*cos - rotate_half(dq*sin) using the same rotate_half, "
        "because rotate_half is its own negative inverse; likewise for k. dv "
        "passes straight through, since v is not rotated. Then the three "
        "projections: dq_weight = dq_flat^T @ x summed over B and T, and "
        "dx = dq_flat @ q_weight + dk_flat @ k_weight + dv_flat @ v_weight. "
        "Accumulate every reduction and matmul in float32 before casting each "
        "gradient to its input's dtype."
    ),
    extra_constraints=(
        "This is the prefix of LlamaAttention: it ends with (q, k, v) ready for "
        "scaled_dot_product_attention, and it does not contain SDPA or o_proj. "
        "The residual RMSNorm that produces x is also outside it. There is no "
        "per-head query/key RMSNorm -- adding one would make this "
        "qwen3_qkv_norm_rope. cos and sin are shared tables computed once per "
        "step, so they are inputs and receive no gradient. The outputs are "
        "non-contiguous head-major views (q stride [B*T*HQ*D, D, HQ*D, 1]) "
        "because the model reaches them by view-then-transpose; a kernel may "
        "work in any internal layout but must return that one. HQ must be "
        "divisible by HK, and both projection fan-outs must be multiples of D. "
        "For Llama-3-8B QO happens to equal H; do not rely on that."
    ),
    grad_order=("dx", "dq_weight", "dk_weight", "dv_weight"),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    # The observed suite is what a *harvest* recorded, so it appears only once
    # one has been run and tracked. Until then this declaration still benches --
    # on the configured shape above -- but nothing here claims to have been
    # observed. `HARVESTED` is the flag to read rather than probing the dict.
    benchmark_suites=(
        {"llama_3_2_1b_observed": _BENCHMARK} if HARVESTED else {}
    ),
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    make_inputs=make_llama3_qkv_rope_inputs,
)
