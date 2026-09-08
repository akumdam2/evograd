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
step from position ids and shares them across all 32 layers, so within this
boundary they are tables, not activations, and they receive no gradient.

**A note on ``QO``.** For Llama-3, ``n_heads * head_dim == hidden``, so ``QO``
and ``H`` are both 4096 and ``q_proj`` is square. They remain separate declared
dims because they are separate concepts -- Qwen3's ``QO`` is 2048 against a
1024 hidden -- and a kernel that fuses them would be correct here and wrong
there.

**Provenance.** Every shape below is derived from ``LLAMA_3_8B`` in
:mod:`evograd.opdecl.models`, which is Meta-Llama-3-8B's published
configuration. That is weaker than Qwen3's chain, which reads its dims out of a
harvest, and it is weaker on purpose rather than by oversight: the shapes here
are what the architecture *will* run, derived from the config, and they do not
become claims about an observation just because a harvest later appears next to
them. :data:`HARVESTED` is what says whether one has, and it gates the observed
suite alone. See :data:`CALIBRATED` for what is still owed on tolerances.
"""

from evograd.benchmark.topdown import has_snapshot as _has_snapshot
from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.opdecl.models import LLAMA_3_8B
from evograd.opdecl.tolerance import ReductionScaledAtol

#: The workload these dims describe; also the ``Provenance`` model key.
_WORKLOAD = "llama_3_8b"

#: Whether a harvest for this workload has been run and tracked. Everything
#: below that says "observed" is a *derivation from the published config* until
#: this is true; the observed suite is attached only when it is.
HARVESTED = _has_snapshot(_WORKLOAD)

#: Has the tolerance been measured at the shape the *model* runs?
#:
#: **Yes**, on a GH200, at ``[2, 2048, 4096]`` with the full 4096-token
#: contraction. Measured with synthetic inputs at the model's widths, because
#: the harvested capture cannot answer this question: its gradients arrive at
#: ``ref_absmax`` 0.0 with errors near 1e-07, so it exercises the forward
#: tolerances and nothing else. Required base ``t`` there, against the 2e-2 the
#: declaration carries:
#:
#:     result       required_t   declared atol   supplied / required
#:     q             1.039e-02      3.657e-02          3.5x
#:     k             1.004e-02      3.933e-02          3.9x
#:     v             0                3.933e-02        --
#:     dx            2.415e-02      4.388e-02          1.5x
#:     dq_weight     1.795e-01      1.687e+00          ~1.9x
#:     dk_weight     1.437e-01      2.182e+00          ~2.4x
#:     dv_weight     0                4.546e-01        --
#:
#: The two weight-gradient ratios are approximate because ``required_t`` couples
#: atol and rtol (it is the smallest ``t`` accepting at ``atol=ma*t, rtol=t``)
#: while the declaration holds rtol at the base; read
#: ``minimal_atol_multiplier["2e-02"]`` in the artifact for the decoupled
#: number. Both are above the 1.5x the multipliers were declared with -- the
#: hook's ``gain=2.0`` is deliberately conservative past the anchor -- and
#: `levels.level2.negative_controls` says that headroom costs nothing
#: measurable: every result's scaled-fault floor is 2.0%, matching
#: ``qwen3_attention`` and ``qwen3_swiglu_mlp``, because ``rtol`` is what
#: catches a scaled fault and the hook never touches it.
#:
#: ``v`` and ``dv_weight`` measure exactly 0.0 at every shape -- the value path
#: has no rotation, so the two *spellings* are the same computation there. That
#: is a fact about the oracle pair, not about a candidate: a generated kernel
#: accumulating ``dv_weight`` over 4096 tokens in bfloat16 has the same rounding
#: growth as the other two, which is why they keep the hook's term rather than
#: being pinned to a measurement of zero.
#:
#: The *correctness grid* is a different question, and it has been measured --
#: see :data:`GRID_CALIBRATED`. The grid is declared in this file and both
#: spellings are in ``forward_ref``, so nothing about a GPU or a harvest is
#: needed to compare them.
#:
#: Reproduced by::
#:
#:     python -m evograd.benchmark.topdown.llama3_8b.levels.level2.qkv_rope \
#:         calibrate --source results/llama3-level4/layer16.pt --device cuda
CALIBRATED = True

#: Have the correctness-grid tolerances been measured rather than inherited?
#:
#: **Yes.** ``levels.level2.qkv_rope calibrate --skip-canonical`` compared the
#: declared float32 reference against ``runtime_forward`` on every workload in
#: :data:`_CORRECTNESS`, which is the smallest disagreement any correct
#: implementation can have with the oracle. The numbers are in the
#: ``tolerance_multipliers`` comment below.
#:
#: Measured on **CPU**. The two spellings differ in bfloat16 rounding, and a GPU
#: accumulates ``F.linear`` differently (tensor cores, split-k), so the values
#: can shift. That is why each declared multiplier carries a 1.5x margin over
#: its measurement rather than being the measurement.
GRID_CALIBRATED = True

#: Once per decoder layer. Read off the published layer count rather than a
#: harvest, so this is what the canonical step *will* invoke, not what one did.
FREQUENCY = LLAMA_3_8B.layers

PROVENANCE_CHAIN = (
    f"published configuration {LLAMA_3_8B.source}",
    "LlamaAttention.forward, projection and rotary prefix",
    "llama3_qkv_rope",
)

#: The canonical run's free dimensions; the same batch and sequence Qwen3 uses,
#: so the two workloads' observed shapes can be read side by side.
_BATCH, _SEQ = 2, 2048

_DIMS = ("B", "T", "H", "HQ", "HK", "D", "QO", "KVO")

_CONFIG_DIMS = LLAMA_3_8B.qkv_norm_rope_dims(batch=_BATCH, seq=_SEQ)

_CONFIGURED = Provenance(
    model=_WORKLOAD,
    component="qkv_norm_rope",
    free={"batch": _BATCH, "seq": _SEQ},
    source="hf_config",
    # The model reaches q/k/v by view-then-transpose, so what a kernel receives
    # and must return is the non-contiguous head-major view. Qwen3 reads this
    # off recorded strides; here it is a property of the spelling above, which
    # `forward_ref` reproduces and `verify_runtime_forward` checks.
    layout="head_major_view",
)

_SHRUNK = Provenance(
    model=_WORKLOAD,
    component="qkv_norm_rope",
    free={},
    source="handpicked",
    scaled=True,
    note=(
        "hidden size, head count, head dimension and sequence length reduced "
        "from Llama-3-8B's 4096/32-8 heads/128-wide heads/2048 tokens so the "
        "correctness cases run on CPU; the 4:1 grouped-query ratio, the absence "
        "of projection biases and the head-major output layout are preserved, "
        "and one 2:1 case is included so a kernel cannot hard-code the ratio"
    ),
)

_BENCHMARK = (
    Workload(dims=dict(_CONFIG_DIMS), dtype="bfloat16", provenance=_CONFIGURED),
)

_CORRECTNESS = tuple(
    Workload(
        dims={
            "B": batch,
            "T": tokens,
            "H": hidden,
            "HQ": heads,
            "HK": kv_heads,
            "D": head_dim,
            "QO": heads * head_dim,
            "KVO": kv_heads * head_dim,
        },
        dtype=dtype,
        provenance=_SHRUNK,
    )
    for batch, tokens, hidden, heads, kv_heads, head_dim, dtype in (
        # Llama's own 4:1 grouping, and QO == H as in the real model.
        (1, 16, 32, 4, 1, 8, "float32"),
        (2, 32, 64, 4, 1, 16, "float32"),
        (2, 32, 64, 4, 1, 16, "bfloat16"),
        # 2:1 grouping and QO != H, so a kernel that folded the query fan-out
        # into the hidden size -- which Llama-3 alone would let it get away
        # with -- fails here.
        (1, 24, 32, 4, 2, 16, "bfloat16"),
    )
)


#: The bf16 correctness case the declared multipliers were measured on, and the
#: largest case in the grid: every other case is at or below it on every term,
#: so the hook below is *exactly* the identity on the whole grid at both dtypes
#: (worst raw factor 1.000) and cannot loosen a case that passes today.
_ANCHOR = {"B": 2, "T": 32, "H": 64, "HQ": 4, "HK": 1, "D": 16, "QO": 64, "KVO": 16}

#: What the grid cannot see. The three projection gradients accumulate one term
#: per token, and the model runs 4096 of them where the grid's longest is 64 --
#: the random walk alone is 8x, before the element-count term. ``q``, ``k``,
#: ``v`` and ``dx`` keep the token axis and so have no reduction; they pick up
#: the extreme-value term only, which is a property of how many elements
#: ``allclose`` maximizes over rather than of the arithmetic.
#:
#: The same structure as ``qwen3_qkv_norm_rope``'s at the same boundary, minus
#: the two per-head norm weights Llama does not have. It supplies 1.83x to 1.97x
#: on the outputs and ``dx``, and 21.6x to 22.7x on the three projection
#: gradients, at the observed shape. Both ends are now checked: it is exactly
#: the identity on the grid, and :data:`CALIBRATED` records the measurement that
#: says the growth it supplies covers the growth that was needed.
_REDUCTION_SCALED = ReductionScaledAtol(
    anchor_dims=_ANCHOR,
    reduction_dims={
        "dq_weight": ("B", "T"),
        "dk_weight": ("B", "T"),
        "dv_weight": ("B", "T"),
    },
    result_dims={
        "q": ("B", "HQ", "T", "D"),
        "k": ("B", "HK", "T", "D"),
        "v": ("B", "HK", "T", "D"),
        "dx": ("B", "T", "H"),
        "dq_weight": ("QO", "H"),
        "dk_weight": ("KVO", "H"),
        "dv_weight": ("KVO", "H"),
    },
    gain=2.0,
)


def make_llama3_qkv_rope_inputs(torch, op, workload, device="cuda"):
    """Inputs at the magnitudes a real step has, and real rotary tables.

    ``cos``/``sin`` are built from an inverse-frequency schedule rather than
    drawn at random: a random pair is not a rotation, and the reference's
    accuracy -- and therefore any tolerance calibrated against it -- would then
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
        LLAMA_3_8B.rope_theta
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
        "evograd.ops.level2.llama3_qkv_rope.forward_ref:llama3_qkv_rope_forward_ref"
    ),
    # Timed through what the model runs: RoPE applied in the model dtype. The
    # declared forward rotates in float32, which is more accurate and
    # correspondingly slower.
    runtime_forward=(
        "evograd.ops.level2.llama3_qkv_rope.forward_ref:"
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
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    # The observed suite is what a *harvest* recorded, so it appears only once
    # one has been run and tracked. Until then this declaration still benches --
    # on the configured shape above -- but nothing here claims to have been
    # observed. `HARVESTED` is the flag to read rather than probing the dict.
    benchmark_suites=(
        {"llama_3_8b_observed": _BENCHMARK} if HARVESTED else {}
    ),
    memory_inputs=("x", "q_weight", "k_weight", "v_weight", "cos", "sin"),
    # Measured, not chosen -- on the correctness grid.
    # `levels.level2.qkv_rope calibrate --skip-canonical` compares the declared
    # float32 reference against `runtime_forward`, the spelling the model runs
    # and therefore the smallest disagreement any correct implementation can
    # have with the oracle, and reports the smallest base `t` for which
    # `allclose(atol=ma*t, rtol=t)` accepts each result.
    #
    # float32: 0.0 on every result of every case. The two spellings differ only
    # in whether the rotation is carried in float32, and at float32 input they
    # are the same computation. 2e-5 is the repository's ordinary float32 base.
    #
    # bfloat16, worst over both grid cases:
    #   q 5.18e-03   k 5.18e-03   v 0
    #   dx 1.52e-02  dq_weight 5.21e-02  dk_weight 6.32e-02  dv_weight 0
    #
    # `v` and `dv_weight` measure exactly 0.0 at both dtypes: the value path has
    # no rotation, so the two spellings are the same computation there.
    #
    # The base is set by the forward outputs -- `q` at 5.18e-03 -- so the
    # outputs are gated by the base alone and only the reductions carry
    # multipliers. 2e-2 leaves 3.9x on `q`. Going lower would mean putting a
    # multiplier on a forward output, which hides the number a candidate is
    # primarily judged on.
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (2e-2, 2e-2)},
    # Each multiplier is the measured minimum at base 2e-2 times a 1.5 safety
    # margin, rounded up to one decimal. The weight gradients need the largest
    # because they reduce over all B*T tokens with cancelling signs; `q`, `k`,
    # `v` and `dv_weight` measured below 1.0 and so have none.
    #
    #   result        required_t   min ma at t=2e-2   declared
    #   dx             1.52e-02          0.76           1.2
    #   dq_weight      5.21e-02          2.60           3.9
    #   dk_weight      6.32e-02          3.16           4.8
    #
    # `dx` measures below 1.0 and is still declared: a modest GPU-vs-CPU shift
    # in accumulation order would otherwise put it over the base, and the cost
    # of stating it is 1.2x on one gradient.
    tolerance_multipliers={
        "dx": (1.2, 1.0),
        "dq_weight": (3.9, 1.0),
        "dk_weight": (4.8, 1.0),
    },
    # Those multipliers were measured on the correctness grid, whose longest
    # token reduction is 64 terms. The model runs 4096, and a constant cannot
    # describe a quantity that grows with the sum. This operator ran without the
    # hook for exactly one release, and `levels.level2.negative_controls`
    # reported what the file predicted it would: `clean accepted: False` at the
    # observed shape -- the grid's constants rejecting a *correct* kernel at
    # production width. The hook supplies the missing term and is the identity
    # at and below the anchor, so every correctness case keeps the tolerance it
    # has. The term it supplies covers the term that was needed, with the
    # margins `CALIBRATED` records.
    tolerance_hook=_REDUCTION_SCALED,
    make_inputs=make_llama3_qkv_rope_inputs,
)
