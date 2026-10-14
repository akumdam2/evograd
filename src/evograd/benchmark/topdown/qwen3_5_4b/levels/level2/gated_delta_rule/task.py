"""Operator declaration: the chunked gated delta rule (site ``gated_delta_rule``).

The Gated DeltaNet's sequence mixer: q and k L2-normalized inside the kernel, q
scaled by ``Dk^-0.5``, a ``[Dk, Dv]`` state per head that decays by ``exp(g)``
and is corrected toward ``v`` with strength ``beta`` at every token. No initial
and no final state -- a training step from the start of the sequence. The
reference defines it token by token; production computes it in chunks of 64.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import make_pair_baseline

from .. import library
from ..task_tools import CONFIG, OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Calibrated: see the calibrated-tolerances block below.
PROVISIONAL_TOLERANCES = False

_DIMS = ("B", "T", "HV", "D")


def model_strides(dims, channels: int):
    """``v`` as the block hands it: a channel slice of the transposed convolution
    output ``[B, C, T]``, viewed as ``[B, T, HV, D]`` -- batch stride ``C * T``
    (gaps where q and k sit), token stride 1."""
    t, d = dims["T"], dims["D"]
    return (("v", (channels * t, 1, d * t, t)),)


_BENCHMARK = observed("gated_delta_rule",
                      strides=lambda dims: model_strides(dims, CONFIG.conv_dim))

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, HV=hv, D=d), dtype=dtype,
             provenance=shrunk("gated_delta_rule"))
    for b, t, hv, d, dtype in (
        (1, 16, 2, 8, "float32"),
        # Two chunks, the second padded: the chunk boundary and the padding.
        (2, 70, 4, 16, "float32"),
        (1, 70, 4, 16, "bfloat16"),
        (2, 33, 2, 16, "bfloat16"),
    )
) + (
    # v as the model lays it out (conv channels = 2 * HV * D here).
    Workload(dims=dict(B=2, T=70, HV=4, D=16), dtype="float32",
             provenance=shrunk("gated_delta_rule",
                               model_strides(dict(B=2, T=70, HV=4, D=16), 2 * 4 * 16))),
)


def make_gated_delta_rule_inputs(torch, op, workload, device="cuda"):
    """q, k, v and do at unit scale; g and beta as the model's gating produces them.

    ``g = -A * softplus(a + 1)`` with ``A`` drawn per head from U(0.01, 16) --
    the model's A_log initialization -- so the per-step decay spans almost none
    to complete forgetting. Not rescaled to the model's activation magnitudes:
    that belongs to a calibration step, from Qwen3.5's own measurements.
    """
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 12))
    shape = (dims["B"], dims["T"], dims["HV"], dims["D"])
    gates = (dims["B"], dims["T"], dims["HV"])
    A = torch.empty((dims["HV"],), device=device).uniform_(0.01, 16)
    g = -A * torch.nn.functional.softplus(torch.randn(gates, device=device) + 1.0)
    return {
        "q": torch.randn(shape, device=device).to(dtype),
        "k": torch.randn(shape, device=device).to(dtype),
        "v": torch.randn(shape, device=device).to(dtype),
        "g": g.float(),
        "beta": torch.sigmoid(torch.randn(gates, device=device)).to(dtype),
        "do": torch.randn(shape, device=device).to(dtype),
    }


# --- calibrated tolerances (scripts/qwen3_5_4b/calibration.py apply) ---
from evograd.opdecl.tolerance import ReductionScaledAtol  # noqa: E402

#: Calibrated for Qwen3.5-4B (run ``20261011_calib``; GH200, grid also on CPU):
#: ``op.forward`` against ``op.runtime_forward`` only -- never a library or
#: a candidate -- on captured, correctness, correctness[cpu], observed;
#: four repeats each (production was deterministic). The bfloat16 base
#: atol is the smallest 1-2-5 step at or above both 1.5x the forward
#: outputs' grid need and one bfloat16 ulp of their magnitude; rtol 2e-2.
#: Multipliers are per dtype (in the hook): 1.5x that dtype's grid need,
#: rounded up (none below 1). Each gain is the smallest that carries the
#: bfloat16 multiplier over the real-size cases with the same 1.5x.
#: ``need`` is the atol required at the base rtol, in units of the base atol.
#:
#:   result           fp32 grid bf16 grid real need fp32x bf16x  gain margin@real
#:   dbeta              0.0148     1.18      0.73     1   1.8     0     2.47
#:   dg                  0.022   0.0533     0.382     1     1     0     2.62
#:   dk                 0.0374    0.411    0.0484     1     1     0     20.7
#:   dq                 0.0405      0.9    0.0512     1   1.4     0     27.3
#:   dv                 0.0148    0.177    0.0773     1     1     0     12.9
#:   o                  0.0487    0.259     0.107     1     1     0     9.33
CALIBRATED = True
CALIBRATED_TOLERANCES = {"float32": (2e-05, 2e-05), "bfloat16": (0.002, 0.02)}
#: Per-dtype multipliers live in the hook; none apply to every dtype.
CALIBRATED_MULTIPLIERS = {}
CALIBRATED_HOOK = ReductionScaledAtol(
    anchor_dims={"B": 1, "D": 16, "HV": 4, "T": 70},
    reduction_dims={},
    result_dims={"dbeta": ('B', 'T', 'HV'), "dg": ('B', 'T', 'HV'), "dk": ('B', 'T', 'HV', 'D'), "dq": ('B', 'T', 'HV', 'D'), "dv": ('B', 'T', 'HV', 'D'), "o": ('B', 'T', 'HV', 'D')},
    gain=0.0,
    result_gain={},
    dtype_multipliers={"bfloat16": {"dbeta": 1.8, "dq": 1.4}},
)
# --- end calibrated tolerances ---


op = declare_op(
    name="qwen3_5_gated_delta_rule",
    level=2,
    family="linear_attention",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_delta_rule.reference:"
            "gated_delta_rule_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.gated_delta_rule."
                    "reference:gated_delta_rule_forward_production",
    dims=_DIMS,
    args=(
        Active("q", "[B, T, HV, D]"),
        Active("k", "[B, T, HV, D]"),
        Active("v", "[B, T, HV, D]"),
        Active("g", "[B, T, HV]", dtype="float32"),
        Active("beta", "[B, T, HV]"),
    ),
    output=Active("o", "[B, T, HV, D]"),
    parameter_args=(),
    forward_semantics=(
        "Per batch and head, in float32: qn = l2norm(q) * D^-0.5 and kn = "
        "l2norm(k) with l2norm(x) = x * rsqrt(sum(x^2, -1) + 1e-6). Starting from "
        "S = 0 ([D, D]): for t = 0..T-1, S = exp(g_t) * S; "
        "S = S + kn_t outer (beta_t * (v_t - S^T kn_t)); o_t = S^T qn_t. Return o "
        "[B, T, HV, D] in q's dtype, contiguous. No initial or final state. A "
        "chunked (e.g. 64-token) WY/UT formulation computing the same recurrence "
        "is the expected implementation. Do not call autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives do and returns dq, dk, dv, dg, dbeta IN THIS ORDER: "
        "the exact gradient of the recurrence above, including through both L2 "
        "normalizations and the q scale. dg is float32 (g's dtype); the others in "
        "their inputs' dtypes. Accumulate states and reductions in float32."
    ),
    extra_constraints=(
        "Dk == Dv == D. g is float32 and negative (a log decay); exp(g) may "
        "underflow to zero. T need not be a multiple of the chunk size. v may be a "
        "strided view (a channel slice of the convolution output)."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=CALIBRATED_TOLERANCES,
    tolerance_multipliers=CALIBRATED_MULTIPLIERS,
    tolerance_hook=CALIBRATED_HOOK,
    performance_baselines={"fla": make_pair_baseline(
        library.fla_gated_delta_rule, ("q", "k", "v", "g", "beta"), sidecar="fla")},
    make_inputs=make_gated_delta_rule_inputs,
)
