"""Operator declaration: residual add + RMSNorm, two outputs (site ``residual_rmsnorm``).

Between the attention and the MoE block of every layer: the post-attention
``residual + branch`` and ``post_attention_layernorm``. Returns the normalized
value and the sum itself, so the backward combines two upstream gradients -- the
characteristic error is to ignore ``dsummed``. The layer's input norm and its
closing residual add are not part of this site (the cross-layer fusion is a
later step's). OLMoE's norm weight is a plain scale initialized at one, not the
zero-centred ``(1 + w)`` of Qwen3.5.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op

from ..task_tools import OBSERVED_SUITE, PROVISIONAL, observed, seed_of, shrunk

#: Not calibrated: see ``task_tools.PROVISIONAL``.
PROVISIONAL_TOLERANCES = True

_DIMS = ("B", "T", "H")

_BENCHMARK = observed("residual_rmsnorm")

_CORRECTNESS = tuple(
    Workload(dims=dict(B=b, T=t, H=h), dtype=dtype, provenance=shrunk("residual_rmsnorm"))
    for b, t, h, dtype in (
        (1, 8, 64, "float32"),
        (2, 17, 128, "float32"),
        (2, 32, 256, "bfloat16"),
        (1, 64, 640, "bfloat16"),
    )
)


def make_residual_rmsnorm_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 24))
    shape = (dims["B"], dims["T"], dims["H"])
    return {
        "branch": torch.randn(shape, device=device, dtype=dtype),
        "residual": torch.randn(shape, device=device, dtype=dtype),
        # Centred on one, as the model initializes and holds it.
        "weight": (1.0 + 0.1 * torch.randn((dims["H"],), device=device)).to(dtype),
        "eps": 1e-5,
        # Two independent upstream gradients: a backward that ignored dsummed,
        # or assumed the two equal, must not pass.
        "dnormed": torch.randn(shape, device=device, dtype=dtype),
        "dsummed": torch.randn(shape, device=device, dtype=dtype),
    }


op = declare_op(
    name="olmoe_residual_rmsnorm",
    level=2,
    family="norm",
    forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.residual_rmsnorm.reference:"
            "residual_rmsnorm_forward_ref",
    runtime_forward="evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.residual_rmsnorm."
                    "reference:residual_rmsnorm_forward_production",
    dims=_DIMS,
    args=(
        Active("branch", "[B, T, H]"),
        Active("residual", "[B, T, H]"),
        Active("weight", "[H]"),
        Inactive("eps", default=1e-5),
    ),
    output=(
        Active("normed", "[B, T, H]"),
        Active("summed", "[B, T, H]"),
    ),
    parameter_args=("weight",),
    forward_semantics=(
        "s = residual + branch; with the reduction in float32, "
        "normed = s * rsqrt(mean(s^2, -1) + eps) * weight, computed in float32 and "
        "cast to the input dtype once at the end. Return (normed, summed) IN THAT "
        "ORDER, both [B, T, H] in the input dtype, summed = s itself. Do not call "
        "F.rms_norm or autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dnormed, dsummed) and returns dbranch, dresidual, "
        "dweight IN THIS ORDER. With r = rsqrt(mean(s^2) + eps), xhat = s * r and "
        "g = dnormed * weight: ds = dsummed + r * (g - xhat * mean(g * xhat, -1)); "
        "dbranch = dresidual = ds; dweight = sum over B and T of dnormed * xhat. "
        "Accumulate the reductions in float32."
    ),
    correctness=_CORRECTNESS,
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    tolerances=PROVISIONAL,
    make_inputs=make_residual_rmsnorm_inputs,
)
