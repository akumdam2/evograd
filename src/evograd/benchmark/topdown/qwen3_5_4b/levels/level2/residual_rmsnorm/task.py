"""Operator declaration: residual add + zero-centred RMSNorm, two outputs
(site ``residual_rmsnorm``).

Between the token mixer and the MLP of both block kinds: the post-attention
``residual + branch`` and ``post_attention_layernorm``. Returns the normalized
value and the sum itself, so the backward combines two upstream gradients -- the
characteristic error is to ignore ``dsummed``. Block semantics: the layer's input
norm and its closing residual add are not part of this site (the cross-layer
fusion is a later step's).
"""

from evograd.opdecl import Active, Inactive, declare_op

from ..task_tools import OBSERVED_SUITE, observed, seed_of


_DIMS = ("B", "T", "H")

_BENCHMARK = observed("residual_rmsnorm")


def make_residual_rmsnorm_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(seed_of(dims, 16))
    shape = (dims["B"], dims["T"], dims["H"])
    return {
        "branch": torch.randn(shape, device=device, dtype=dtype),
        "residual": torch.randn(shape, device=device, dtype=dtype),
        # Zero-centred, as the model stores it.
        "weight": (0.1 * torch.randn((dims["H"],), device=device)).to(dtype),
        "eps": 1e-6,
        # Two independent upstream gradients: a backward that ignored dsummed,
        # or assumed the two equal, must not pass.
        "dnormed": torch.randn(shape, device=device, dtype=dtype),
        "dsummed": torch.randn(shape, device=device, dtype=dtype),
    }


op = declare_op(
    name="qwen3_5_residual_rmsnorm",
    level=2,
    family="norm",
    forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.residual_rmsnorm.reference:"
            "residual_rmsnorm_forward_ref",
    runtime_forward="evograd.benchmark.topdown.qwen3_5_4b.levels.level2.residual_rmsnorm."
                    "reference:residual_rmsnorm_forward_production",
    dims=_DIMS,
    args=(
        Active("branch", "[B, T, H]"),
        Active("residual", "[B, T, H]"),
        Active("weight", "[H]"),
        Inactive("eps", default=1e-6),
    ),
    output=(
        Active("normed", "[B, T, H]"),
        Active("summed", "[B, T, H]"),
    ),
    parameter_args=("weight",),
    forward_semantics=(
        "s = residual + branch; with the reduction in float32, "
        "normed = s * rsqrt(mean(s^2, -1) + eps) * (1 + weight), computed in "
        "float32 and cast to the input dtype once at the end (zero-centred "
        "weight: zero is the identity scale). Return (normed, summed) IN THAT "
        "ORDER, both [B, T, H] in the input dtype, summed = s itself. Do not "
        "call F.rms_norm or autograd in the generated math."
    ),
    backward_semantics=(
        "The backward receives (dnormed, dsummed) and returns dbranch, dresidual, "
        "dweight IN THIS ORDER. With r = rsqrt(mean(s^2) + eps), xhat = s * r and "
        "g = dnormed * (1 + weight): ds = dsummed + r * (g - xhat * mean(g * "
        "xhat, -1)); dbranch = dresidual = ds; dweight = sum over B and T of "
        "dnormed * xhat. Accumulate the reductions in float32."
    ),
    coverage=_BENCHMARK,
    benchmark=_BENCHMARK,
    benchmark_suites={OBSERVED_SUITE: _BENCHMARK},
    make_inputs=make_residual_rmsnorm_inputs,
)
