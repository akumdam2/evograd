"""A sample site registry, for tests that need *a* registry rather than a model's.

The old monolithic Tier-3 patch module shipped ``LLAMA_SITES``, and much of the tier-3
suite reached for it as a convenient stand-in. That was the coupling those very
tests exist to forbid: a registry names one model's patchable places, so it
belongs to a workload package, and the patcher must not carry one.

The tests still need a registry to exercise the machinery with -- and, for the
invariant that two registries never share a namespace, two of them. So they are
built here, out of declared tasks, and owned by the tests.

``SAMPLE_SITES`` deliberately reuses the three sites the deleted built-in
carried: ``rmsnorm``, ``swiglu`` and ``fused_linear_cross_entropy`` are real
declarations with real pair baselines, which is what lets the baseline-discovery
tests find something to discover.

The three production spellings a site is patched *over* used to be imported
from the legacy ``llama3_decoder_layer`` reference. That declaration has been
deleted, so they are written here instead -- same mathematics, owned by the
tests that need them rather than by a task that no longer exists. Keeping them
numerically identical matters: several tests assert that patching a site with
its own production spelling changes nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.evaluation.tier3.patch import Site, SiteRegistry


def fused_rms_norm(x, weight, eps):
    """PyTorch's fused RMSNorm -- one kernel instead of pow/mean/rsqrt/mul."""
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def float32_swiglu(gate, up):
    """SiLU in float32: the activation is where a bfloat16 MLP loses the most."""
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def eager_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                        target: torch.Tensor):
    """Logits materialized, then cross-entropy -- what an unfused head costs.

    At a 128256 vocabulary this tensor is the memory story: 2.1 GB at 8192
    tokens in bfloat16, and the same again for its gradient. A fused loss never
    materializes it, which is why it moves peak memory rather than latency, and
    why peak memory is a reported metric at this tier.
    """
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float().flatten(0, -2), target.flatten())


#: Three sites over declared tasks, standing in for a real workload's.
SAMPLE_SITES = SiteRegistry(
    name="sample_decoder",
    sites=(
        Site("rms_norm", "rmsnorm", fused_rms_norm),
        Site("swiglu", "swiglu", float32_swiglu),
        Site("cross_entropy", "fused_linear_cross_entropy", eager_cross_entropy),
    ),
)

#: ``{site: task}`` for the sample registry.
SAMPLE_SITE_OPS = SAMPLE_SITES.site_ops


def other_registry(name: str = "other_decoder") -> SiteRegistry:
    """A second registry sharing no site name with :data:`SAMPLE_SITES`.

    The two-registry invariant needs two, and needs them disjoint: a site name
    existing somewhere is not a reason to accept a candidate for it here.
    """
    return SiteRegistry(
        name=name,
        sites=(
            Site("swiglu_mlp", "swiglu", float32_swiglu),
            Site("residual_rmsnorm", "rmsnorm", fused_rms_norm),
        ),
    )


# ── an injectable reference, shaped the way the deleted block's was ──────────
#
# The legacy ``llama3_decoder_layer`` reference took its normalization and its
# activation as injected parameters -- one ahead of the declared arguments, one
# appended behind a default -- so that a positional call with the declared
# arguments alone still landed exactly as before. That declaration is gone, but
# the property is generic: any reference that accepts injected spellings has to
# keep the declared arguments contiguous and positionally addressable, and has
# to default the trailing injection. This fixture is what the tests assert that
# against now.


def _synthetic_block(rms_norm, x, weight, cos, sin, eps=1e-5, swiglu=float32_swiglu):
    """Normalize, rotate, gate. Not a real architecture -- a real *shape*."""
    normalized = rms_norm(x, weight, eps)
    rotated = normalized * cos + normalized.roll(1, dims=-1) * sin
    return swiglu(rotated, normalized)


def synthetic_block_forward_ref(*args, **kwargs):
    """The definition: normalization spelled out, the way an oracle sees it."""
    def _rms_norm(t, w, eps):
        scale = torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps)
        return (t.float() * scale).to(t.dtype) * w

    return _synthetic_block(_rms_norm, *args, **kwargs)


def synthetic_block_op():
    """A declaration over :func:`synthetic_block_forward_ref`.

    Deliberately built here rather than registered: the registry holds tasks the
    benchmark measures, and this one exists only so tests have a declaration
    whose reference takes injected spellings.
    """
    from evograd.opdecl import Active, Inactive, Workload, declare_op

    return declare_op(
        name="synthetic_block",
        forward="tests._registry_fixture:synthetic_block_forward_ref",
        dims=("rows", "hidden"),
        args=(
            Active("x", "[rows, hidden]"),
            Active("weight", "[hidden]"),
            Active("cos", "[rows, hidden]"),
            Active("sin", "[rows, hidden]"),
            Inactive("eps", default=1e-5),
        ),
        output=Active("y", "[rows, hidden]"),
        forward_semantics="synthetic",
        backward_semantics="synthetic",
        correctness=(Workload(dims={"rows": 4, "hidden": 8}, dtype="float32"),),
        tolerances={"float32": (1e-5, 1e-5)},
    )
