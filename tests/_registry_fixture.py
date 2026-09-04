"""A sample site registry, for tests that need *a* registry rather than a model's.

``bench/tier3_patch.py`` used to ship ``LLAMA_SITES``, and much of the tier-3
suite reached for it as a convenient stand-in. That was the coupling those very
tests exist to forbid: a registry names one model's patchable places, so it
belongs to a workload package, and the patcher must not carry one.

The tests still need a registry to exercise the machinery with -- and, for the
invariant that two registries never share a namespace, two of them. So they are
built here, out of declared operators, and owned by the tests.

``SAMPLE_SITES`` deliberately reuses the three sites the deleted built-in
carried: ``rmsnorm``, ``swiglu`` and ``fused_linear_cross_entropy`` are real
declarations with real pair baselines, which is what lets the baseline-discovery
tests find something to discover.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evograd.bench.tier3_patch import Site, SiteRegistry
from evograd.ops.level3.llama3_decoder_layer.forward_ref import (
    _default_swiglu,
    _rms_norm_fused,
)


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


#: Three sites over declared operators, standing in for a real workload's.
SAMPLE_SITES = SiteRegistry(
    name="sample_decoder",
    sites=(
        Site("rms_norm", "rmsnorm", _rms_norm_fused),
        Site("swiglu", "swiglu", _default_swiglu),
        Site("cross_entropy", "fused_linear_cross_entropy", eager_cross_entropy),
    ),
)

#: ``{site: operator}`` for the sample registry.
SAMPLE_SITE_OPS = SAMPLE_SITES.site_ops


def other_registry(name: str = "other_decoder") -> SiteRegistry:
    """A second registry sharing no site name with :data:`SAMPLE_SITES`.

    The two-registry invariant needs two, and needs them disjoint: a site name
    existing somewhere is not a reason to accept a candidate for it here.
    """
    return SiteRegistry(
        name=name,
        sites=(
            Site("swiglu_mlp", "swiglu", _default_swiglu),
            Site("residual_rmsnorm", "rmsnorm", _rms_norm_fused),
        ),
    )
