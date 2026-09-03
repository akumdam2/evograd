"""Faults the whole-model gate has to catch, and the smallest one it does.

A threshold that accepts everything is not a threshold. The calibrated envelope
is derived from reference runs alone, so the only way to know it still rejects a
wrong kernel is to build wrong kernels and watch it reject them.

Six shapes of fault, chosen because they fail differently:

``grad_scale``      the forward is exact and the backward is off by a factor.
                    Nothing in a forward comparison can see it; it is the
                    canonical reason a whole-model gate exists.
``output_scale``    a small relative error in the forward, propagated.
``dropped_row``     one token's contribution removed at a reduction site --
                    an off-by-one bound, the error a reduction-scaled tolerance
                    is most at risk of hiding.
``one_role``        a perturbation confined to a single projection role, which a
                    pooled threshold would average away.
``stateful``        correct for the first few calls and wrong afterwards, so it
                    passes site preflight and fails in the model. This is the
                    fault the site gate structurally cannot catch.
``non_finite``      an infinity in one site's output.

Each is a *reference* kernel with a defect injected, never an evolved program.

**Why this is Qwen3's and not shared.** The mechanics -- the backward-scaling
autograd function, the optimizer-state faults, the sensitivity table -- live in
:mod:`evograd.bench.tier3_gate.faults`. Constructing a *kernel* fault does not
generalize, because three of these encode facts about this model:

* ``one_role`` unpacks ``q, k, v`` and disturbs only the query projection, which
  is a statement about a three-output boundary;
* ``non_finite`` unpacks ``normed, summed`` and poisons only the first, which is
  a statement about a two-output boundary;
* ``layer_subset`` and ``single_layer`` are wrong in 14 of 28 layers and in 1 of
  28. The fraction is the control: 1/28th is where a pooled whole-model
  statistic stops being able to see a defect.

Rewritten as "perturb output 0 of an n-output operator at a configurable depth",
they would be the same code making a weaker claim. These are the evidence the
gate works, so they say exactly what they mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from evograd.bench.tier3_gate.faults import (  # noqa: F401  (re-export)
    GradScale as _GradScale,
    StateFault,
    runtime_forward_for as _runtime,
    scale_grad as _scale_grad,
    smallest_rejected,
    state_catalogue,
)

from .sites import (
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    structural_identity_kernels,
)


@dataclass(frozen=True)
class Fault:
    """One injected defect, named and sized so a rejection table can be read."""

    name: str
    site: str
    magnitude: float
    describe: str

    def apply(self, workload, kernels=None):
        """A kernel set identical to structural identity except at one site."""
        from evograd.bench.tier3_patch import KernelSource, patch

        base = kernels or structural_identity_kernels(workload.site_registry)
        registry = base.registry
        kernel = _build(self, registry)
        return patch(
            base, self.site, kernel,
            # The magnitude travels in the origin so a child process can rebuild
            # this exact fault. A stateful fault must be rebuilt rather than
            # reused: the purity gate calls it dozens of times, and a counter
            # the model then inherits would make the two stages interfere.
            source=KernelSource(site=self.site, op_name=registry.require(self.site).op,
                                module=None,
                                origin=f"fault:{self.name}@{self.magnitude}"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "site": self.site, "magnitude": self.magnitude,
                "describes": self.describe}



def _build(fault: "Fault", registry) -> Callable:
    op_name = registry.require(fault.site).op

    if fault.name == "grad_scale":
        reference = _runtime(op_name)

        def kernel(*args):
            return _scale_grad(reference(*args), 1.0 + fault.magnitude)
        return kernel

    if fault.name == "output_scale":
        reference = _runtime(op_name)

        def kernel(*args):
            return reference(*args) * (1.0 + fault.magnitude)
        return kernel

    if fault.name == "dropped_row":
        reference = _runtime(op_name)

        def kernel(x, *rest):
            # One token contributes nothing, exactly as an off-by-one bound on
            # a row loop would produce.
            damaged = x.clone()
            damaged[:, -1, :] = 0.0
            return reference(damaged, *rest)
        return kernel

    if fault.name == "one_role":
        reference = _runtime(op_name)

        def kernel(*args):
            q, k, v = reference(*args)
            # Only the query projection is disturbed; k and v are exact.
            return q * (1.0 + fault.magnitude), k, v
        return kernel

    if fault.name == "stateful":
        reference = _runtime(op_name)
        state = {"calls": 0}

        def kernel(*args):
            state["calls"] += 1
            out = reference(*args)
            # Correct while a site gate is exercising it, wrong once a model has
            # called it more times than any preflight does.
            return out * 1.02 if state["calls"] > fault.magnitude else out
        return kernel

    if fault.name == "non_finite":
        reference = _runtime(op_name)

        def kernel(*args):
            normed, summed = reference(*args)
            return normed * float("inf"), summed
        return kernel

    if fault.name == "layer_subset":
        reference = _runtime(op_name)
        state = {"calls": 0}

        def kernel(*args):
            # Wrong in the first half of the model, exact in the second. The
            # kind of defect a kernel with a shape- or index-dependent bug
            # produces, and the kind a single representative layer misses.
            state["calls"] += 1
            out = reference(*args)
            return out * (1.0 + fault.magnitude) if state["calls"] <= 14 else out
        return kernel

    if fault.name == "single_layer":
        reference = _runtime(op_name)
        state = {"calls": 0}

        def kernel(*args):
            # One layer out of 28. Averaged over the whole model this is a
            # 1/28th-weight perturbation, which is exactly the regime where a
            # pooled whole-model statistic stops being able to see it.
            state["calls"] += 1
            out = reference(*args)
            return out * (1.0 + fault.magnitude) if state["calls"] == 14 else out
        return kernel

    raise ValueError(f"unknown fault {fault.name!r}")


#: The catalogue, smallest magnitude per kind first. The reported sensitivity is
#: the smallest magnitude rejected on every holdout seed.
def catalogue(magnitudes=(0.001, 0.005, 0.02)) -> list[Fault]:
    faults: list[Fault] = []
    for magnitude in magnitudes:
        faults += [
            Fault("grad_scale", SITE_QKV, magnitude,
                  "exact forward, backward scaled by 1+m"),
            Fault("output_scale", SITE_MLP, magnitude,
                  "MLP output scaled by 1+m"),
            Fault("one_role", SITE_QKV, magnitude,
                  "only the query projection scaled by 1+m"),
        ]
    faults += [
        Fault("dropped_row", SITE_MLP, 1.0,
              "one token's MLP contribution zeroed"),
        Fault("stateful", SITE_MLP, 8.0,
              "correct for the first 8 calls, 2% high afterwards"),
        Fault("non_finite", SITE_RESIDUAL, float("inf"),
              "an infinity in one residual-norm output"),
        Fault("layer_subset", SITE_MLP, 0.02,
              "2% high in the first 14 layers, exact in the rest"),
        Fault("single_layer", SITE_MLP, 0.02,
              "2% high in one layer out of 28, exact everywhere else"),
    ]
    return faults
