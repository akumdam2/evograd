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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .sites import (
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    structural_identity_kernels,
)


class _GradScale(torch.autograd.Function):
    """Identity forward, scaled backward. The forward cannot detect this."""

    @staticmethod
    def forward(ctx, tensor, factor):
        ctx.factor = factor
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


def _scale_grad(value, factor: float):
    if isinstance(value, tuple):
        return tuple(_GradScale.apply(v, factor) for v in value)
    return _GradScale.apply(value, factor)


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


def _runtime(op_name: str) -> Callable:
    """The declaration's production spelling, called the way a candidate is.

    A fault must have the *declared operator's* signature, not the adapter's
    internal production one: once a site is patched the adapter calls it with
    the contract's argument list, and a fault written against the other
    signature would be rejected by a TypeError. That is not the gate working.
    """
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.ops import get_op

    return resolve_runtime_forward(get_op(op_name))


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


def smallest_rejected(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per fault kind, the smallest magnitude rejected on every seed tested."""
    by_kind: dict[str, dict[float, list[bool]]] = {}
    for record in results:
        by_kind.setdefault(record["fault"]["name"], {}).setdefault(
            record["fault"]["magnitude"], []
        ).append(bool(record["rejected"]))
    out: dict[str, Any] = {}
    for kind, magnitudes in by_kind.items():
        always = [m for m, verdicts in magnitudes.items() if verdicts and all(verdicts)]
        out[kind] = {
            "smallest_always_rejected": min(always) if always else None,
            "tested": sorted(magnitudes),
            "per_magnitude": {
                str(m): {"rejected": sum(v), "of": len(v)}
                for m, v in sorted(magnitudes.items())
            },
        }
    return out


# ── faults that are not in a kernel ──────────────────────────────────────────
#
# A kernel fault reaches the gate through the model. These do not: a wrong
# parameter update with perfectly correct gradients, a corrupted Adam moment, a
# step counter that has lost count. Nothing a kernel does can produce them, so
# nothing a kernel-shaped control can prove the gate would catch them. They are
# applied to a captured step instead, which is the only place they exist.


@dataclass(frozen=True)
class StateFault:
    """A defect injected into the step's recorded state, not into a kernel."""

    name: str
    family: str
    magnitude: float
    describe: str

    def apply(self, captured: dict[str, Any]) -> dict[str, Any]:
        damaged = dict(captured)
        if self.family == "steps":
            damaged["steps"] = {
                name: (None if value is None else value + self.magnitude)
                for name, value in captured["steps"].items()
            }
            return damaged
        damaged[self.family] = {
            name: tensor * (1.0 + self.magnitude)
            for name, tensor in captured[self.family].items()
        }
        return damaged

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family,
                "magnitude": self.magnitude, "describes": self.describe}


def state_catalogue(magnitude: float = 0.02) -> list[StateFault]:
    """The four defects that live in the optimizer rather than in a kernel."""
    return [
        StateFault("wrong_update", "updates", magnitude,
                   "correct gradients, an update scaled by 1+m"),
        StateFault("corrupt_exp_avg", "exp_avg", magnitude,
                   "Adam's first moment scaled by 1+m"),
        StateFault("corrupt_exp_avg_sq", "exp_avg_sq", magnitude,
                   "Adam's second moment scaled by 1+m"),
        StateFault("wrong_step_count", "steps", 1.0,
                   "the optimizer has taken one more step than the reference"),
    ]
