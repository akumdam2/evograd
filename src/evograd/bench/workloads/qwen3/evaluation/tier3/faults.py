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
    runtime_forward_for as _runtime,
    scale_grad as _scale_grad,
    smallest_rejected,
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


# ── Qwen3's own optimizer-side faults ────────────────────────────────────────
#
# `StateFault` and `state_catalogue` are defined here rather than re-exported
# from `tier3_gate.faults`, because both differ for this model.
#
# The shared `tier3_gate.faults.state_catalogue` is deliberately not re-exported
# here. It lists `wrong_update` as a required control, and at this model's
# magnitudes bfloat16 rounds that fault away: one AdamW step at lr=1e-4 moves a
# projection weight by exactly two bfloat16 ULPs, so scaling the update by 1.02
# leaves most stored values bit-identical and a norm weight entirely unchanged.
# Qwen3 therefore requires a two-ULP perturbation of the *stored* parameter,
# which changes every element by construction, and keeps `wrong_update` as a
# diagnostic that reports how much of itself reached memory.



# ── faults that are not in a kernel ──────────────────────────────────────────
#
# A kernel fault reaches the gate through the model. These do not: a wrong
# parameter update with perfectly correct gradients, a corrupted Adam moment, a
# step counter that has lost count. Nothing a kernel does can produce them, so
# nothing a kernel-shaped control can prove the gate would catch them. They are
# applied to a captured step instead, which is the only place they exist.


_BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
         torch.float32: torch.int32, torch.float64: torch.int64}


def perturb_ulps(tensor: torch.Tensor, ulps: int) -> torch.Tensor:
    """Move every element ``ulps`` representable steps away from zero.

    Stepping the integer bit pattern is what makes this a *guaranteed* change:
    adding a float quantity can round back to where it started, and at bfloat16
    that is the normal case rather than the corner case. Incrementing the
    pattern cannot -- the result is a different bit pattern by construction, and
    the smallest change the format is able to represent.

    Away from zero in both directions. IEEE floats are sign-magnitude, so the
    raw pattern grows with |value| on each side independently -- one increment
    serves both signs, and no branch on sign is needed or correct.
    """
    wide = _BITS[tensor.dtype]
    bits = tensor.view(wide).clone()
    return (bits + ulps).view(tensor.dtype)


def observability(clean: dict[str, torch.Tensor],
                  damaged: dict[str, torch.Tensor]) -> dict[str, Any]:
    """How much of a fault actually reached stored model state.

    Without this a control report cannot distinguish "the gate looked and saw
    nothing" from "there was nothing to see". A fault whose stored footprint is
    zero has not been missed; it has not happened.
    """
    from evograd.bench.tier3_gate.numerics import role_of

    changed = total = 0
    by_role: dict[str, dict[str, int]] = {}
    for name, before in clean.items():
        after = damaged.get(name)
        if after is None or after.shape != before.shape:
            continue
        wide = _BITS.get(before.dtype)
        if wide is None:
            continue
        differing = int((before.view(wide) != after.view(wide)).sum())
        changed += differing
        total += before.numel()
        entry = by_role.setdefault(role_of(name), {"changed": 0, "elements": 0})
        entry["changed"] += differing
        entry["elements"] += before.numel()
    return {
        "stored_elements": total,
        "stored_elements_changed": changed,
        "stored_fraction_changed": (changed / total) if total else 0.0,
        "observable": changed > 0,
        "roles_with_no_stored_change": sorted(
            r for r, e in by_role.items() if e["changed"] == 0
        ),
        "per_role_fraction_changed": {
            r: (e["changed"] / e["elements"] if e["elements"] else 0.0)
            for r, e in sorted(by_role.items())
        },
    }


@dataclass(frozen=True)
class StateFault:
    """A defect injected into the step's recorded state, not into a kernel."""

    name: str
    family: str
    magnitude: float
    describe: str

    #: Which quantity this fault corrupts, for reading its evidence. A moment
    #: or a step counter lives in the optimizer, not in the parameters, so its
    #: stored-parameter footprint is zero by design and says nothing about
    #: whether the fault happened.
    @property
    def scope(self) -> str:
        return "stored_parameters" if self.family == "updates" else "optimizer_state"

    def apply(self, captured: dict[str, Any]) -> dict[str, Any]:
        damaged = dict(captured)
        if self.family == "steps":
            damaged["steps"] = {
                name: (None if value is None else value + self.magnitude)
                for name, value in captured["steps"].items()
            }
            return damaged
        scaled = {
            name: tensor * (1.0 + self.magnitude)
            for name, tensor in captured[self.family].items()
        }
        damaged[self.family] = scaled
        if self.family == "updates":
            # A wrong update is only wrong once it is *stored*. Re-deriving the
            # parameter from the corrupted update and casting it back is what
            # makes the evidence real: at bfloat16 most of a 2% change does not
            # survive that cast, and a control that skipped it would report a
            # fault the model never saw.
            stored = {}
            for name, value in captured["stored"].items():
                update = captured["updates"].get(name)
                if update is None:
                    stored[name] = value
                    continue
                before = value.float() - update
                stored[name] = (before + scaled[name]).to(value.dtype)
            damaged["stored"] = stored
        return damaged

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "scope": self.scope,
                "magnitude": self.magnitude, "describes": self.describe}


@dataclass(frozen=True)
class StoredParameterFault:
    """A wrong parameter value that bfloat16 cannot round away.

    The multiplicative ``wrong_update`` control was written before it was
    measured, and the measurement says it cannot work as a required control.
    One AdamW step at lr=1e-4 moves a projection weight by 1.22e-4 -- exactly
    two bfloat16 ULPs at |p| ~ 1.3e-2 -- so scaling that update by 1.02 changes
    the stored value for about four percent of elements and leaves the rest
    bit-identical. On a norm weight, where |p| = 1 and the ULP is 7.8e-3, the
    whole update rounds away and the realized update is exactly zero; scaling
    zero by anything is still zero.

    This one is defined in units of the storage format instead. Stepping the
    bit pattern is guaranteed to change every element it touches, whatever the
    dtype and whatever the magnitude of the value, so "not detected" can only
    ever mean the gate missed it.
    """

    name: str
    ulps: int
    describe: str
    family: str = "stored"
    magnitude: float = 0.0

    def apply(self, captured: dict[str, Any]) -> dict[str, Any]:
        damaged = dict(captured)
        stored, updates = {}, dict(captured["updates"])
        for name, value in captured["stored"].items():
            moved = perturb_ulps(value, self.ulps)
            stored[name] = moved
            # The update is what the envelope judges, and it is the difference
            # between stored endpoints -- so a wrong stored value *is* a wrong
            # update, by exactly the amount the storage moved.
            if name in updates:
                updates[name] = updates[name] + (moved.float() - value.float())
        damaged["stored"] = stored
        damaged["updates"] = updates
        return damaged

    scope = "stored_parameters"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "scope": self.scope,
                "ulps": self.ulps, "magnitude": float(self.ulps),
                "describes": self.describe}


def state_catalogue(magnitude: float = 0.02) -> list[StateFault | StoredParameterFault]:
    """The optimizer-side defects a provider must not be able to hide.

    ``wrong_update`` is deliberately absent: see :func:`diagnostic_catalogue`.
    """
    return [
        StoredParameterFault(
            "stored_param_ulp", 2,
            "every stored parameter moved two ULPs away from zero",
        ),
        StateFault("corrupt_exp_avg", "exp_avg", magnitude,
                   "Adam's first moment scaled by 1+m"),
        StateFault("corrupt_exp_avg_sq", "exp_avg_sq", magnitude,
                   "Adam's second moment scaled by 1+m"),
        StateFault("wrong_step_count", "steps", 1.0,
                   "the optimizer has taken one more step than the reference"),
    ]


def diagnostic_catalogue(magnitude: float = 0.02) -> list[StateFault]:
    """Faults that are run and reported but not required to be rejected.

    ``wrong_update`` at 2% is here because bfloat16 storage erases most of it
    before it reaches model state -- entirely, on every norm role. Requiring
    its rejection would be requiring the gate to detect something that did not
    happen. It is still worth running: its observability evidence is the
    standing measurement of how much of a small multiplicative update fault
    this dtype can even express, and that number is what justifies the ULP
    control replacing it.
    """
    return [
        StateFault("wrong_update", "updates", magnitude,
                   "correct gradients, an update scaled by 1+m"),
    ]
