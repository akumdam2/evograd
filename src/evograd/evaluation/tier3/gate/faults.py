"""Fault-injection machinery that is the same whatever model is being gated.

A threshold that accepts everything is not a threshold. The calibrated envelope
is derived from reference runs alone, so the only way to know it still rejects a
wrong kernel is to build wrong kernels and watch it rejected them. That argument
is not about any architecture, and neither is most of what it needs.

**What is here.** The autograd trick for "exact forward, wrong backward", the
resolution of a declaration's production spelling, the defects that live in the
optimizer rather than in a kernel, and the table that reads a run of controls
back as a sensitivity.

**What is not, and why.** The *catalogue* of kernel faults stays in the workload
package, because building one needs facts a model owns:

* which site to damage -- site names belong to a workload's registry;
* the operator's **output arity**. ``one_role`` perturbs the query projection of
  a three-output boundary and leaves k and v exact; ``non_finite`` poisons the
  normalized output of a two-output boundary and passes the residual through.
  Written generically, both become "perturb output 0", which is a different and
  weaker control;
* the model's **layer count**. ``layer_subset`` is wrong in the first half of
  the layers and ``single_layer`` in exactly one, and the whole point of those
  two is the fraction of the model they touch -- 1/28th is the regime where a
  pooled whole-model statistic stops being able to see a defect. A generic
  version parameterized by depth would be the same code and a different claim.

Generalizing those would change what the controls test, which is the one thing
in this repository that must not drift quietly: they are the evidence the gate
works at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


class GradScale(torch.autograd.Function):
    """Identity forward, scaled backward. The forward cannot detect this."""

    @staticmethod
    def forward(ctx, tensor, factor):
        ctx.factor = factor
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


def scale_grad(value, factor: float):
    """Scale the gradient flowing back through ``value``, leaving it unchanged.

    The canonical reason a whole-model gate exists: nothing in a forward
    comparison can see this, so a site check that only compares outputs passes
    a kernel whose backward is off by a constant.
    """
    if isinstance(value, tuple):
        return tuple(GradScale.apply(v, factor) for v in value)
    return GradScale.apply(value, factor)


def runtime_forward_for(op_name: str) -> Callable:
    """The declaration's production spelling, called the way a candidate is.

    A fault must have the *declared operator's* signature, not the adapter's
    internal production one: once a site is patched the adapter calls it with
    the contract's argument list, and a fault written against the other
    signature would be rejected by a TypeError. That is not the gate working.
    """
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.ops import get_op

    return resolve_runtime_forward(get_op(op_name))


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
#
# These are AdamW's state, not any architecture's, so both the faults and the
# catalogue are shared.


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
