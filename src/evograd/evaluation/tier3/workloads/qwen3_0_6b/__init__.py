"""Tier 3 for Qwen3-0.6B: replace an operator inside the real model.

Four sites in the live ``Qwen3ForCausalLM`` can be swapped without touching the
call site, model state, parameter names, or training loop. The workload owns
the model-specific local output/gradient, valid-token KL, full-gradient-vector,
training-behavior, invocation-count, purity, and patch-provenance checks.
Thresholded checks consume a calibration frozen before candidate evaluation;
training/validation loss behavior remains diagnostic where the current policy
does not promote it to a gate.

:mod:`.calibrate` measures the envelope on the machine it will be enforced on,
:mod:`.faults` and :mod:`.controls` show it still rejects a wrong provider.

**Where the halves live.** Steps 2 and 4 ask questions about a *kernel*, not
about an architecture, so they are asked by
:mod:`evograd.evaluation.tier3.gate` -- shared with every other workload. What stays
here is what only Qwen3 can answer:

    sites.py      the four adapters, and what "production" spells at each
    workload.py   building the canonical step, feeding it, its loss
    adapter.py    what the tier-3 CLI needs to know, so the CLI need not
    validate.py   preflight at this model's observed shapes
    boundary.py   140 invocations, because this model has 28 layers
    calibrate.py  the envelope, measured on this model on this machine
    faults.py     the kernel-fault catalogue: which site, which output, how deep
    purity.py     how many calls a site is worth, and how to rebuild a provider
    gate.py       the order the stages run in, and what a refusal is called

The dependency runs one way. This package imports the shared gate; the shared
gate imports nothing from here, which is what keeps a threshold from quietly
acquiring a branch on a model name.
"""

from .workload import Qwen3Workload

__all__ = ["Qwen3Workload"]
