"""Tier 3 for Qwen3-0.6B: replace an operator inside the real model.

Four sites in the live ``Qwen3ForCausalLM`` can be swapped without touching the
call site, the model state, the parameter names, or the training loop. What
makes that measurable rather than merely possible is the untimed gate in
:mod:`.gate`, which runs in a fixed order and refuses to time anything that
fails it:

1. :mod:`.validate` -- the sites hold at the shapes this model supplies;
2. purity -- the provider is a function of its arguments;
3. :mod:`.boundary` -- all 140 invocations match their declaration;
4. numerics -- the whole model stays inside a calibrated envelope;
5. the calibrated loss trajectory;
6. invocation counts and patch provenance.

:mod:`.calibrate` measures the envelope on the machine it will be enforced on,
:mod:`.faults` and :mod:`.controls` show it still rejects a wrong provider.

**Where the halves live.** Steps 2 and 4 ask questions about a *kernel*, not
about an architecture, so they are asked by
:mod:`evograd.bench.tier3_gate` -- shared with every other workload. What stays
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
