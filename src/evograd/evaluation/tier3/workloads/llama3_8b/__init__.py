"""Tier 3 for Llama-3-8B: replace an operator inside the real model.

Four sites in the live ``LlamaForCausalLM`` can be swapped without touching the
call site, the model state, the parameter names, or the training loop. What
makes that measurable rather than merely possible is the untimed gate in
:mod:`.gate`, which runs in a fixed order and refuses to time anything that
fails it:

1. :mod:`.validate` -- the sites hold at the shapes this model supplies;
2. purity -- the provider is a function of its arguments;
3. :mod:`.boundary` -- all 160 invocations match their declaration;
4. numerics -- the whole model stays inside a calibrated envelope;
5. the calibrated loss trajectory;
6. invocation counts and patch provenance.

:mod:`.calibrate` measures the envelope on the machine it will be enforced on,
:mod:`.faults` and :mod:`.controls` show it still rejects a wrong provider.

**Where the halves live.** Steps 2 and 4 ask questions about a *kernel*, not
about an architecture, so they are asked by
:mod:`evograd.evaluation.tier3.gate` -- shared with every other workload. What stays
here is what only Llama-3 can answer:

    sites.py      the four adapters, and what "production" spells at each
    workload.py   building the canonical step, feeding it, its loss
    adapter.py    what the tier-3 CLI needs to know, so the CLI need not
    validate.py   preflight at this model's shapes
    boundary.py   160 invocations, because this model has 32 layers
    calibrate.py  the envelope, measured on this model on this machine
    faults.py     the kernel-fault catalogue: which site, which output, how deep
    purity.py     how many calls a site is worth, and how to rebuild a provider
    gate.py       the order the stages run in, and what a refusal is called

The dependency runs one way. This package imports the shared gate; the shared
gate imports nothing from here, which is what keeps a threshold from quietly
acquiring a branch on a model name.

**What is absent relative to Qwen3, and why.** Two things, for two different
reasons.

The simplified numerics policy (``simple.py``/``calibrate_simple.py``) was
declined while its trusted reference was the bound pair, which recomputes
through the same ``runtime_forward`` -- measured drift identically zero, and
every threshold derived from it collapsing to its floor. That is no longer the
default anchor: Qwen3's ``calibrate_simple`` now takes
``--trusted-reference {torch_compile,bound_pair}`` and defaults to
``torch_compile``, which is genuinely different arithmetic. The original
objection therefore does *not* apply here unchanged, and adopting it is an open
decision rather than a settled one. It needs no harvest and no pretrained
weights.

The four-part real-text protocol (``protocol4.py``, ``prediction.py``,
``training.py``, ``textdata.py``) needs a pinned pretrained checkpoint and
tokenizer. Meta-Llama-3-8B is a gated repository, so adopting it would forfeit
the "no Hub token" property this workload is built around. A design decision,
not a port.

**State.** Everything in this package is written and importable. What none of it
can do yet is *run against a calibrated gate*, because a calibration is measured
per model per machine and none has been executed for Llama. See
:data:`gate.DEFAULT_ARTIFACT` and ``README.md``.
"""

from .workload import Llama3Workload

__all__ = ["Llama3Workload"]
