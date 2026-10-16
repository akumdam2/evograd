"""Tier 3 for OLMoE-1B-7B: replace an operator inside the real model.

The four sites are patched on all 16 layers of ``OlmoeForCausalLM`` at
B2 x T2048 (benchmark Level 3, the whole-model step).

    sites.py      the four sites, their production spellings, the adapters
    workload.py   the canonical step: build, batch, loss, describe, report
    boundary.py   per-invocation checks on chosen layers
    purity.py     calls per site and provider reconstruction
    adapter.py    what the tier-3 CLI needs to know

No whole-model gate exists yet: every provider row says
``evaluation_complete: false`` and why.
"""

from .workload import OlmoeWorkload

__all__ = ["OlmoeWorkload"]
