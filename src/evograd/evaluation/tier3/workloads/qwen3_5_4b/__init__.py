"""Tier 3 for Qwen3.5-4B: replace an operator inside the real model.

The seven sites are patched on all 32 layers of ``Qwen3_5ForCausalLM`` at
B1 x T2048, each layer by its type (benchmark Level 3, the whole-model step).

    sites.py      the seven sites, their production spellings, the adapters
    workload.py   the canonical step: build, batch, loss, describe, report
    boundary.py   per-invocation checks on chosen layers
    purity.py     calls per site (by layer type) and provider reconstruction
    adapter.py    what the tier-3 CLI needs to know

No whole-model gate exists yet: every provider row says
``evaluation_complete: false`` and why.
"""

from .workload import Qwen3_5Workload

__all__ = ["Qwen3_5Workload"]
