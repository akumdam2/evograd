"""Tier 3 for OLMoE-1B-7B: replace an operator inside the real model.

The three sites are patched on all 16 layers of ``OlmoeForCausalLM`` at
B2 x T2048 (benchmark Level 3, the whole-model step).

    sites.py      the three sites, their production spellings, the adapters
    workload.py   the canonical step: build, batch, loss, describe, report
"""

from .workload import OlmoeWorkload

__all__ = ["OlmoeWorkload"]
