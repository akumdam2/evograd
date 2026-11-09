"""Tier 3 for Qwen3-0.6B: replace an operator inside the real model.

Four sites in the live ``Qwen3ForCausalLM`` can be swapped without touching the
call site, model state, parameter names, or training loop. Correctness is
judged by the verification package's Gates 1/2/3.

    sites.py      the four adapters, and what "production" spells at each
    workload.py   building the canonical step, feeding it, its loss
    textdata.py   the pinned WikiText-2 real-text stream
"""

from .workload import Qwen3Workload

__all__ = ["Qwen3Workload"]
