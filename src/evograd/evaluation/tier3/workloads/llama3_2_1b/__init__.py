"""Tier 3 for Llama-3.2-1B: replace an operator inside the real model.

Four sites in the live ``LlamaForCausalLM`` can be swapped without touching the
call site, the model state, the parameter names, or the training loop.
Correctness is judged by the verification package's Gates 1/2/3.

    sites.py      the four adapters, and what "production" spells at each
    workload.py   building the canonical step, feeding it, its loss
    training.py   the synthetic train/validation batch streams

**Real text** is a separate workload identity: the pinned Llama-3.2-1B
checkpoint on WikiText-2 raw, packed by Qwen3's ``textdata``. The canonical
synthetic workload keeps its "no Hub token" property; only the one-time
download of the gated checkpoint needs one.
"""

from .workload import Llama3Workload

__all__ = ["Llama3Workload"]
