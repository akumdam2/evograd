"""The seven fused sites a Qwen3.5 decoder block decomposes into.

Each site package owns its contract (``task``), its reference (``reference``:
float32 oracle and the production spelling, bitwise the block's own) and its
capture (``capture``: one real invocation, from a captured layer when one
exists, else from a config-derived block); :mod:`.manifest` is the one place the
site names, task keys, block kinds and frequencies are written down. Judging an
implementation of one of these contracts is evaluation's.
"""

from .manifest import SITES, SITES_BY_BLOCK, SITE_TASKS, TASKS, task_key

#: Site -> task key, under the name the other workloads use for it.
DECLARATIONS = SITE_TASKS

#: The Level-2 task keys this workload exercises.
OPERATORS = TASKS

__all__ = ["DECLARATIONS", "OPERATORS", "SITES", "SITES_BY_BLOCK", "SITE_TASKS", "TASKS",
           "task_key"]
