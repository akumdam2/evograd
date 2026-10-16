"""The four fused sites an OLMoE decoder layer decomposes into.

Each site package owns its contract (``task``), its reference (``reference``:
float32 oracle and the production spelling, bitwise the layer's own) and its
capture (``capture``: one real invocation, from a captured layer when one
exists, else from a config-derived layer); :mod:`.manifest` is the one place the
site names, task keys and frequencies are written down. Judging an
implementation of one of these contracts is evaluation's.
"""

from .manifest import SITES, SITE_TASKS, TASKS, task_key

#: Site -> task key, under the name the other workloads use for it.
DECLARATIONS = SITE_TASKS

#: The Level-2 task keys this workload exercises.
OPERATORS = TASKS

__all__ = ["DECLARATIONS", "OPERATORS", "SITES", "SITE_TASKS", "TASKS", "task_key"]
