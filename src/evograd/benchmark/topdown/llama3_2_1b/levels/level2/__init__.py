"""The three fused tasks one Llama-3.2-1B decoder layer decomposes into.

Each site package owns its contract and its reference. :mod:`manifest` is the
one place the site names, task keys and frequencies are written down.

Judging an implementation of one of these contracts is not done here -- that
belongs to :mod:`evograd.verification`.
"""

from .manifest import SITES, SITE_TASKS, TASKS, task_key

#: Site -> task key. Kept as the historical name for the same table.
DECLARATIONS = SITE_TASKS

#: The Level-2 task keys this workload exercises.
OPERATORS = TASKS

__all__ = ["DECLARATIONS", "OPERATORS", "SITES", "SITE_TASKS", "TASKS", "task_key"]
