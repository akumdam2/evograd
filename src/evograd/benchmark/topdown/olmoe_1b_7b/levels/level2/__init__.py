"""The three fused sites an OLMoE decoder layer decomposes into.

Each site package owns its contract (``task``) and its reference
(``reference``: float32 oracle and the production spelling, bitwise the
model's own); :mod:`.manifest` is the one place the site names, task keys and
frequencies are written down. Real invocations are the verification capture's
(:mod:`evograd.verification.integrations.olmoe_1b_7b.capture`); judging an
implementation of one of these contracts is evaluation's.
"""

from .manifest import SITES, SITE_TASKS, TASKS, task_key

#: Site -> task key, under the name the other workloads use for it.
DECLARATIONS = SITE_TASKS

#: The Level-2 task keys this workload exercises.
OPERATORS = TASKS

__all__ = ["DECLARATIONS", "OPERATORS", "SITES", "SITE_TASKS", "TASKS", "task_key"]
