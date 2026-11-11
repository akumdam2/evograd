"""The four fused tasks of a Qwen3-0.6B training step: three per decoder layer
(attention block, SwiGLU MLP, residual add + RMSNorm) and the LM head with the loss.

Each site package owns its contract and its reference; the real-payload capture
(:mod:`evograd.verification.integrations.qwen3_0_6b`) records its calls.
:mod:`manifest` is the one place the site names, task keys, frequencies and
module paths are written down; everything else reads them from there or from
the frozen harvest snapshot.

Judging an implementation of one of these contracts is not done here -- that
belongs to :mod:`evograd.verification`.
"""

from .manifest import SITES, SITE_TASKS, TASKS, task_key

#: Site -> task key. Kept as the historical name for the same table.
DECLARATIONS = SITE_TASKS

#: The Level-2 task keys this workload exercises.
OPERATORS = TASKS

__all__ = ["DECLARATIONS", "OPERATORS", "SITES", "SITE_TASKS", "TASKS", "task_key"]
