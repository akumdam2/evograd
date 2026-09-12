"""One Llama-3-8B Level-2 boundary: its contract, reference and capture.

``task`` is the contract, ``reference`` defines it, and ``capture`` derives
this site's case from a captured layer. Judging an implementation of it belongs
to :mod:`evograd.evaluation.workloads.llama3_2_1b.level2`.
"""

from .task import op

__all__ = ["op"]
