"""The Transformers module the production spellings call, resolved at import.

Inside a function, ``from transformers.models.olmoe import modeling_olmoe``
goes through Transformers' lazy package object, which ``torch.compile`` refuses
to trace; resolved here once, the name is an ordinary module global. ``None``
without Transformers.
"""

from __future__ import annotations

try:
    from transformers.models.olmoe import modeling_olmoe as modeling
except ImportError:  # pragma: no cover - transformers is optional for declarations
    modeling = None

__all__ = ["modeling"]
