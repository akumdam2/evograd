"""The Transformers module the production spellings call, resolved at import.

Inside a function, ``from transformers.models.qwen3_5 import modeling_qwen3_5``
goes through Transformers' lazy package object, which ``torch.compile`` refuses
to trace; resolved here once, the name is an ordinary module global.
``None`` without Transformers.

The import-time kernel dispatch (T-dispatch) is unaffected: the spellings reach
the torch functions through ``__wrapped__``.
"""

from __future__ import annotations

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
except ImportError:  # pragma: no cover - transformers is optional for declarations
    modeling = None

__all__ = ["modeling"]
