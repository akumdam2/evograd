"""Model integrations of the verification gates.

Each module here connects one benchmark model to the generic gates: it reads
the model's task declarations and harvest snapshot, never edits them, and owns
what only that model knows (how to record its real training calls).

    python -m evograd.verification.integrations <model> {capture,fidelity,verify} ...
"""

from __future__ import annotations

import importlib

#: Model name (registry key) -> integration package.
PACKAGES = {
    "qwen3_0_6b": "qwen3_0_6b",
    "qwen3_5_4b": "qwen3_5_4b",
}
#: Other spellings accepted on the command line.
ALIASES: dict[str, str] = {}


def load(name: str):
    """The :class:`~.base.Model` of one integrated model."""
    key = ALIASES.get(name, name)
    if key not in PACKAGES:
        raise KeyError(f"no verification integration {name!r}; known: {sorted(PACKAGES)}")
    return importlib.import_module(f"{__name__}.{PACKAGES[key]}").MODEL
