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
    "llama_3_2_1b": "llama3_2_1b",
    "torchrec_dlrm": "torchrec_dlrm",
    "dit_s_2": "dit_s_2",
    "convnextv2_tiny": "convnextv2_tiny",
    "qwen2_vl_2b": "qwen2_vl_2b",
    "olmoe_1b_7b": "olmoe_1b_7b",
}
#: Other spellings accepted on the command line (package names).
ALIASES: dict[str, str] = {"llama3_2_1b": "llama_3_2_1b"}


def load(name: str):
    """The :class:`~.base.Model` of one integrated model."""
    key = ALIASES.get(name, name)
    if key not in PACKAGES:
        raise KeyError(f"no verification integration {name!r}; known: {sorted(PACKAGES)}")
    return importlib.import_module(f"{__name__}.{PACKAGES[key]}").MODEL
