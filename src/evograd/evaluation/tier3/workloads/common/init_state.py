"""A frozen model state a workload starts from instead of its initialization.

Written by :mod:`evograd.verification.integrations.warm_start` (DiT-S/2,
MegaFold): ``{"schema", "model", "recipe", "state_dict"}`` in one ``torch.save``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

SCHEMA = "evograd-generated-init-state/1"


def load(path: str | Path, model: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """``(state_dict, identity)`` of the state at ``path``, which must be ``model``'s."""
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema") != SCHEMA or data.get("model") != model:
        raise ValueError(f"{path} is not an initial state of {model}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return data["state_dict"], {"sha256": digest.hexdigest(), "recipe": data.get("recipe")}
