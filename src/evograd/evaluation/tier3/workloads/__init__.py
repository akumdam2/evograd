"""Workload-specific adapters for Tier-3 model evaluation.

The Tier-3 runner and patcher are architecture-independent.  This registry is
the only place that maps a public ``--model`` name to model-specific build and
provider hooks.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Tier3Adapter:
    """How the generic Tier-3 CLI builds and patches one workload."""

    name: str
    build: Callable[[Any], Any]
    providers: Callable[[Any, Any], dict[str, Any]] | None = None
    options: frozenset[str] = field(default_factory=frozenset)
    summary: str = ""


TIER3_ADAPTERS: dict[str, str] = {
    "qwen3_0_6b": (
        "evograd.evaluation.tier3.workloads.qwen3_0_6b.adapter:ADAPTER"
    ),
    "llama_3_2_1b": (
        "evograd.evaluation.tier3.workloads.llama3_2_1b.adapter:ADAPTER"
    ),
    "alphafold3_2l": (
        "evograd.evaluation.tier3.workloads.alphafold3.adapter:ADAPTER_2L"
    ),
    "alphafold3": (
        "evograd.evaluation.tier3.workloads.alphafold3.adapter:ADAPTER"
    ),
}


class UnknownWorkload(KeyError):
    """A workload name with no Tier-3 adapter."""


def tier3_model_names() -> tuple[str, ...]:
    return tuple(TIER3_ADAPTERS)


def tier3_adapter(name: str) -> Tier3Adapter:
    try:
        target = TIER3_ADAPTERS[name]
    except KeyError:
        raise UnknownWorkload(
            f"no tier-3 workload named {name!r}; this repository carries "
            f"{sorted(TIER3_ADAPTERS)}"
        ) from None
    module_path, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_path), attribute)


__all__ = [
    "TIER3_ADAPTERS",
    "Tier3Adapter",
    "UnknownWorkload",
    "tier3_adapter",
    "tier3_model_names",
]
