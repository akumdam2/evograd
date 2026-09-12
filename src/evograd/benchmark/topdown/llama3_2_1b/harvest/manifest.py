"""Llama-3's manifest: its schema, the call sites it patched, and its label.

Aggregating observations into a deduplicated, deterministically hashed manifest
is shared; see :mod:`...common.manifest`. What is Llama's is the schema string,
the exact call sites its observer patches, and the summary label.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...common.manifest import (  # noqa: F401  (re-export)
    NOT_OBSERVED,
    Configuration,
    capture_scope as _capture_scope,
    deduplicate,
    semantic_hash,
    write_manifest,
)
from ...common import manifest as _common

SCHEMA_VERSION = "evograd-llama3-harvest/1"

#: The exact call sites ``harvest.observe`` wraps. Not a description of them --
#: the observer patches these attributes by name, and if one moves in a future
#: Transformers release the harvest fails loudly rather than recording a scope
#: that no longer matches what ran.
FUNCTION_WRAPPERS: tuple[str, ...] = (
    "transformers.models.llama.modeling_llama.apply_rotary_pos_emb",
    "torch.nn.functional.scaled_dot_product_attention",
    "transformers.loss.loss_utils.LOSS_MAPPING['ForCausalLM']",
    "transformers.loss.loss_utils.fixed_cross_entropy",
)

LABEL = "Llama-3 harvest"


def capture_scope(observation, *, function_wrappers: Sequence[str] = FUNCTION_WRAPPERS,
                  not_observed: Sequence[str] = NOT_OBSERVED) -> dict[str, Any]:
    """What the Llama-3 harvest watched, and what it deliberately did not."""
    return _capture_scope(observation, function_wrappers=function_wrappers,
                          not_observed=not_observed)


def build_manifest(observation, **kwargs) -> dict[str, Any]:
    """The Llama-3 manifest, with this workload's schema and patched call sites."""
    kwargs.setdefault("schema_version", SCHEMA_VERSION)
    kwargs.setdefault("function_wrappers", FUNCTION_WRAPPERS)
    return _common.build_manifest(observation, **kwargs)


def summarize(manifest: dict[str, Any], *, top_linear: int = 8,
              label: str = LABEL) -> str:
    """The human-readable summary, named as Llama-3's."""
    return _common.summarize(manifest, top_linear=top_linear, label=label)
