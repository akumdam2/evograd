"""Qwen3's manifest: its schema, the call sites it patched, and its label.

Aggregating observations into a deduplicated, deterministically hashed manifest
is the same operation for every workload, and it lives in
:mod:`...common.manifest`. Three things are not the same, and they are all that
is here:

* the **schema string**, which is versioned per workload because what a manifest
  records is a claim about one model's boundaries;
* the **function wrappers**, naming the exact call sites the observer patched.
  Which module holds ``apply_rotary_pos_emb`` is a fact about Qwen3 and the
  installed Transformers, and the capture scope must record what was actually
  installed rather than a second copy free to drift from it;
* the **summary label**.
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

#: Bumped when the recorded structure changes; the tests assert it, and a
#: manifest that claims a schema it does not match is refused rather than read.
SCHEMA_VERSION = "evograd-qwen3-harvest/1"

#: The exact call sites ``harvest.observe`` wraps. Not a description of them --
#: the observer patches these attributes by name, and if one moves in a future
#: Transformers release the harvest fails loudly rather than recording a scope
#: that no longer matches what ran.
FUNCTION_WRAPPERS: tuple[str, ...] = (
    "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
    "torch.nn.functional.scaled_dot_product_attention",
    "transformers.loss.loss_utils.LOSS_MAPPING['ForCausalLM']",
    "transformers.loss.loss_utils.fixed_cross_entropy",
)

LABEL = "Qwen3 harvest"


def capture_scope(observation, *, function_wrappers: Sequence[str] = FUNCTION_WRAPPERS,
                  not_observed: Sequence[str] = NOT_OBSERVED) -> dict[str, Any]:
    """What the Qwen3 harvest watched, and what it deliberately did not."""
    return _capture_scope(observation, function_wrappers=function_wrappers,
                          not_observed=not_observed)


def build_manifest(observation, **kwargs) -> dict[str, Any]:
    """The Qwen3 manifest, with this workload's schema and patched call sites."""
    kwargs.setdefault("schema_version", SCHEMA_VERSION)
    kwargs.setdefault("function_wrappers", FUNCTION_WRAPPERS)
    return _common.build_manifest(observation, **kwargs)


def summarize(manifest: dict[str, Any], *, top_linear: int = 8,
              label: str = LABEL) -> str:
    """The human-readable summary, named as Qwen3's."""
    return _common.summarize(manifest, top_linear=top_linear, label=label)
