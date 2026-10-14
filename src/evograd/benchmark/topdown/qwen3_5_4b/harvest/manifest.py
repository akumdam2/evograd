"""Qwen3.5's manifest: its schema, the call sites it patched, and its label.

Aggregation is shared (:mod:`...common.manifest`); the schema string, the
exact call sites the observer wraps and what it deliberately does not watch
are this workload's.
"""

from __future__ import annotations

from typing import Any

from ...common import manifest as _common
from ...common.manifest import (  # noqa: F401  (re-export)
    semantic_hash,
    write_manifest,
)
from .observe import FUNCTIONS, MODELING

SCHEMA_VERSION = "evograd-qwen3_5-harvest/1"

#: The exact call sites the harvest patches, recorded in the capture scope.
FUNCTION_WRAPPERS: tuple[str, ...] = (
    f"{MODELING}.apply_rotary_pos_emb",
    "torch.nn.functional.scaled_dot_product_attention",
    "transformers.loss.loss_utils.LOSS_MAPPING['ForCausalLM']",
    "transformers.loss.loss_utils.fixed_cross_entropy",
    *(boundary.wrapper_name for boundary in FUNCTIONS),
)

NOT_OBSERVED = (
    "the softmax and matmuls inside scaled_dot_product_attention",
    "the residual additions",
    "the embedding lookup and the rotary table construction's arithmetic",
    "the GDN's sigmoid(b), exp(A_log) and the q/k repeat_interleave (the gating's "
    "softplus is the boundary)",
    "view / reshape / transpose / contiguous / split / cat",
    "every backward kernel: the backward pass runs, and nothing observes it",
    "the AdamW step (the harvested step is loss + backward, as every workload's harvest)",
    "all ATen operations below the listed boundaries",
    "CUDA kernels",
)

LABEL = "Qwen3.5 harvest"


def build_manifest(observation, **kwargs) -> dict[str, Any]:
    kwargs.setdefault("schema_version", SCHEMA_VERSION)
    kwargs.setdefault("function_wrappers", FUNCTION_WRAPPERS)
    kwargs.setdefault("not_observed", NOT_OBSERVED)
    return _common.build_manifest(observation, **kwargs)


def summarize(manifest: dict[str, Any], *, top_linear: int = 8, label: str = LABEL) -> str:
    return _common.summarize(manifest, top_linear=top_linear, label=label)


__all__ = ["FUNCTION_WRAPPERS", "LABEL", "NOT_OBSERVED", "SCHEMA_VERSION", "build_manifest",
           "semantic_hash", "summarize", "write_manifest"]
