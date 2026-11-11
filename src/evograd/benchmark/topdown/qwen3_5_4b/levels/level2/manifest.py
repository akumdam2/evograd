"""The one place Qwen3.5's Level-2 sites are named, by block kind.

A Gated DeltaNet block (layer 14) runs ``gdn_in_proj``, ``gated_delta_rule`` and
``gdn_norm_out``; a gated full-attention block (layer 15) runs
``attention_block``, the whole attention module; both run the residual fusion
and the dense SwiGLU MLP. Every site runs once per block forward but the
residual fusion, which runs twice: each of the layer's two residual adds fused
with the (1 + w) RMSNorm that follows it -- the post-attention norm, and the
next layer's input norm (``model.norm`` after the last layer). Layer 0's input
norm follows the embedding, not an add, and is no site.

The whole-model call counts (24 GDN layers, 8 full-attention layers, 32 MLPs,
64 residual fusions) are **observed**: :func:`observed_model_frequency` reads
them off the tracked whole-model harvest snapshot.
:func:`expected_model_frequency` derives the same numbers from the layer types,
which is what a tiny test model or a Tier-3 count check holds the observation
to.

``lm_head_ce`` (lm_head + cross entropy, ~15% of a step's FLOPs) is not a site
in this step.
"""

from __future__ import annotations

import importlib
from typing import Any

from ...architecture import QWEN3_5_4B, REPRESENTATIVE_LAYERS, layer_kind, layer_types

#: The workload these sites belong to. Also the snapshot and provenance key.
WORKLOAD = "qwen3_5_4b"

BLOCK_GDN = "linear_attention"
BLOCK_FULL = "full_attention"
BLOCK_KINDS: tuple[str, ...] = (BLOCK_GDN, BLOCK_FULL)

SITE_GDN_IN = "gdn_in_proj"
SITE_GDR = "gated_delta_rule"
SITE_GDN_OUT = "gdn_norm_out"
SITE_ATTENTION_BLOCK = "attention_block"
SITE_RESIDUAL = "residual_rmsnorm"
SITE_MLP = "swiglu_mlp"

#: All six, in the order a block of either kind executes its own subset.
SITES: tuple[str, ...] = (SITE_GDN_IN, SITE_GDR, SITE_GDN_OUT, SITE_ATTENTION_BLOCK,
                          SITE_RESIDUAL, SITE_MLP)

#: Site -> the task key its contract is registered under.
SITE_TASKS: dict[str, str] = {site: f"qwen3_5_{site}" for site in SITES}

#: The sites each block kind contains, in execution order.
SITES_BY_BLOCK: dict[str, tuple[str, ...]] = {
    BLOCK_GDN: (SITE_GDN_IN, SITE_GDR, SITE_GDN_OUT, SITE_RESIDUAL, SITE_MLP),
    BLOCK_FULL: (SITE_ATTENTION_BLOCK, SITE_RESIDUAL, SITE_MLP),
}

#: How many times one block forward invokes each of its sites: once, but the
#: residual fusion twice (the layer's attention add and its closing MLP add).
FREQUENCY: dict[str, dict[str, int]] = {
    kind: {site: 2 if site == SITE_RESIDUAL else 1 for site in sites}
    for kind, sites in SITES_BY_BLOCK.items()
}

#: Sites named by the design and not built in this step, and why.
DEFERRED: dict[str, str] = {
    "lm_head_ce": "lm_head + cross entropy: optional site, deferred to a later step",
}

#: The ``*_dims`` component each site's canonical case is derived from.
COMPONENTS: dict[str, str] = {site: site for site in SITES}

TASKS: tuple[str, ...] = tuple(SITE_TASKS[site] for site in SITES)


class UnknownSite(KeyError):
    """A site name this workload does not declare."""


def task_key(site: str) -> str:
    try:
        return SITE_TASKS[site]
    except KeyError:
        raise UnknownSite(f"{site!r} is not a Qwen3.5 Level-2 site; known: {list(SITES)}") from None


def site_for(task: str) -> str:
    for site, key in SITE_TASKS.items():
        if key == task:
            return site
    raise UnknownSite(f"no Qwen3.5 site serves task {task!r}")


def block_kind_for_layer(layer_index: int, arch=QWEN3_5_4B) -> str:
    return layer_kind(layer_index, arch)


def sites_for_layer(layer_index: int, arch=QWEN3_5_4B) -> tuple[str, ...]:
    return SITES_BY_BLOCK[block_kind_for_layer(layer_index, arch)]


def kind_of_site(site: str) -> str:
    """The block kind a site's representative invocation is taken from (GDN first)."""
    for kind in BLOCK_KINDS:
        if site in SITES_BY_BLOCK[kind]:
            return kind
    raise UnknownSite(site)


def expected_model_frequency(site: str, arch=QWEN3_5_4B) -> int:
    """How often one whole-model step invokes this site, from the layer types."""
    return sum(FREQUENCY[kind].get(site, 0) for kind in layer_types(arch))


def observed_model_frequency(site: str, snapshot: dict[str, Any] | None = None) -> int:
    """How often the harvested canonical step *did* invoke this site.

    Read from the tracked snapshot (``tasks[<task>].observed``); raises
    :class:`~evograd.benchmark.topdown.UnharvestedWorkload` until a harvest has
    been frozen.
    """
    if snapshot is None:
        from evograd.benchmark.topdown import load_snapshot

        snapshot = load_snapshot(WORKLOAD)
    return int(snapshot["tasks"][task_key(site)]["observed"]["whole_model_invocations"])


def dims_for(site: str, record, *, batch: int, seq: int) -> dict[str, int]:
    """A site's declared dims under any :class:`HybridDenseModelConfig` record."""
    return getattr(record, f"{COMPONENTS[site]}_dims")(batch=batch, seq=seq)


def declarations() -> dict[str, Any]:
    """Site -> the declared task, imported from that site's package."""
    return {site: importlib.import_module(f"{__package__}.{site}.task").op for site in SITES}


__all__ = [
    "BLOCK_FULL", "BLOCK_GDN", "BLOCK_KINDS", "COMPONENTS", "DEFERRED", "FREQUENCY",
    "REPRESENTATIVE_LAYERS", "SITES", "SITES_BY_BLOCK", "SITE_ATTENTION_BLOCK", "SITE_GDN_IN",
    "SITE_GDN_OUT", "SITE_GDR", "SITE_MLP", "SITE_RESIDUAL", "SITE_TASKS", "TASKS",
    "UnknownSite", "WORKLOAD", "block_kind_for_layer", "declarations", "dims_for",
    "expected_model_frequency", "kind_of_site", "observed_model_frequency", "site_for",
    "sites_for_layer", "task_key",
]
