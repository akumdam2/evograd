"""The one place OLMoE's four Level-2 sites are named and described.

One decoder layer, in execution order::

    x1          = input_layernorm(h)                       (native in the layer)
    q, k, v     = RoPE(q_norm(q_proj(x1)), k_norm(k_proj(x1))), v_proj(x1)   qkv_norm_rope
    a           = o_proj(causal_sdpa(q, k, v))                                attention
    s, x2       = h + a, post_attention_layernorm(h + a)                      residual_rmsnorm
    idx, w      = top_k(softmax(router(x2)))               (native: the router)
    m           = experts(x2, idx, w)                                         moe_experts
    h'          = s + m                                    (native in the layer)

Every site runs once per layer; a canonical step runs each 16 times. The
whole-model call counts are **observed**: :func:`observed_model_frequency`
reads them off the tracked harvest snapshot and :func:`expected_model_frequency`
derives the same number from the depth, which is what a tiny test model or a
Tier-3 count check holds the observation to.

The router stays native. Its ``[T, E]`` GEMM, softmax and top-k are a few
percent of the block and routing is control flow a kernel contract would have to
reproduce exactly; it supplies ``top_k_index`` (non-differentiable) and
``top_k_weights`` (differentiable) to the experts site. The cross-layer
residual fusion and ``lm_head_ce`` are not sites in this step either.
"""

from __future__ import annotations

import importlib
from typing import Any

from ...architecture import BLOCK_KIND, OLMOE_1B_7B, REPRESENTATIVE_LAYERS

#: The workload these sites belong to. Also the snapshot and provenance key.
WORKLOAD = "olmoe_1b_7b"

SITE_QKV = "qkv_norm_rope"
SITE_ATTENTION = "attention"
SITE_RESIDUAL = "residual_rmsnorm"
SITE_MOE = "moe_experts"

#: The four sites, in the order a decoder layer executes them.
SITES: tuple[str, ...] = (SITE_QKV, SITE_ATTENTION, SITE_RESIDUAL, SITE_MOE)

#: Site -> the task key its contract is registered under. Every key is this
#: model's own: a report row, a candidate program and a calibration file all
#: key off the name, and another architecture's key would make each ambiguous.
SITE_TASKS: dict[str, str] = {site: f"olmoe_{site}" for site in SITES}

#: The sites each block kind contains (one kind), in execution order.
SITES_BY_BLOCK: dict[str, tuple[str, ...]] = {BLOCK_KIND: SITES}

#: How many times one layer forward invokes each of its sites.
FREQUENCY: dict[str, dict[str, int]] = {BLOCK_KIND: {site: 1 for site in SITES}}

#: Boundaries named by the design and not built in this step, and why.
DEFERRED: dict[str, str] = {
    "router": "the [T, E] GEMM, softmax and top-k that choose the experts: native, "
              "supplies the experts site's routing tensors",
    "lm_head_ce": "lm_head + cross entropy: optional site, deferred to a later step",
    "residual_rmsnorm(mlp_to_next_input)": "cross-layer fusion, deferred as for Qwen3.5",
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
        raise UnknownSite(f"{site!r} is not an OLMoE Level-2 site; known: {list(SITES)}") from None


def site_for(task: str) -> str:
    for site, key in SITE_TASKS.items():
        if key == task:
            return site
    raise UnknownSite(f"no OLMoE site serves task {task!r}")


def kind_of_site(site: str) -> str:
    if site not in SITES:
        raise UnknownSite(site)
    return BLOCK_KIND


def sites_for_layer(layer_index: int, arch=OLMOE_1B_7B) -> tuple[str, ...]:
    from ...architecture import layer_kind

    return SITES_BY_BLOCK[layer_kind(layer_index, arch)]


def expected_model_frequency(site: str, arch=OLMOE_1B_7B) -> int:
    """How often one whole-model step invokes this site: once per layer."""
    if site not in SITES:
        raise UnknownSite(site)
    return int(arch["num_hidden_layers"]) * FREQUENCY[BLOCK_KIND][site]


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
    """A site's declared dims under any :class:`MoeModelConfig` record."""
    return getattr(record, f"{COMPONENTS[site]}_dims")(batch=batch, seq=seq)


def declarations() -> dict[str, Any]:
    """Site -> the declared task, imported from that site's package."""
    return {site: importlib.import_module(f"{__package__}.{site}.task").op for site in SITES}


def captures() -> dict[str, Any]:
    """Site -> its ``capture`` function (``source -> DerivedCall``)."""
    return {site: importlib.import_module(f"{__package__}.{site}.capture").capture
            for site in SITES}


__all__ = [
    "BLOCK_KIND", "COMPONENTS", "DEFERRED", "FREQUENCY", "REPRESENTATIVE_LAYERS", "SITES",
    "SITES_BY_BLOCK", "SITE_ATTENTION", "SITE_MOE", "SITE_QKV", "SITE_RESIDUAL", "SITE_TASKS",
    "TASKS", "UnknownSite", "WORKLOAD", "captures", "declarations", "dims_for",
    "expected_model_frequency", "kind_of_site", "observed_model_frequency", "site_for",
    "sites_for_layer", "task_key",
]
