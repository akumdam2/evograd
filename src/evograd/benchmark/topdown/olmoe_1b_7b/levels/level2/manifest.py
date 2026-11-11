"""The one place OLMoE's three Level-2 sites are named and described.

One decoder layer, in execution order::

    x1      = input_layernorm(h)            (layer 0: native; after it, the last line's site)
    q, k, v = RoPE(q_norm(q_proj(x1)), k_norm(k_proj(x1))), v_proj(x1)
    a       = o_proj(causal_sdpa(q, k, v))                     attention_block (both lines)
    s, x2   = h + a, post_attention_layernorm(h + a)           residual_rmsnorm
    idx, w  = top_k(softmax(router(x2)))                       (native: the router)
    m       = experts(x2, idx, w)                              moe_experts
    h', x1' = s + m, input_layernorm'(s + m)                   residual_rmsnorm
              (the next layer's input norm; model.norm after the last layer)

``attention_block`` and ``moe_experts`` run once per layer. ``residual_rmsnorm``
is every residual add an RMSNorm follows: twice per layer -- the post-attention
add, and the layer's closing add fused with the norm that follows it (the next
layer's ``input_layernorm``, or ``model.norm`` after the last layer). Layer 0's
``input_layernorm`` follows the embedding, not a residual add, so a canonical
step runs 16, 32 and 16 calls. The whole-model call counts are **observed**:
:func:`observed_model_frequency` reads them off the tracked harvest snapshot and
:func:`expected_model_frequency` derives the same number from the depth, which
is what a tiny test model or a Tier-3 count check holds the observation to.

The router stays native. Its ``[T, E]`` GEMM, softmax and top-k are a few
percent of the block and routing is control flow a kernel contract would have to
reproduce exactly; it supplies ``top_k_index`` (non-differentiable) and
``top_k_weights`` (differentiable) to the experts site. ``lm_head_ce`` is not a
site in this step either.
"""

from __future__ import annotations

import importlib
from typing import Any

from ...architecture import BLOCK_KIND, OLMOE_1B_7B, REPRESENTATIVE_LAYERS

#: The workload these sites belong to. Also the snapshot and provenance key.
WORKLOAD = "olmoe_1b_7b"

SITE_ATTENTION_BLOCK = "attention_block"
SITE_RESIDUAL = "residual_rmsnorm"
SITE_MOE = "moe_experts"

#: The three sites, in the order a decoder layer executes them.
SITES: tuple[str, ...] = (SITE_ATTENTION_BLOCK, SITE_RESIDUAL, SITE_MOE)

#: Site -> the task key its contract is registered under. Every key is this
#: model's own: a report row, a candidate program and a calibration file all
#: key off the name, and another architecture's key would make each ambiguous.
SITE_TASKS: dict[str, str] = {site: f"olmoe_{site}" for site in SITES}

#: The sites each block kind contains (one kind), in execution order.
SITES_BY_BLOCK: dict[str, tuple[str, ...]] = {BLOCK_KIND: SITES}

#: How many times one layer forward invokes each of its sites. The residual
#: site's second call is the layer's closing add with the norm after it (the
#: next layer's ``input_layernorm``, or ``model.norm``).
FREQUENCY: dict[str, dict[str, int]] = {
    BLOCK_KIND: {SITE_ATTENTION_BLOCK: 1, SITE_RESIDUAL: 2, SITE_MOE: 1}}

#: Boundaries named by the design and not built in this step, and why.
DEFERRED: dict[str, str] = {
    "router": "the [T, E] GEMM, softmax and top-k that choose the experts: native, "
              "supplies the experts site's routing tensors",
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
    """How often one whole-model step invokes this site: once per layer, the
    residual site twice."""
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


__all__ = [
    "BLOCK_KIND", "COMPONENTS", "DEFERRED", "FREQUENCY", "REPRESENTATIVE_LAYERS", "SITES",
    "SITES_BY_BLOCK", "SITE_ATTENTION_BLOCK", "SITE_MOE", "SITE_RESIDUAL", "SITE_TASKS",
    "TASKS", "UnknownSite", "WORKLOAD", "declarations", "dims_for",
    "expected_model_frequency", "kind_of_site", "observed_model_frequency", "site_for",
    "sites_for_layer", "task_key",
]
