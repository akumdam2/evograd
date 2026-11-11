"""The one place Llama-3.2-1B's three Level-2 sites are named and described.

A site is a place in the live model where a fused kernel can be installed. Four
facts follow it everywhere -- what the site is called, which task key its
contract is registered under, how often the canonical step runs it, and which
modules it sits on -- and they are written here once.

Every task key here is this model's own. The MLP and residual boundaries
compute the same mathematics Qwen3-0.6B does, and the implementation is shared
(see ``benchmark.topdown.common.level2_references``), but the *task* is not: a
report row, a candidate program and a calibration file all key off the name,
and two architectures sharing one key would make each of them ambiguous about
which model's widths produced it.
"""

from __future__ import annotations

# The workload pins Llama-3.2-1B. `opdecl.models` also carries LLAMA_3_8B,
# a different model.
from evograd.opdecl.models import LLAMA_3_2_1B

#: The workload that owns these sites.
WORKLOAD = "llama_3_2_1b"

SITE_ATTENTION = "attention_block"
SITE_MLP = "swiglu_mlp"
SITE_RESIDUAL = "residual_rmsnorm"

#: Site -> the task key its contract is registered under. The attention site is
#: the whole ``LlamaAttention`` module (projections, RoPE, SDPA and o_proj).
SITE_TASKS: dict[str, str] = {
    SITE_ATTENTION: "llama3_attention_block",
    SITE_MLP: "llama3_swiglu_mlp",
    SITE_RESIDUAL: "llama3_residual_rmsnorm",
}

#: The three sites, in the order a decoder layer executes them.
SITES: tuple[str, ...] = (SITE_ATTENTION, SITE_MLP, SITE_RESIDUAL)

#: The task keys this workload's Level-2 contracts are registered under.
TASKS: tuple[str, ...] = tuple(SITE_TASKS[site] for site in SITES)

#: Site -> the package in this directory that owns its contract and reference.
#: Import paths are derived from it rather than spelled out again.
SITE_PACKAGES: dict[str, str] = {site: site for site in SITES}

#: How many times the canonical step runs each site. Derived from the frozen
#: architecture, not counted by hand and not read from a snapshot.
FREQUENCY: dict[str, int] = {
    SITE_ATTENTION: LLAMA_3_2_1B.layers,
    SITE_MLP: LLAMA_3_2_1B.layers,
    SITE_RESIDUAL: LLAMA_3_2_1B.residual_rmsnorm_fusion_sites()["total"],
}


class UnknownSite(KeyError):
    """A site name that this workload does not declare."""


def task_key(site: str) -> str:
    """The registered task key for one site."""
    try:
        return SITE_TASKS[site]
    except KeyError:
        raise UnknownSite(
            f"{site!r} is not a Llama-3.2-1B Level-2 site; known: {list(SITES)}"
        ) from None


def site_for(task: str) -> str:
    """The site a task key is deployed at, inverse of :func:`task_key`."""
    for site, key in SITE_TASKS.items():
        if key == task:
            return site
    raise UnknownSite(f"no Llama-3.2-1B site serves task {task!r}")


def frequency(site: str) -> int:
    """How many times the canonical step runs this site."""
    try:
        return FREQUENCY[site]
    except KeyError:
        raise UnknownSite(f"{site!r} is not a Llama-3.2-1B Level-2 site") from None


def declarations() -> dict:
    """Site -> the declared task, imported from that site's package."""
    import importlib

    return {
        site: importlib.import_module(f"{__package__}.{SITE_PACKAGES[site]}.task").op
        for site in SITES
    }


__all__ = [
    "FREQUENCY", "SITES", "SITE_ATTENTION", "SITE_MLP", "SITE_PACKAGES",
    "SITE_RESIDUAL", "SITE_TASKS", "TASKS", "UnknownSite",
    "WORKLOAD", "declarations", "frequency", "site_for", "task_key",
]
