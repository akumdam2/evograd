"""The one place Qwen3-0.6B's four Level-2 sites are named and described.

A site is a place in the live model where a fused kernel can be installed. Four
facts follow it everywhere -- what the site is called, which task key its
contract is registered under, how often the canonical step runs it, and which
modules it sits on -- and before this module they were written down in the task
declarations, in the capture code, in the evaluation gate and in the Tier-3
adapter, which is four chances to disagree.

They are written here once, and everything below the model reads them from
here. The shapes, strides, dtypes, frequencies and module paths themselves are
*not* copied into this file: they live in the tracked harvest snapshot, which
is derived from a real canonical step and hashed. This module is the narrow API
over that snapshot, not a second copy of it.
"""

from __future__ import annotations

from typing import Any

#: The workload whose harvest owns these sites.
WORKLOAD = "qwen3_0_6b"

SITE_QKV = "qkv_norm_rope"
SITE_ATTENTION = "attention"
SITE_MLP = "swiglu_mlp"
SITE_RESIDUAL = "residual_rmsnorm"

#: Site -> the task key its contract is registered under.
#:
#: The two differ on purpose. A site names a place in *this* model; a task key
#: names a contract an implementation is written against. ``residual_rmsnorm``
#: is served by ``fused_add_rms_norm`` because that contract is a residual-add
#: plus RMSNorm wherever it occurs, and this model's residual site is the one
#: currently deployed against it. Another architecture may bind its own site to
#: its own task rather than reusing this key.
SITE_TASKS: dict[str, str] = {
    SITE_QKV: "qwen3_qkv_norm_rope",
    SITE_ATTENTION: "qwen3_attention",
    SITE_MLP: "qwen3_swiglu_mlp",
    SITE_RESIDUAL: "fused_add_rms_norm",
}

#: The four sites, in the order a decoder layer executes them.
SITES: tuple[str, ...] = (SITE_QKV, SITE_ATTENTION, SITE_MLP, SITE_RESIDUAL)

#: The task keys this workload's Level-2 contracts are registered under.
TASKS: tuple[str, ...] = tuple(SITE_TASKS[site] for site in SITES)

#: Site -> the module in this package that owns its contract, reference and
#: capture. Import paths are derived from it rather than spelled out again.
SITE_PACKAGES: dict[str, str] = {site: site for site in SITES}


class UnknownSite(KeyError):
    """A site name that this workload does not declare."""


def task_key(site: str) -> str:
    """The registered task key for one site."""
    try:
        return SITE_TASKS[site]
    except KeyError:
        raise UnknownSite(
            f"{site!r} is not a Qwen3-0.6B Level-2 site; known: {list(SITES)}"
        ) from None


def site_for(task: str) -> str:
    """The site a task key is deployed at, inverse of :func:`task_key`."""
    for site, key in SITE_TASKS.items():
        if key == task:
            return site
    raise UnknownSite(f"no Qwen3-0.6B site serves task {task!r}")


def snapshot_entry(site: str) -> dict[str, Any]:
    """This site's frozen harvest record: shapes, strides, dtype, frequency,
    module paths, layer indices, attributes and supporting invocations.

    Read from the tracked snapshot every time rather than cached into module
    state, so a caller can never hold a record from a snapshot that has since
    been replaced.
    """
    from evograd.benchmark.topdown import load_snapshot_task

    return load_snapshot_task(WORKLOAD, task_key(site))


def frequency(site: str) -> int:
    """How many times the canonical step runs this site."""
    return int(snapshot_entry(site)["frequency"])


def module_paths(site: str) -> tuple[str, ...]:
    """The live model modules this site sits on, as the harvest recorded them."""
    return tuple(snapshot_entry(site)["module_paths"])


def config_id(site: str) -> str:
    """The harvested configuration identifier this site's cases derive from."""
    return str(snapshot_entry(site)["config_id"])


def workload_id() -> str:
    """The canonical workload these sites were observed in."""
    from evograd.benchmark.topdown import load_snapshot

    return str(load_snapshot(WORKLOAD)["workload_id"])


def declarations() -> dict[str, Any]:
    """Site -> the declared task, imported from that site's package."""
    import importlib

    resolved = {}
    for site in SITES:
        module = importlib.import_module(f"{__package__}.{SITE_PACKAGES[site]}.task")
        resolved[site] = module.op
    return resolved


__all__ = [
    "SITES",
    "SITE_ATTENTION",
    "SITE_MLP",
    "SITE_PACKAGES",
    "SITE_QKV",
    "SITE_RESIDUAL",
    "SITE_TASKS",
    "TASKS",
    "UnknownSite",
    "WORKLOAD",
    "config_id",
    "declarations",
    "frequency",
    "module_paths",
    "site_for",
    "snapshot_entry",
    "task_key",
    "workload_id",
]
