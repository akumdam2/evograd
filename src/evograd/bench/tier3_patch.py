"""**Patcher** -- what a kernel is, and how a model comes to hold one.

The middle of tier 3's three parts:

    tier3_model.py    the architecture, and how to bring your own
    tier3_patch.py    this file -- kernels, sites, and the two ways to insert one
    tier3_runner.py   building, timing, and reporting

A :class:`KernelSet` says which implementation each patchable site reaches for.
There are two ways a model can be made to reach for a different one, and which
applies depends on whether you wrote the model:

* **By construction** -- the model holds a ``KernelSet`` and calls through it, so
  building with a different set *is* the patch. No surgery. This is what the
  built-in Llama workload does.
* **By module surgery** -- :func:`patch_modules` walks a built tree and replaces
  matching submodules, carrying their trained weights across. This is the route
  for a model you did not write, and it is what Liger's
  ``apply_liger_kernel_to_llama`` does.

Either way the set carries its own **provenance**: which sites were requested,
which were actually replaced, and at what module paths. That travels with the
provider rather than being written back onto the workload, because the workload
is built once per provider and the last build would otherwise overwrite what
every earlier one recorded.

This module depends on nothing else in tier 3, which is what keeps the layering
acyclic: the model layer imports from here, the runner imports from both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from evograd.opdecl.activity import OpDecl
from evograd.opdecl.bind import bind
from evograd.opdecl.inputs import as_output_tuple
from evograd.opdecl.oracle import resolve_runtime_forward

# ── the swappable surface ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Site:
    """One patchable place in one model family.

    ``op`` names the declaration a candidate for this site must implement, and
    ``default`` is the production spelling the unpatched model runs. ``preflight``
    carries model-derived correctness workloads the operator's own declared grid
    does not contain -- the shapes this model actually presents, which is where
    a kernel that is correct at 32 rows and wrong at 4096 gets caught.
    """

    name: str
    op: str
    default: Callable
    preflight: tuple = ()


@dataclass(frozen=True)
class SiteRegistry:
    """Which sites a model family has. Owned by a workload, never global.

    There was one module-level ``SITE_OPS`` dict, and it worked exactly as long
    as there was one model. With two it is wrong in both directions: Llama's
    identity control would patch Qwen's sites, and a Qwen candidate would be
    accepted for ``rms_norm`` because the name happens to exist. A registry is
    the smallest thing that makes ownership explicit -- the workload says which
    sites it has, and every operation reads it from the kernel set rather than
    from the module.
    """

    name: str
    sites: tuple[Site, ...]

    def __post_init__(self) -> None:
        names = [site.name for site in self.sites]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate site names in {names}")

    def __contains__(self, site: str) -> bool:
        return any(s.name == site for s in self.sites)

    def __iter__(self):
        return iter(self.names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.sites)

    @property
    def site_ops(self) -> dict[str, str]:
        """``{site: declared operator}`` -- the mapping a report serializes."""
        return {site.name: site.op for site in self.sites}

    def get(self, site: str) -> Site | None:
        for entry in self.sites:
            if entry.name == site:
                return entry
        return None

    def require(self, site: str) -> Site:
        entry = self.get(site)
        if entry is None:
            raise ValueError(
                f"unknown patch site {site!r} for workload {self.name!r}; "
                f"its sites are {sorted(self.names)}"
            )
        return entry

    def defaults(self) -> dict[str, Callable]:
        return {site.name: site.default for site in self.sites}

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_family": self.name,
            "site_ops": self.site_ops,
            "preflight_workloads": {
                site.name: len(site.preflight) for site in self.sites
            },
        }


#: The default registry, which deliberately has no sites.
#:
#: A registry names one model's patchable places, so it belongs to a workload
#: package -- this module is the patcher and must not know any architecture.
#: Defaulting to some model's registry is how a kernel set for one model ends up
#: silently claiming another's sites, so the default here patches nothing and
#: says why when asked to.
NO_SITES = SiteRegistry(name="unspecified", sites=())


@dataclass(frozen=True)
class KernelSource:
    """Where one patched site's kernel came from.

    Carried so the runner can do two things the callable alone cannot support:
    **preflight** it through the tier-1 correctness gate before any of it is
    timed, and **report** what was actually patched in. ``module`` is the
    forward/backward pair as ``lookup_pair`` accepts it -- ``None`` for a plain
    callable that no declaration governs, which is then unverifiable and says
    so rather than being quietly waved through.
    """

    site: str
    op_name: str | None = None
    module: Any = None
    #: "candidate", "baseline:liger", "identity_control", "callable".
    origin: str = "callable"

    @property
    def verifiable(self) -> bool:
        return self.op_name is not None and self.module is not None

    def to_dict(self) -> dict[str, Any]:
        return {"site": self.site, "op": self.op_name, "origin": self.origin}


@dataclass(frozen=True)
class KernelSet:
    """Which implementation the model reaches for at each patchable site.

    Defaults are the declared production spelling, taken from the registry. A
    provider replaces one or more entries; nothing else about the model changes,
    so a difference in the measurement is attributable to the swap.

    Sites are reached by attribute -- ``kernels.rms_norm(x, w, eps)`` -- because
    that is how a model written against a kernel set calls one, and it stays
    true whatever the registry contains. Which sites exist is the registry's
    answer, not this class's: the fields used to be hard-coded Llama site names,
    which made a second model impossible to express.
    """

    registry: SiteRegistry = NO_SITES
    #: Site -> replacement callable. Absent sites use the registry's default.
    overrides: dict[str, Callable] = field(default_factory=dict)
    #: Names of the sites actually replaced, for the report.
    patched: tuple[str, ...] = ()
    #: One entry per replaced site, in the order the sites were patched.
    sources: tuple[KernelSource, ...] = ()

    def __getattr__(self, name: str) -> Callable:
        # Only reached when normal attribute lookup fails, so the declared
        # fields above are never shadowed by a site of the same name.
        registry = self.__dict__.get("registry")
        if registry is not None:
            site = registry.get(name)
            if site is not None:
                return self.__dict__.get("overrides", {}).get(name, site.default)
        raise AttributeError(name)

    def kernel_for(self, site: str) -> Callable:
        """The callable this set reaches for at ``site``. Explicit spelling."""
        entry = self.registry.require(site)
        return self.overrides.get(site, entry.default)

    def source_for(self, site: str) -> KernelSource | None:
        for source in self.sources:
            if source.site == site:
                return source
        return None


@dataclass(frozen=True)
class PatchProvenance:
    """What one provider's build actually replaced.

    Per provider, never on the workload. ``ModuleWorkload`` used to write the
    replaced paths onto itself, so with four providers the report described the
    last build four times -- and every provider looked like it had patched
    whatever the final one did.
    """

    #: "by_construction" (the model calls through the KernelSet) or
    #: "module_surgery" (submodules were swapped in a built tree).
    method: str
    #: The sites the provider asked for.
    requested_sites: tuple[str, ...] = ()
    #: The sites that were actually installed.
    actual_sites: tuple[str, ...] = ()
    #: site -> the module paths replaced. Empty for by-construction patching,
    #: where there is no tree to walk and the site *is* the location.
    paths: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "requested_sites": list(self.requested_sites),
            "actual_sites": list(self.actual_sites),
            "paths": {site: list(paths) for site, paths in self.paths.items()},
            "counts": {site: len(paths) for site, paths in self.paths.items()},
            "modules_replaced": sum(len(p) for p in self.paths.values()),
        }


def _declared_rank(shape: str | None) -> int | None:
    """How many dimensions a declared shape string has. ``None`` if not a tensor."""
    if not shape:
        return None
    inner = shape.strip()[1:-1].strip()
    return len(inner.split(",")) if inner else 0


def _restore_leading(
    value: Any, rank: int | None, leading: tuple[int, ...] | None
) -> Any:
    """Put the model's leading dimensions back on one result.

    Applied per output, independently. A multi-output operator can mix ranks --
    a per-row tensor and a scalar in the same return -- and restoring the wrong
    one is not a shape error that surfaces later, it is a silently reshaped
    result. So each output is decided on its own declared rank.
    """
    if (
        leading is not None
        and torch.is_tensor(value)
        and rank
        and value.dim() == rank
        and value.shape[0] == math.prod(leading)
    ):
        return value.reshape(*leading, *value.shape[1:])
    return value


def kernel_from_pair(op: OpDecl, module) -> Callable:
    """A candidate's forward/backward pair as a callable the model can hold.

    ``bind`` is what makes it differentiable: it routes the saved state through
    ``save_for_backward``, checks the returned gradient count, and places each
    gradient in its declared argument slot.

    Around that sits a rank adapter, because the model and the declaration
    disagree about shape and both are right. A model carries activations as
    ``[batch, tokens, hidden]``; every declaration here is written for rows --
    ``rmsnorm`` takes ``[rows, hidden]``, ``fused_linear_cross_entropy`` takes
    ``target: [rows]``. Eager PyTorch hides the difference by broadcasting over
    leading dimensions, so the unpatched model runs and a patched one fails
    inside someone else's kernel with an unpacking error.

    Flattening the leading dimensions and restoring them afterwards is what a
    real integration does too: HuggingFace flattens before the loss, and Liger's
    patches operate on 2D rows. Doing it here keeps the declaration honest --
    a kernel is measured on exactly the shape it was declared, verified and
    evolved against -- rather than silently requiring every candidate to handle
    a rank its contract never mentioned.

    A multi-output declaration returns an ordered tuple, and every output is
    restored on its own declared rank; a single-output one still returns a bare
    Tensor, so the model sees exactly what it saw before.
    """
    call = bind(op, module)
    ranks = [_declared_rank(getattr(arg, "shape", None)) for arg in op.args]
    output_ranks = tuple(
        _declared_rank(getattr(out, "shape", None)) for out in op.outputs
    )
    multi_output = op.is_multi_output

    def flattened(*args):
        leading: tuple[int, ...] | None = None
        adapted = []
        for value, rank in zip(args, ranks):
            if rank is None or not torch.is_tensor(value) or value.dim() <= rank:
                adapted.append(value)
                continue
            extra = value.dim() - rank + 1
            if leading is None:
                leading = tuple(value.shape[:extra])
            adapted.append(value.reshape(-1, *value.shape[extra:]))

        result = call(*adapted)

        # Normalized through the declaration's own helper, so an arity mismatch
        # is a named contract error rather than a tuple quietly treated as one
        # tensor. Restore is per output: a fused loss returns a scalar, and
        # unflattening that would be nonsense.
        outputs = as_output_tuple(op, result)
        restored = tuple(
            _restore_leading(value, rank, leading)
            for value, rank in zip(outputs, output_ranks)
        )
        return restored if multi_output else restored[0]

    return flattened


# ── providers and the run ────────────────────────────────────────────────────

def patch(
    kernels: KernelSet,
    site: str,
    callable_: Callable,
    *,
    source: KernelSource | None = None,
) -> KernelSet:
    """Replace one site, recording that it was replaced and where it came from.

    The site must belong to *this* kernel set's registry. Naming a site another
    model family has is the error the global namespace could not detect.
    """
    kernels.registry.require(site)
    entry = source or KernelSource(site=site, op_name=kernels.registry.require(site).op)
    if entry.site != site:
        raise ValueError(
            f"source names site {entry.site!r} but is being patched into {site!r}"
        )
    kept = tuple(s for s in kernels.sources if s.site != site)
    return replace(
        kernels,
        overrides={**kernels.overrides, site: callable_},
        patched=tuple(sorted({*kernels.patched, site})),
        sources=tuple(sorted(kept + (entry,), key=lambda s: s.site)),
    )


def restrict(kernels: KernelSet, sites: tuple[str, ...]) -> KernelSet:
    """Keep only the named sites patched; revert the rest to eager.

    Patching three sites at once and reading one number cannot say which site
    the number came from. Restricting to one at a time is how a blended result
    becomes an attribution.
    """
    for site in sites:
        kernels.registry.require(site)
    result = KernelSet(registry=kernels.registry)
    for site in kernels.patched:
        if site in sites:
            result = patch(
                result, site, kernels.kernel_for(site), source=kernels.source_for(site)
            )
    return result


from evograd.pipelines.shared.artifact import (  # noqa: E402
    DEPLOYMENT_ENTRY_ATTR,
    ArtifactError,
    deployment_entry,
    validate_artifact,
)

#: Provenance labels. A report must be able to say which route a candidate took.
DIRECT_DEPLOYMENT = "direct_deployment"
LEGACY_BIND = "legacy_bind"


def patched_kernels(
    candidates: dict[str, Any],
    ops: dict[str, OpDecl],
    *,
    registry: SiteRegistry = NO_SITES,
    origin: str = "candidate",
) -> KernelSet:
    """Build a kernel set from ``{site: candidate module}``.

    Each candidate is bound through ``opdecl.bind``, so the model reaches an
    evolved kernel by exactly the path a deployment would: a
    ``torch.autograd.Function`` built from the declaration, not a hand-wired
    call. The pair module is kept in the set's ``sources`` so the runner can put
    it through the tier-1 correctness gate before timing anything.
    """
    kernels = KernelSet(registry=registry)
    for site, module in candidates.items():
        op_name = registry.require(site).op
        op = ops[op_name]
        entry = deployment_entry(module)
        if entry is not None:
            # The artifact ships its own differentiable callable. Validate it
            # rather than trusting it: an artifact that declares
            # DEPLOYMENT_ENTRY and then fails the contract must be rejected,
            # never quietly demoted to the binder -- a silent fallback would
            # measure a different execution path than the one being reported.
            validate_artifact(op, module)
            kernel, kind = entry, f"{origin}:{DIRECT_DEPLOYMENT}"
        else:
            # Compatibility only: a pair-only candidate predating the artifact
            # contract. Labelled so a report never confuses the two.
            kernel, kind = kernel_from_pair(op, module), f"{origin}:{LEGACY_BIND}"
        kernels = patch(
            kernels,
            site,
            kernel,
            source=KernelSource(
                site=site, op_name=op_name, module=module, origin=kind
            ),
        )
    return kernels


@dataclass(frozen=True)
class ModulePatch:
    """Replace every submodule matching ``match`` with ``build(original, kernel)``.

    The route for a model you did not write. ``match`` identifies the sites --
    usually ``lambda m: type(m).__name__ == "LlamaRMSNorm"`` -- and ``build``
    returns the replacement, which is responsible for carrying the original's
    trained weights across. Getting that wrong produces a model that runs and
    has been silently reinitialized, so the replacement receives the original
    rather than just its shapes.
    """

    site: str
    match: Callable[[nn.Module], bool]
    build: Callable[[nn.Module, Callable], nn.Module]
    #: Further sites this one module also covers. A Qwen3Attention holds both
    #: the QKV/norm/RoPE boundary and the SDPA/output-projection one; replacing
    #: it twice would mean two wrappers and an ordering question, so one patch
    #: declares both and the adapter selects them independently.
    sites: tuple[str, ...] = ()

    @property
    def covered(self) -> tuple[str, ...]:
        return (self.site, *self.sites)


def patch_modules(
    model: nn.Module, patches: tuple[ModulePatch, ...], kernels: KernelSet
) -> PatchProvenance:
    """Swap matching submodules in a built tree. Returns what was replaced.

    Only sites whose kernel the provider actually replaced are touched: an
    eager :class:`KernelSet` leaves the model alone, so the unpatched provider
    is the original model rather than a rebuilt lookalike.

    A requested site that matches **nothing** raises. A patch spec whose
    predicate no longer recognizes the model -- a renamed HuggingFace class, a
    wrapper that moved -- otherwise produces a provider that is labelled
    "patched", is byte-identical to eager, and reports whatever eager reports as
    though the kernel had achieved it.
    """
    requested = tuple(
        site
        for spec in patches
        for site in spec.covered
        if site in kernels.patched
    )
    paths: dict[str, tuple[str, ...]] = {}
    for patch_spec in patches:
        if not any(site in kernels.patched for site in patch_spec.covered):
            continue
        kernel = getattr(kernels, patch_spec.site)
        replaced: list[str] = []
        for path, submodule in list(model.named_modules()):
            if not patch_spec.match(submodule) or not path:
                continue
            parent_path, _, attribute = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attribute, patch_spec.build(submodule, kernel))
            replaced.append(path)
        if not replaced:
            raise ValueError(
                f"patch site {patch_spec.site!r} matched no submodule in "
                f"{type(model).__name__}; the provider would be labelled patched "
                "and be identical to eager"
            )
        for site in patch_spec.covered:
            if site not in kernels.patched:
                continue
            paths[site] = tuple(sorted(set(paths.get(site, ())) | set(replaced)))
    return PatchProvenance(
        method="module_surgery",
        requested_sites=tuple(sorted(set(requested))),
        actual_sites=tuple(sorted(paths)),
        paths=paths,
    )


def by_construction_provenance(kernels: KernelSet) -> PatchProvenance:
    """Provenance for a model that calls through the :class:`KernelSet` itself.

    There is no tree to walk: the sites the set carries *are* what was replaced,
    because the model reaches for them by name at every call.
    """
    return PatchProvenance(
        method="by_construction",
        requested_sites=tuple(kernels.patched),
        actual_sites=tuple(kernels.patched),
        paths={},
    )


def eager_pair_for(op: OpDecl):
    """The declared eager kernel, presented as a forward/backward pair.

    Not a baseline -- a *control*. Patching a site with this changes none of the
    mathematics and all of the plumbing: the same ``F.rms_norm`` now arrives
    through ``bind``, a Python ``autograd.Function``, and the rank adapter,
    exactly as an evolved kernel does.

    Its backward **recomputes the forward** and differentiates it, where a real
    candidate's backward is a kernel. So the gap between this provider and plain
    eager is an *upper bound* on what the patch plumbing costs, not a
    measurement of it: the bound includes one extra forward per patched site
    that no candidate pays. Read it as a ceiling. If it lands near the unpatched
    eager number the plumbing is free and a slowdown belongs to the kernels; if
    a patched provider is slower than this bound, the kernels are the story
    regardless; in between it is inconclusive and ``--sites`` narrows it.
    """
    reference = resolve_runtime_forward(op)
    arg_names = tuple(arg.name for arg in op.args)
    active_names = tuple(arg.name for arg in op.active_args())
    by_grad = {arg.grad_name: arg.name for arg in op.active_args()}

    def forward_with_saved(*args):
        # Save the raw inputs, not a graph. Building the graph here and
        # differentiating it in the backward is the obvious construction and it
        # does not survive contact with `autograd.Function`: forward runs with
        # grad mode disabled, and its returned output is detached, so the graph
        # either never exists or does not reach the backward. Recomputing under
        # `enable_grad` in the backward is the checkpointing pattern and has
        # neither problem.
        with torch.no_grad():
            output = reference(*args)
        return output, tuple(args)

    def backward_from_saved(dout, saved, **_kwargs):
        values = dict(zip(arg_names, saved))
        prepared, leaves = [], []
        for name in arg_names:
            value = values[name]
            if name in active_names and torch.is_tensor(value):
                value = value.detach().requires_grad_(True)
                leaves.append(value)
            prepared.append(value)
        # `enable_grad` because a backward also runs with grad mode off.
        with torch.enable_grad():
            output = reference(*prepared)
        # One upstream gradient per declared output, in declared order -- the
        # control has to differentiate every result a candidate would.
        outputs = as_output_tuple(op, output)
        douts = tuple(dout) if isinstance(dout, (tuple, list)) else (dout,)
        if len(douts) != len(outputs):
            raise ValueError(
                f"{op.name}: got {len(douts)} upstream gradients for "
                f"{len(outputs)} outputs"
            )
        gradients = torch.autograd.grad(outputs, leaves, douts)
        by_name = dict(zip(active_names, gradients))
        return tuple(by_name[by_grad[name]] for name in op.grad_names())

    module = SimpleNamespace(
        forward_with_saved=forward_with_saved,
        backward_from_saved=backward_from_saved,
    )
    module.__name__ = f"<eager pair control for {op.name}>"
    return module


def identity_control_kernels(
    ops: dict[str, OpDecl],
    sites: tuple[str, ...] | None = None,
    *,
    registry: SiteRegistry = NO_SITES,
) -> KernelSet:
    """Every site patched with the eager kernel it already had.

    The tier-3 analogue of ``--identity-control`` at tiers 1 and 2. There the
    control is one provider timed against itself, because those tiers compare
    two providers in symmetric slots. Here the asymmetry is not the slot, it is
    that a patched model routes through machinery an unpatched one never
    touches -- so the control has to be the same mathematics on both sides of
    that machinery.
    """
    kernels = KernelSet(registry=registry)
    for site in sites or registry.names:
        op_name = registry.require(site).op
        op = ops[op_name]
        module = eager_pair_for(op)
        kernels = patch(
            kernels,
            site,
            kernel_from_pair(op, module),
            source=KernelSource(
                site=site, op_name=op_name, module=module, origin="identity_control"
            ),
        )
    return kernels
