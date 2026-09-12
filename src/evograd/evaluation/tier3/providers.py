"""Building tier-3 providers that are not candidates.

A tier-3 run compares a candidate against things already known to be correct.
This is where those are built: ``torch.compile`` of a site's own production
spelling, a declaration's reviewed Liger pair, or an evolved program loaded from
a path -- each patched in by the route it would really take.

None of it is about an architecture. A site name, a ``SiteRegistry`` and the
declared task registry are the whole input, which is why this lives beside
:mod:`evograd.evaluation.tier3.patch` rather than inside any workload. It was
written inside Qwen3's ``simple`` module because that is where the simplified
numerics policy needed it first; nothing in it ever knew which model it was
patching, and keeping it there meant Llama-3 could not offer the same providers.

**What a workload has to do to get these.** Declare ``compile_site`` and
``patch_set`` in its :class:`Tier3Adapter` options and call
:func:`compile_and_patch_set_providers` from its ``providers`` hook. Two lines;
the CLI already parses the flags and refuses them by name for any workload that
has not opted in.
"""

from __future__ import annotations

from typing import Any

#: The replacement routes ``--patch-set`` understands, besides a file path.
PATCH_SPEC_KINDS = ("compile", "liger")


def compiled_site_kernel(site: str, registry):
    """The site's own declared production spelling, compiled.

    ``torch.compile`` of the exact function the unpatched model already calls,
    so the arithmetic is the model's and only the lowering changes.
    ``dynamic=False`` specializes on the shape the model presents;
    ``fullgraph=True`` makes a graph break raise instead of silently falling
    back to eager and being reported as a compiled result.
    """
    import torch

    from evograd.benchmark import get_task
    from evograd.opdecl.oracle import resolve_runtime_forward

    reference = resolve_runtime_forward(get_task(registry.require(site).op))
    return torch.compile(reference, dynamic=False, fullgraph=True)


def compiled_kernels(sites, registry):
    """A trusted provider patching exactly ``sites``, each compiled.

    Site-matched by construction: it patches what it was given and nothing
    else, so one site's measurement can never be built from another's drift.
    """
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    kernels = KernelSet(registry=registry)
    for site in sites:
        kernels = patch(
            kernels, site, compiled_site_kernel(site, registry),
            source=KernelSource(site=site, op_name=registry.require(site).op,
                                module=None, origin="trusted_torch_compile"),
        )
    return kernels


def parse_patch_specs(entries) -> dict[str, str]:
    """``["attention=compile", "swiglu_mlp=liger"]`` -> ``{site: spec}``."""
    patches: dict[str, str] = {}
    for entry in entries or ():
        site, _, spec = str(entry).partition("=")
        if not site or not spec:
            raise ValueError(f"--patch wants SITE=compile|liger|PATH, got {entry!r}")
        if site in patches:
            raise ValueError(f"site {site!r} patched twice")
        patches[site] = spec
    return patches


def parse_patch_set_spec(text: str) -> tuple[str, dict[str, str]]:
    """``"name:site=spec,site=spec"`` -> ``(name, patches)``."""
    name, sep, rest = str(text).partition(":")
    if not sep or not name or not rest:
        raise ValueError(f"--patch-set wants NAME:SITE=SPEC[,SITE=SPEC...], got {text!r}")
    return name, parse_patch_specs(rest.split(","))


def kernels_from_patches(patches: dict[str, str], registry, ops=None, *,
                         load_program=None):
    """One kernel set holding several sites' replacements, each by its real route.

    ``compile`` -> ``torch.compile`` of the site's declared runtime_forward
    (origin ``trusted_torch_compile``); ``liger`` -> the declaration's reviewed
    Liger pair through ``kernel_from_pair`` (origin ``baseline:liger``, a bind
    pair wrapper -- *not* Liger's own autograd Function); a path -> the evolved
    program through ``patched_kernels``. Every site is patched exactly once, so
    the resulting patch set is the union and the adapters carry whatever comes
    along.
    """
    import importlib.util
    from pathlib import Path

    from evograd.evaluation.tier3.patch import (
        KernelSet, KernelSource, kernel_from_pair, patch, patched_kernels)
    from evograd.opdecl.baselines import baseline_candidate_module
    from evograd.benchmark import TASKS

    ops = dict(ops or TASKS)
    kernels = KernelSet(registry=registry)
    for site, spec in patches.items():
        decl = registry.require(site)
        if spec == "compile":
            kernels = patch(kernels, site, compiled_site_kernel(site, registry),
                            source=KernelSource(site=site, op_name=decl.op, module=None,
                                                origin="trusted_torch_compile"))
        elif spec == "liger":
            op = ops[decl.op]
            if "liger" not in op.performance_baselines:
                raise ValueError(f"{decl.op} declares no liger baseline")
            module = baseline_candidate_module(op, "liger")
            kernels = patch(kernels, site, kernel_from_pair(op, module),
                            source=KernelSource(site=site, op_name=decl.op, module=module,
                                                origin="baseline:liger"))
        else:
            path = Path(spec)
            if load_program is None:
                s = importlib.util.spec_from_file_location(
                    f"evograd_patch_{path.stem}", path)
                module = importlib.util.module_from_spec(s)
                s.loader.exec_module(module)
            else:
                module = load_program(path)
            single = patched_kernels({site: module}, ops, registry=registry)
            kernels = patch(kernels, site, single.kernel_for(site),
                            source=single.source_for(site))
    return kernels


def compile_and_patch_set_providers(args, registry) -> dict[str, Any]:
    """The ``--compile-site`` and ``--patch-set`` providers, for any workload.

    Both flags mean the same thing for every architecture, so this is written
    once and each adapter decides whether to offer it. ``--compile-site`` gives
    **one provider per site**, leaving every other site native -- that turns a
    blended number into an attribution. ``--patch-set`` gives **one provider
    replacing several sites at once**, which is what a real candidate looks
    like.

    An unknown site name is refused against the registry rather than silently
    patching nothing, and the known names are listed, because a site name is
    the one thing a caller cannot guess from the model's documentation.
    """
    providers: dict[str, Any] = {}

    for site in getattr(args, "compile_site", []) or []:
        if site not in registry:
            raise ValueError(
                f"--compile-site {site!r} is not a site of {registry.name!r}; "
                f"known: {sorted(registry.names)}"
            )
        providers[f"torch_compile_{site}"] = compiled_kernels((site,), registry)

    for entry in getattr(args, "patch_set", []) or []:
        name, patches = parse_patch_set_spec(entry)
        if name in providers or name == "eager":
            raise ValueError(f"--patch-set name {name!r} is already a provider")
        unknown = [s for s in patches if s not in registry]
        if unknown:
            raise ValueError(
                f"--patch-set {name!r}: unknown sites {unknown}; "
                f"known: {sorted(registry.names)}"
            )
        providers[name] = kernels_from_patches(patches, registry)

    return providers


__all__ = [
    "PATCH_SPEC_KINDS",
    "compile_and_patch_set_providers",
    "compiled_kernels",
    "compiled_site_kernel",
    "kernels_from_patches",
    "parse_patch_set_spec",
    "parse_patch_specs",
]
