"""**Model** -- what the tier measures, and how to bring your own.

The first of tier 3's three parts:

    model.py    this file -- the workload protocol, and the import path
    patch.py    kernels, sites, and the two ways to insert one
    runner.py   building, timing, and reporting

The runner never asks what model it is driving. It asks a
:class:`TrainingWorkload` four questions -- how do I build you, what do I feed
you, how do I get a loss, how much work was that -- and measures whatever comes
back. A Llama, a HuggingFace checkpoint, a vision model, someone's research
code: if it can answer those, it can be measured.

Workloads live in ``evaluation/tier3/workloads/<name>/``: each declares its model, its
site registry and its adapters. :class:`ModuleWorkload` is the shortcut for one
you did not write -- hand it a factory for any ``nn.Module``, say how to feed
it, and list which submodules a kernel replaces.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import nn

from evograd.evaluation.tier3.patch import (
    KernelSet,
    ModulePatch,
    PatchProvenance,
    Site,
    SiteRegistry,
    by_construction_provenance,
    patch_modules,
)

# ── what the harness needs from a model ──────────────────────────────────────


class TrainingWorkload(Protocol):
    """Everything the tier needs in order to measure a model it did not define.

    Four questions, and nothing else: how do I build you, what do I feed you,
    how do I get a loss out, and how much work was that. A workload that can
    answer them can be measured here, whatever it is -- a Llama, a HuggingFace
    checkpoint, a vision model, someone's research code.

    ``build`` takes the :class:`KernelSet` so a workload can patch by
    construction. A workload that cannot -- because it did not write the model --
    ignores it and calls :func:`patch_modules` on the built tree instead; see
    :class:`ModuleWorkload`.

    Either may additionally declare a keyword-only ``provider`` parameter, and
    is then told which provider's model it is building. Only a workload whose
    providers are different *trees* rather than different kernels needs it; see
    :func:`accepts_provider`.

    Two things are optional. ``build_patched`` returns the model *and* what it
    replaced, and a workload that does module surgery should implement it so its
    provenance travels with the provider instead of being stored on the workload
    -- see :func:`build_with_provenance`. ``loss_delta_threshold`` declares how
    far this workload's loss trajectory may move before the divergence is a
    failure rather than a diagnostic; without one, the runner reports the deltas
    and gates on lower-tier correctness and finiteness instead of inventing a
    number.
    """

    #: Human-readable, appears in the report.
    name: str
    #: What ``units_per_step`` counts: "tokens", "samples", "images".
    unit_name: str
    #: Which sites this model has, and which declared operator each one is.
    #: Owned by the workload: two model families do not share a namespace, and
    #: a site name existing somewhere is not a reason to accept a candidate for
    #: it here. See :func:`site_registry_for`.
    site_registry: SiteRegistry

    def units_per_step(self) -> int:
        """Work per step, for throughput. Tokens for a language model."""

    def build(self, kernels: "KernelSet") -> nn.Module:
        """A fresh model. Must be deterministic: providers are compared on it."""

    def batch_for(self, *, seed: int) -> Any:
        """One batch. Seeded, so every provider sees the same sequence."""

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """A scalar to call ``.backward()`` on."""

    def describe(self) -> dict[str, Any]:
        """Whatever identifies this workload in the report."""


class VerifiableWorkload(TrainingWorkload, Protocol):
    """What the verification gates additionally need, on real training data.

    See :mod:`evograd.verification.integrations`. Every batch is real data
    in a fixed *data order*: an integer naming one deterministic sequence of
    batches (calibration and acceptance use different orders).

    * ``train_batches(order)`` -- the sequence: ``.batch(i)`` for any ``i``
      (epochs reshuffled deterministically), ``.describe()``;
    * ``batch_for(seed=order)`` -- ``train_batches(order).batch(0)``;
    * ``data_identity()`` -- pins of the data (dataset, revision, file hashes,
      preprocessing), independent of the order.

    A next-token language model needs nothing more: the gates read its
    logits and token-mean NLL. Any other workload also declares

    * ``primary_output(model, batch) -> (loss, primary)`` -- the scalar
      ``loss(model, batch)`` returns, bit for bit, and one output beside it;
    * ``primary_output_name`` and ``objective_definition`` (bound into policies);
    * ``expected_site_calls() -> {site: calls}`` per forward and backward
      (with activation checkpointing off, as verification requires);
    * ``data_recipe() -> dict`` -- batch geometry for the Gate 3 binding.
    """

    def train_batches(self, order: int) -> Any:
        """The batch sequence of one data order."""

    def data_identity(self) -> dict[str, Any]:
        """Pins of the real data."""


def site_registry_for(workload: TrainingWorkload) -> SiteRegistry:
    """The registry a workload owns, refusing to guess when it declares none.

    Guessing would mean defaulting to Llama's three sites, which is exactly the
    global-namespace assumption this replaced: a workload that forgot to declare
    its sites would silently be measured against another model's.
    """
    registry = getattr(workload, "site_registry", None)
    if registry is None:
        raise ValueError(
            f"{getattr(workload, 'name', workload)!r}: the workload does not "
            "declare a site_registry, so tier 3 cannot say which sites it has "
            "or which operator each one implements. Set `site_registry` to a "
            "SiteRegistry; a workload package owns its own."
        )
    return registry


def accepts_provider(builder: Callable) -> bool:
    """Whether a workload's builder wants to be told which provider it is.

    Most do not. A workload whose providers differ only in their kernels learns
    everything it needs from the :class:`KernelSet`, and asking it to carry a
    name it never reads would be noise in every signature.

    Some cannot work that way. A workload whose providers are different
    *trees* -- MegaFold builds its fused Triton kernels straight into the module
    tree, so "which implementation" is decided at construction and not by any
    kernel -- has no way to tell two providers apart from an identical kernel
    set. Those declare a ``provider`` parameter and are handed the name.

    Asked of the builder rather than configured, so adding a workload of the
    second kind is a parameter in that workload and nothing anywhere else.
    """
    try:
        parameters = inspect.signature(builder).parameters
    except (TypeError, ValueError):  # a builtin or C callable: assume not
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "provider" in parameters


def build_with_provenance(
    workload: TrainingWorkload, kernels: KernelSet, *, provider: str | None = None
) -> tuple[nn.Module, PatchProvenance]:
    """Build one provider's model and record what that build actually patched.

    Provenance is a property of the *build*, not of the workload. The workload
    is built once per provider, so anything written back onto it describes only
    whichever provider ran last -- which is how a report ends up claiming that
    the eager provider replaced the same 98 modules the candidate did.

    A workload that swaps submodules implements ``build_patched`` and hands both
    back. One that patches by construction does not need to: the sites the
    :class:`KernelSet` carries are what it replaced, because the model reaches
    for them by name at every call.

    ``provider`` is the name the report will file this build under, passed
    through to a workload that declares it (see :func:`accepts_provider`). It is
    passed explicitly rather than smuggled inside the kernel set: a provider
    that differs by construction used to have to encode itself in
    ``overrides`` under a key chosen not to collide with a site name, which put
    a workload's private convention inside a shared type.
    """
    build_patched = getattr(workload, "build_patched", None)
    if callable(build_patched):
        if accepts_provider(build_patched):
            return build_patched(kernels, provider=provider)
        return build_patched(kernels)
    if accepts_provider(workload.build):
        return workload.build(kernels, provider=provider), by_construction_provenance(
            kernels
        )
    return workload.build(kernels), by_construction_provenance(kernels)


@dataclass
class ModuleWorkload:
    """Measure a model you did not write.

    Supply a factory, a batch maker, a loss, and the module patches that say
    which submodules a kernel replaces. Everything else the harness handles.

        ModuleWorkload(
            name="llama-3-8b-hf",
            factory=lambda: LlamaForCausalLM(config).cuda(),
            make_batch=lambda seed: tokens_and_labels(seed),
            compute_loss=lambda model, b: model(**b).loss,
            units=batch * seq_len,
            patches=(ModulePatch("rms_norm", is_llama_rms_norm, replace_norm),),
        )

    A ``ModuleWorkload`` is not picklable in general -- the factory and loss are
    closures -- so the CLI's per-provider process isolation does not apply to
    it. Driving one means calling :func:`~evograd.evaluation.tier3.runner.run_tier3`
    in-process, where a kernel that hangs or corrupts the CUDA context takes the
    whole run with it.
    """

    name: str
    factory: Callable[[], nn.Module]
    make_batch: Callable[[int], Any]
    compute_loss: Callable[[nn.Module, Any], torch.Tensor]
    units: int
    patches: tuple[ModulePatch, ...] = ()
    unit_name: str = "samples"
    #: The sites this model has. Defaults to one derived from ``patches``, whose
    #: operator names must then be supplied per patch; pass a real registry when
    #: the sites correspond to declared operators a candidate can be verified
    #: against.
    registry: SiteRegistry | None = None
    #: Optional: how far this workload's loss trajectory may move from the
    #: reference before it is a failure. ``None`` leaves it a diagnostic.
    loss_delta_threshold: float | None = None

    @property
    def site_registry(self) -> SiteRegistry:
        if self.registry is not None:
            return self.registry
        # A model with patches but no declared operators still needs a namespace
        # of its own; it just has nothing preflight can verify, which
        # `preflight` reports as unverifiable rather than passing.
        return SiteRegistry(
            name=self.name,
            sites=tuple(
                Site(patch.site, None, _unpatched_placeholder)
                for patch in _unique_sites(self.patches)
            ),
        )

    def units_per_step(self) -> int:
        return self.units

    def build(self, kernels: KernelSet) -> nn.Module:
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels: KernelSet) -> tuple[nn.Module, PatchProvenance]:
        """The model, and what this build replaced -- returned, never stored."""
        model = self.factory()
        return model, patch_modules(model, self.patches, kernels)

    def batch_for(self, *, seed: int) -> Any:
        return self.make_batch(seed)

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        return self.compute_loss(model, batch)

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "module",
            "name": self.name,
            "patch_sites": [p.site for p in self.patches],
        }


def _unique_sites(patches: tuple[ModulePatch, ...]) -> tuple[ModulePatch, ...]:
    seen, out = set(), []
    for patch in patches:
        if patch.site not in seen:
            seen.add(patch.site)
            out.append(patch)
    return tuple(out)


def _unpatched_placeholder(*args, **kwargs):
    """The default for a surgery-only site: never called.

    A by-construction model calls through its kernel set, so every site has a
    real default. A surgery model does not -- the eager provider is the original
    tree, untouched -- so the default exists only to complete the registry.
    """
    raise RuntimeError(
        "a module-surgery site has no by-construction default; the unpatched "
        "provider is the original model"
    )
