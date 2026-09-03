"""Tier 3 (model): evolved kernels inside a training step.

Tier 2 asks what one operator costs when PyTorch calls it. Tier 3 asks whether
that difference survives into a model -- and the honest answer is often no, for
a reason this tier is built to expose rather than hide. A patched step runs one
Python ``autograd.Function`` callback per patched site: 98 of them in a 32-layer
Llama with RMSNorm, SwiGLU and the loss replaced. So ``cpu_bound_fraction`` is
reported alongside throughput, because without it "the kernels did not help"
and "the GPU was never the bottleneck" look identical in tokens/s.

Nothing is timed until it is known to be correct. Every kernel a provider
patches in goes through the same tier-1 pair gate tiers 1 and 2 use, and every
loss must be a finite scalar; a provider that fails either is recorded as failed
and never measured. A wrong kernel at this tier does not raise -- it returns a
throughput -- which is why the gate is the first thing the runner does.

This module is a facade. The implementation is three parts, each of which can
be read and changed on its own:

    tier3_model.py    **what** is measured -- the ``TrainingWorkload`` protocol,
                      and ``ModuleWorkload`` for bringing your own ``nn.Module``
    tier3_patch.py    **how a kernel gets in** -- ``KernelSet``, the patch sites,
                      ``bind`` wrapping, module surgery, and the identity control
    tier3_runner.py   **how it is measured** -- verify, build, step, time, memory,
                      loss agreement, report

    tier3_llama.py    the built-in architecture, one implementation of part one

The layering is acyclic and worth preserving: ``patch`` imports nothing from
tier 3, ``model`` imports ``patch``, ``runner`` imports both. Anything that
would make ``patch`` reach back into ``model`` or ``runner`` belongs somewhere
else.

Import from here for convenience, or from the parts directly when you only need
one of them.
"""

from __future__ import annotations

from evograd.bench.tier3_model import (  # noqa: F401  (re-export)
    ModuleWorkload,
    TrainingWorkload,
    build_with_provenance,
    site_registry_for,
)
from evograd.bench.tier3_patch import (  # noqa: F401  (re-export)
    LLAMA_SITE_OPS,
    LLAMA_SITES,
    KernelSet,
    KernelSource,
    ModulePatch,
    PatchProvenance,
    Site,
    SiteRegistry,
    by_construction_provenance,
    eager_pair_for,
    identity_control_kernels,
    kernel_from_pair,
    patch,
    patch_modules,
    patched_kernels,
    restrict,
)
from evograd.bench.tier3_runner import (  # noqa: F401  (re-export)
    OPTIMIZER,
    OPTIMIZER_DEFAULTS,
    TIER3_PROTOCOL_VERSION,
    ModelCorrectnessFailure,
    NonFiniteLoss,
    PreflightFailure,
    Tier3Error,
    assemble_report,
    check_loss,
    loss_agreement,
    loss_trajectory,
    make_training_step,
    measure_one,
    model_correctness_check,
    measure_provider,
    measure_step,
    preflight,
    provider_order,
    run_tier3,
    speedup_intervals,
    verification_policy,
)

__all__ = [
    "LLAMA_SITES",
    "LLAMA_SITE_OPS",
    "OPTIMIZER",
    "OPTIMIZER_DEFAULTS",
    "TIER3_PROTOCOL_VERSION",
    "KernelSet",
    "KernelSource",
    "ModulePatch",
    "ModuleWorkload",
    "ModelCorrectnessFailure",
    "NonFiniteLoss",
    "PatchProvenance",
    "PreflightFailure",
    "Site",
    "SiteRegistry",
    "Tier3Error",
    "TrainingWorkload",
    "assemble_report",
    "build_with_provenance",
    "by_construction_provenance",
    "check_loss",
    "eager_pair_for",
    "identity_control_kernels",
    "kernel_from_pair",
    "loss_agreement",
    "loss_trajectory",
    "make_training_step",
    "measure_one",
    "measure_provider",
    "measure_step",
    "model_correctness_check",
    "patch",
    "patch_modules",
    "patched_kernels",
    "preflight",
    "provider_order",
    "restrict",
    "run_tier3",
    "site_registry_for",
    "speedup_intervals",
    "verification_policy",
]
