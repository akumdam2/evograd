"""Tier 3 (model): evolved kernels inside a training step.

Tier 2 asks what one operator costs when PyTorch calls it. Tier 3 asks whether
that difference survives into a model — and the honest answer is often no, for
a reason this tier is built to expose rather than hide. A patched step runs one
Python ``autograd.Function`` callback per patched site: 98 of them in a 32-layer
Llama with RMSNorm, SwiGLU and the loss replaced. So ``cpu_bound_fraction`` is
reported alongside throughput, because without it "the kernels did not help"
and "the GPU was never the bottleneck" look identical in tokens/s.

This module is a facade. The implementation is three parts, each of which can
be read and changed on its own:

    tier3_model.py    **what** is measured — the ``TrainingWorkload`` protocol,
                      and ``ModuleWorkload`` for bringing your own ``nn.Module``
    tier3_patch.py    **how a kernel gets in** — ``KernelSet``, the patch sites,
                      ``bind`` wrapping, module surgery, and the identity control
    tier3_runner.py   **how it is measured** — build, step, time, memory, loss
                      agreement, report

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
)
from evograd.bench.tier3_patch import (  # noqa: F401  (re-export)
    SITE_OPS,
    KernelSet,
    ModulePatch,
    eager_pair_for,
    identity_control_kernels,
    kernel_from_pair,
    patch,
    patch_modules,
    patched_kernels,
    restrict,
)
from evograd.bench.tier3_runner import (  # noqa: F401  (re-export)
    TIER3_PROTOCOL_VERSION,
    loss_agreement,
    loss_trajectory,
    make_training_step,
    measure_provider,
    measure_step,
    run_tier3,
)

__all__ = [
    "SITE_OPS",
    "TIER3_PROTOCOL_VERSION",
    "KernelSet",
    "ModulePatch",
    "ModuleWorkload",
    "TrainingWorkload",
    "eager_pair_for",
    "identity_control_kernels",
    "kernel_from_pair",
    "loss_agreement",
    "loss_trajectory",
    "make_training_step",
    "measure_provider",
    "measure_step",
    "patch",
    "patch_modules",
    "patched_kernels",
    "restrict",
    "run_tier3",
]
