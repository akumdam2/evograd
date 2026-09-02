"""**Model** — what the tier measures, and how to bring your own.

The first of tier 3's three parts:

    tier3_model.py    this file — the workload protocol, and the import path
    tier3_patch.py    kernels, sites, and the two ways to insert one
    tier3_runner.py   building, timing, and reporting

The runner never asks what model it is driving. It asks a
:class:`TrainingWorkload` four questions — how do I build you, what do I feed
you, how do I get a loss, how much work was that — and measures whatever comes
back. A Llama, a HuggingFace checkpoint, a vision model, someone's research
code: if it can answer those, it can be measured.

``bench.tier3_llama`` is the built-in architecture. :class:`ModuleWorkload` is
the other door: hand it a factory for any ``nn.Module``, say how to feed it, and
list which submodules a kernel replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import nn

from evograd.bench.tier3_patch import KernelSet, ModulePatch, patch_modules

# ── what the harness needs from a model ──────────────────────────────────────


class TrainingWorkload(Protocol):
    """Everything the tier needs in order to measure a model it did not define.

    Four questions, and nothing else: how do I build you, what do I feed you,
    how do I get a loss out, and how much work was that. A workload that can
    answer them can be measured here, whatever it is — a Llama, a HuggingFace
    checkpoint, a vision model, someone's research code.

    ``build`` takes the :class:`KernelSet` so a workload can patch by
    construction. A workload that cannot — because it did not write the model —
    ignores it and calls :func:`patch_modules` on the built tree instead; see
    :class:`ModuleWorkload`.
    """

    #: Human-readable, appears in the report.
    name: str
    #: What ``units_per_step`` counts: "tokens", "samples", "images".
    unit_name: str

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
    """

    name: str
    factory: Callable[[], nn.Module]
    make_batch: Callable[[int], Any]
    compute_loss: Callable[[nn.Module, Any], torch.Tensor]
    units: int
    patches: tuple[ModulePatch, ...] = ()
    unit_name: str = "samples"
    #: Filled in by ``build``; reported so a run states what it actually replaced.
    patched_paths: list[str] = field(default_factory=list)

    def units_per_step(self) -> int:
        return self.units

    def build(self, kernels: KernelSet) -> nn.Module:
        model = self.factory()
        self.patched_paths = patch_modules(model, self.patches, kernels)
        return model

    def batch_for(self, *, seed: int) -> Any:
        return self.make_batch(seed)

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        return self.compute_loss(model, batch)

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "module",
            "name": self.name,
            "patch_sites": [p.site for p in self.patches],
            "patched_paths": list(self.patched_paths),
        }


