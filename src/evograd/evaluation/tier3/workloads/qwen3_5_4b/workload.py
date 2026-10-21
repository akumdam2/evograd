"""The canonical Qwen3.5-4B training step, as a tier-3 workload.

    python -m evograd.evaluation.tier3.cli --model qwen3_5_4b --structural-identity

A top-level dataclass of JSON-serializable fields, for the same reason as
Qwen3-0.6B's: tier 3 runs each provider in a killable child process, and
:meth:`Qwen3_5Workload.to_config` / :func:`from_config` rebuild an identical
workload on the other side.

The execution is ``levels.level3.spec.CANONICAL``: the Qwen3.5-4B text model,
batch 1, sequence 2048, BF16, CUDA, SDPA, ``use_cache`` off, no checkpointing,
``model.train()``, random weights through the model's own ``post_init``,
synthetic tokens. The runner adds AdamW (``tier3.runner.OPTIMIZER_DEFAULTS``)
around ``loss.backward()``. Sites are patched on all 32 layers, each by its
type; the counts are held to the layer types (24 / 8 / 32).

**No whole-model gate yet.** :meth:`model_correctness` exists -- the runner
calls it, and a whole-model gate will be bridged in later -- but it reports
the gate as not implemented: ``evaluation_complete: false`` with the reason,
``ok`` so the provider is still timed. Site preflight (declared correctness grid) still runs
before anything is timed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import (
    build_model,
    make_inputs,
    qwen3_5_effective_settings,
)
from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL, WorkloadSpec

from .sites import PatchedModel, SiteCounters, expected_counts, patch_model, qwen3_5_sites

#: The name the tier-3 CLI selects this workload by.
MODEL_KEY = "qwen3_5_4b"

#: Whole-model ``torch.compile`` of the unpatched model: a baseline, not a site
#: provider (Qwen3-0.6B's settings, recorded in every build's provenance).
WHOLE_MODEL_COMPILE_ORIGIN = "whole_model_torch_compile"
WHOLE_MODEL_COMPILE_SITE = "__whole_model__"
WHOLE_MODEL_COMPILE_SETTINGS: dict[str, Any] = {
    "backend": "inductor", "mode": None, "dynamic": False, "fullgraph": False,
}

#: Why the whole-model gate does not run, stated in every provider row.
MODEL_GATE_NOT_IMPLEMENTED = (
    "the Qwen3.5-4B whole-model gate is not implemented yet; only site preflight ran")


def whole_model_compile_kernels(registry):
    """A kernel set that patches nothing and asks for the whole model compiled."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource

    return KernelSet(registry=registry, sources=(KernelSource(
        site=WHOLE_MODEL_COMPILE_SITE, op_name=None, module=None,
        origin=WHOLE_MODEL_COMPILE_ORIGIN),))


def whole_model_compile_requested(kernels) -> bool:
    return any(getattr(s, "origin", None) == WHOLE_MODEL_COMPILE_ORIGIN
               for s in getattr(kernels, "sources", ()))


def _spec_from_config(config: dict[str, Any]) -> WorkloadSpec:
    overrides = {
        key: config[key]
        for key in ("batch_size", "seq_len", "dtype", "device", "attn_implementation", "seed",
                    "weights", "data")
        if key in config and config[key] is not None
    }
    arch = config.get("arch_overrides") or {}
    if arch:
        overrides["arch"] = dict(arch)
    return CANONICAL.replace(**overrides) if overrides else CANONICAL


def model_gate_not_implemented(reason: str = MODEL_GATE_NOT_IMPLEMENTED) -> dict[str, Any]:
    """The verdict row of a provider no whole-model gate judged.

    The field names are :func:`evograd.evaluation.tier3.gate.enforcement.summarize`'s
    (and the runner's own ``_not_checked``), so one reader works for every row:
    ``ok`` -- nothing forbids timing -- with the numerical answer unavailable and
    ``evaluation_complete`` false.
    """
    return {"gate": "qwen3_5_model_gate", "implemented": False, "ok": True, "skipped": True,
            "execution_ok": True, "numerical_ok": None, "numerical_status": "unavailable",
            "numerical_unavailable_at": ["model_gate"], "evaluation_complete": False,
            "evaluation_incomplete_because": [reason], "reason": reason, "findings": []}


@dataclass
class Qwen3_5Workload:
    """Qwen3.5-4B next-token training, one step at a time (tier 3's ``TrainingWorkload``)."""

    batch_size: int = CANONICAL.batch_size
    seq_len: int = CANONICAL.seq_len
    dtype: str = CANONICAL.dtype
    device: str = CANONICAL.device
    attn_implementation: str = CANONICAL.attn_implementation
    seed: int = CANONICAL.seed
    #: Moves the token stream without touching the workload identity.
    data_seed: int = 0
    #: Reduced architectures for tests and debug depths (``--layers``).
    arch_overrides: dict[str, Any] = field(default_factory=dict)
    #: Reserved (random / synthetic only in this step).
    weights: str = "random"
    data: str = "synthetic"

    unit_name = "tokens"

    def __post_init__(self) -> None:
        self.spec = _spec_from_config(self.to_config())
        self.spec.validate()
        self.name = self.spec.workload_id
        self.site_registry = qwen3_5_sites()
        self._last: PatchedModel | None = None

    # ── serialization ────────────────────────────────────────────────────

    def to_config(self) -> dict[str, Any]:
        return {
            "model": MODEL_KEY,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "dtype": self.dtype,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "arch_overrides": dict(self.arch_overrides),
            "weights": self.weights,
            "data": self.data,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Qwen3_5Workload":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in config.items() if k in known})

    # ── the tier-3 protocol ──────────────────────────────────────────────

    @property
    def layer_types(self) -> tuple[str, ...]:
        from evograd.benchmark.topdown.qwen3_5_4b.architecture import layer_types

        return layer_types(self.spec.arch)

    def units_per_step(self) -> int:
        return self.spec.token_count

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        """A fresh model with this provider's sites installed on every layer of
        the matching type. Deterministic: weights are seeded on the CPU."""
        from evograd.evaluation.tier3.patch import PatchProvenance

        model = build_model(self.spec)
        expected = expected_counts(self.layer_types)
        if whole_model_compile_requested(kernels):
            if kernels.patched:
                raise ValueError(
                    "whole-model torch.compile is a baseline of the unpatched model; it cannot "
                    f"be combined with site patches {list(kernels.patched)}")
            model = torch.compile(model, **WHOLE_MODEL_COMPILE_SETTINGS)
            provenance = PatchProvenance(
                method="whole_model_torch_compile", requested_sites=(), actual_sites=(),
                paths={WHOLE_MODEL_COMPILE_SITE: ("model",)})
            self._last = PatchedModel(model=model, provenance=provenance,
                                      counters=SiteCounters(), carrier=None, expected=expected)
            return model, provenance
        if not kernels.patched:
            provenance = PatchProvenance(method="module_surgery", requested_sites=(),
                                         actual_sites=(), paths={})
            self._last = PatchedModel(model=model, provenance=provenance,
                                      counters=SiteCounters(), carrier=None, expected=expected)
            return model, provenance
        provenance, counters, carrier = patch_model(
            model, kernels, expected_layers=self.spec.arch["num_hidden_layers"])
        self._last = PatchedModel(model=model, provenance=provenance, counters=counters,
                                  carrier=carrier, expected=expected)
        return model, provenance

    def batch_for(self, *, seed: int):
        """Deterministic ``(input_ids, labels)``; ``labels = input_ids.clone()``.

        On real text, the first training batch of data order ``data_seed + seed``
        (as for Qwen3-0.6B), so single-step gates and training see the same tokens.
        """
        if self.spec.real_text:
            return self.train_batches(self.data_seed + seed).batch(0)
        spec = self.spec.replace(seed=self.data_seed + seed)
        return make_inputs(spec)

    # ── real text (Qwen3-0.6B's WikiText-2 packing, this model's tokenizer) ──

    def _text(self):
        """Tokenizer and packed splits, built once per workload instance."""
        if not self.spec.real_text:
            raise ValueError(f"{self.spec.workload_id} has synthetic data")
        if getattr(self, "_text_cache", None) is None:
            import os

            from transformers import AutoTokenizer

            from evograd.benchmark.topdown.common.data_paths import hf_cache_dir
            from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import (
                pretrained_snapshot,
            )
            from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata

            snapshot = pretrained_snapshot(self.spec)
            tokenizer = AutoTokenizer.from_pretrained(snapshot)
            cache_dir = hf_cache_dir("qwen3_5_4b")
            train = textdata.pack_split("train", tokenizer, block_len=self.seq_len,
                                        cache_dir=cache_dir)
            validation = textdata.pack_split("validation", tokenizer, block_len=self.seq_len,
                                             cache_dir=cache_dir)
            identity = textdata.data_identity(
                train, validation, block_len=self.seq_len,
                tokenizer_sha256=textdata.file_sha256(os.path.join(snapshot, "tokenizer.json")))
            self._text_cache = {"tokenizer": tokenizer, "train": train, "validation": validation,
                                "identity": identity, "snapshot": snapshot}
        return self._text_cache

    def train_batches(self, seed: int):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata

        return textdata.TextBatches(self._text()["train"], batch_size=self.batch_size,
                                    seed=seed, device=self.device)

    def validation_batches(self):
        """The held-out set in one fixed order -- the same for every provider."""
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata

        return textdata.TextBatches(self._text()["validation"], batch_size=self.batch_size,
                                    seed=0, device=self.device)

    def eager_evaluator(self):
        """A fresh unpatched model: the trusted evaluator for validation NLL."""
        model = build_model(self.spec)
        model.eval()
        return model

    def data_identity(self) -> dict[str, Any]:
        return dict(self._text()["identity"])

    def loss(self, model, batch) -> torch.Tensor:
        input_ids, labels = batch
        return model(input_ids=input_ids, labels=labels, use_cache=False).loss

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "qwen3_5_next_token",
            "name": self.name,
            "workload_id": self.spec.workload_id,
            "workload_hash": self.spec.workload_hash,
            "config_hash": self.spec.config_hash,
            "model": self.spec.model_name,
            "layers": self.spec.arch["num_hidden_layers"],
            "layer_types": {kind: self.layer_types.count(kind)
                            for kind in ("linear_attention", "full_attention")},
            "batch": self.spec.batch_size,
            "tokens": self.spec.seq_len,
            "units_per_step": self.units_per_step(),
            "unit_name": self.unit_name,
            "dtype": self.spec.dtype,
            "attn_implementation": self.spec.attn_implementation,
            "use_cache": self.spec.use_cache,
            "gradient_checkpointing": self.spec.gradient_checkpointing,
            "training": self.spec.training,
            "seed": self.spec.seed,
            "data_seed": self.data_seed,
            "canonical": self.spec.is_canonical,
            "config": self.to_config(),
            "input_checksum": self.input_checksum(),
            "expected_site_counts": expected_counts(self.layer_types),
            "model_gate": {"implemented": False, "reason": MODEL_GATE_NOT_IMPLEMENTED},
        }

    def input_checksum(self, *, seed: int = 0) -> str:
        spec = self.spec.replace(seed=self.data_seed + seed, device="cpu")
        ids, _labels = make_inputs(spec)
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        return f"{digest[:16]}:{tuple(ids.shape)}"

    def runtime_report(self, model) -> dict[str, Any]:
        """What the built model reports (dispatch included), not what was requested."""
        inner = getattr(model, "_orig_mod", model)
        report = qwen3_5_effective_settings(inner, self.spec)
        if inner is not model:
            report["whole_model_torch_compile"] = dict(WHOLE_MODEL_COMPILE_SETTINGS)
        return report

    def site_preflight(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Does each patched site hold on its declared correctness grid?

        No observed shapes are attached (provisional tolerances; see
        ``sites.qwen3_5_sites``), so this is the grid only.
        """
        from evograd.benchmark import TASKS
        from evograd.evaluation.tier3.runner import preflight as run_preflight

        try:
            run_preflight(kernels, TASKS, device=device)
        except Exception as exc:
            return {"ok": False, "sites": list(kernels.patched),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "detail": getattr(exc, "detail", None)}
        return {"ok": True, "sites": list(kernels.patched),
                "checked": "declared correctness grid (no observed shapes: provisional "
                           "tolerances)"}

    def model_correctness(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Not implemented yet: reported, never passed."""
        return model_gate_not_implemented()

    def local_boundary_check(self, kernels, *, layers=None, device: str = "cuda",
                             seed: int = 0) -> dict[str, Any]:
        """Untimed: every live site invocation (in ``layers``, default all) against
        its declaration, on one patched forward + backward. Not a gate."""
        from .boundary import validate_invocations

        model, _ = self.build_patched(kernels)
        types_ = self.layer_types
        chosen = range(len(types_)) if layers is None else layers
        live = expected_counts([types_[i] for i in chosen])
        from .sites import live_sites

        expected = {site: live[site] for site in live_sites(kernels.patched) if live[site]}
        return validate_invocations(model, kernels, self.batch_for(seed=seed), self.loss,
                                    registry=self.site_registry, dtype=self.spec.dtype,
                                    layers=layers, expected=expected)

    @property
    def last_build(self) -> PatchedModel | None:
        return self._last


def from_config(config: dict[str, Any]) -> Qwen3_5Workload:
    """Rebuild a workload in a child process from its serialized fields."""
    return Qwen3_5Workload.from_config(config)


__all__ = ["MODEL_GATE_NOT_IMPLEMENTED", "MODEL_KEY", "Qwen3_5Workload",
           "WHOLE_MODEL_COMPILE_SETTINGS", "from_config", "model_gate_not_implemented",
           "whole_model_compile_kernels"]
