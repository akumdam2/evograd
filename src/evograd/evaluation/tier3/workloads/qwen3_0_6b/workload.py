"""The canonical Qwen3-0.6B training step, as a tier-3 workload.

A top-level class, not a ``ModuleWorkload`` built from closures, for one
reason: tier 3 runs each provider in a killable child process, and a closure
cannot cross that boundary. Everything this workload needs is a handful of
serializable fields; :meth:`Qwen3Workload.to_config` writes them out and
:func:`from_config` rebuilds an identical workload on the other side.

The execution is the canonical one this repository has been measuring since
level 3: Qwen3-0.6B, batch 2, sequence 2048, BF16, CUDA, SDPA, ``use_cache``
off, gradient checkpointing off, ``model.train()``, randomly initialised from
config with no weight or tokenizer download. It is not re-specified here -- it
is ``spec.CANONICAL``, and ``build_model``/``make_inputs`` are the same
functions the level-3 smoke, the harvest and the replay all use.

One thing is new: a **data seed** separate from the workload's own. The tier
needs a fresh batch per loss-trajectory step, and drawing those from the
workload seed would change the canonical identity. So the model, the weights
and the workload hash come from ``spec.seed``, and only the token stream moves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import (
    build_model,
    effective_settings,
    make_inputs,
)
from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import (
    CANONICAL,
    WorkloadSpec,
)
from .sites import (
    PatchedModel,
    SiteCounters,
    expected_counts,
    patch_model,
    qwen3_sites,
)

#: The name the tier-3 CLI selects this workload by.
MODEL_KEY = "qwen3_0_6b"

#: Whole-model ``torch.compile`` of the unpatched model: a *baseline*, not a
#: site provider. It replaces no site, so part A has nothing to validate and the
#: runner's site gates are skipped for it; B/C are measured against eager by
#: ``audit`` scripts rather than gated. The settings are recorded in the
#: provenance of every build so a report can say exactly what was compiled.
WHOLE_MODEL_COMPILE_ORIGIN = "whole_model_torch_compile"
WHOLE_MODEL_COMPILE_SITE = "__whole_model__"
WHOLE_MODEL_COMPILE_SETTINGS: dict[str, Any] = {
    "backend": "inductor", "mode": None, "dynamic": False, "fullgraph": False,
}


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
        for key in ("batch_size", "seq_len", "dtype", "device",
                    "attn_implementation", "seed", "weights", "data")
        if key in config and config[key] is not None
    }
    arch = config.get("arch_overrides") or {}
    if arch:
        overrides["arch"] = dict(arch)
    return CANONICAL.replace(**overrides) if overrides else CANONICAL


@dataclass
class Qwen3Workload:
    """Qwen3-0.6B next-token training, one step at a time.

    Implements tier 3's ``TrainingWorkload`` protocol and owns
    :data:`QWEN3_SITES`. Llama's registry is untouched: the two share no site
    name, and neither one's identity control, baseline discovery or report can
    reach the other's sites.
    """

    #: Every field here is JSON-serializable, and together they reconstruct the
    #: workload exactly. Nothing else is state.
    batch_size: int = CANONICAL.batch_size
    seq_len: int = CANONICAL.seq_len
    dtype: str = CANONICAL.dtype
    device: str = CANONICAL.device
    attn_implementation: str = CANONICAL.attn_implementation
    seed: int = CANONICAL.seed
    #: Moves the token stream without touching the workload identity.
    data_seed: int = 0
    #: Reduced architectures for tests. Empty means the published Qwen3-0.6B.
    arch_overrides: dict[str, Any] = field(default_factory=dict)
    #: Inputs of the removed Tier-3 whole-model gate. Read by nothing; kept
    #: because ``describe()`` -- and through it every Gate 2 policy identity --
    #: records the config, including these fields.
    calibration_path: str | None = None
    simple_calibration_path: str | None = None
    #: Parameter and token provenance. Defaults reproduce the synthetic canonical
    #: workload byte-for-byte; a pinned checkpoint and real text give the
    #: workload a distinct identity (WorkloadSpec.weights / .data) that no
    #: synthetic calibration can bind to.
    weights: str = "random"
    data: str = "synthetic"
    #: Also inputs of the removed whole-model gate, kept for the same reason.
    protocol4_calibration_path: str | None = None
    protocol4_verdict_path: str | None = None
    protocol4_diagnostic_timing: bool = False

    unit_name = "tokens"

    def __post_init__(self) -> None:
        self.spec = _spec_from_config(self.to_config())
        self.spec.validate()
        if self.spec.use_cache:  # pragma: no cover - validate() already refuses
            raise ValueError("cache-enabled execution is not a training workload")
        self.name = self.spec.workload_id
        self.site_registry = qwen3_sites()
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
            "calibration_path": self.calibration_path,
            "simple_calibration_path": self.simple_calibration_path,
            "weights": self.weights,
            "data": self.data,
            "protocol4_calibration_path": self.protocol4_calibration_path,
            "protocol4_verdict_path": self.protocol4_verdict_path,
            "protocol4_diagnostic_timing": self.protocol4_diagnostic_timing,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Qwen3Workload":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in config.items() if k in known})

    # ── the tier-3 protocol ──────────────────────────────────────────────

    def units_per_step(self) -> int:
        return self.spec.token_count

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        """A fresh model with this provider's sites installed.

        Deterministic: ``build_model`` seeds on CPU and moves, so the weights do
        not depend on the device's RNG stream, and nothing is downloaded. The
        patching mutates ``forward`` on the modules this build created and
        touches no parameter, so rebuilding is how a run is reverted.
        """
        model = build_model(self.spec)
        if whole_model_compile_requested(kernels):
            if kernels.patched:
                raise ValueError(
                    "whole-model torch.compile is a baseline of the unpatched model; "
                    f"it cannot be combined with site patches {list(kernels.patched)}")
            from evograd.evaluation.tier3.patch import PatchProvenance

            model = torch.compile(model, **WHOLE_MODEL_COMPILE_SETTINGS)
            provenance = PatchProvenance(
                method="whole_model_torch_compile", requested_sites=(), actual_sites=(),
                paths={WHOLE_MODEL_COMPILE_SITE: ("model",)})
            self._last = PatchedModel(
                model=model, provenance=provenance, counters=SiteCounters(), carrier=None,
                expected=expected_counts(self.spec.arch["num_hidden_layers"]))
            return model, provenance
        if not kernels.patched:
            self._last = PatchedModel(
                model=model,
                provenance=_empty_provenance(),
                counters=SiteCounters(),
                carrier=None,
                expected=expected_counts(len(model.model.layers)),
            )
            return model, self._last.provenance
        provenance, counters, carrier = patch_model(
            model, kernels,
            expected_layers=self.spec.arch["num_hidden_layers"],
        )
        self._last = PatchedModel(
            model=model, provenance=provenance, counters=counters,
            carrier=carrier, expected=expected_counts(len(model.model.layers)),
        )
        return model, provenance

    def batch_for(self, *, seed: int):
        """Deterministic ``(input_ids, labels)``; ``labels = input_ids.clone()``.

        The tier passes a per-step seed; it is combined with this workload's
        ``data_seed`` so a run can move the token stream without changing the
        workload's own identity or its weights. On real text the batch is the
        *first training batch* of that data order -- the batch step 1 of a
        training run would see -- so the single-step gates and the training run
        look at the same tokens.
        """
        if self.spec.real_text:
            return self.train_batches(self.data_seed + seed).batch(0)
        spec = self.spec.replace(seed=self.data_seed + seed)
        return make_inputs(spec)

    # ── real text ─────────────────────────────────────────────────────────

    def _text(self):
        """Tokenizer and packed splits, built once per workload instance."""
        if not self.spec.real_text:
            raise ValueError(f"{self.spec.workload_id} has synthetic data")
        cached = getattr(self, "_text_cache", None)
        if cached is None:
            import os

            from transformers import AutoTokenizer

            from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import (
                pretrained_snapshot,
            )
            from . import textdata

            snapshot = pretrained_snapshot(self.spec)
            tokenizer = AutoTokenizer.from_pretrained(snapshot)
            cache_dir = textdata.hf_cache_dir()
            train = textdata.pack_split("train", tokenizer, block_len=self.seq_len,
                                        cache_dir=cache_dir)
            validation = textdata.pack_split("validation", tokenizer,
                                             block_len=self.seq_len, cache_dir=cache_dir)
            identity = textdata.data_identity(
                train, validation, block_len=self.seq_len,
                tokenizer_sha256=textdata.file_sha256(
                    os.path.join(snapshot, "tokenizer.json")))
            cached = {"tokenizer": tokenizer, "train": train, "validation": validation,
                      "identity": identity, "snapshot": snapshot}
            object.__setattr__(self, "_text_cache", cached)
        return cached

    def train_batches(self, seed: int):
        from . import textdata

        return textdata.TextBatches(self._text()["train"], batch_size=self.batch_size,
                                    seed=seed, device=self.device)

    def validation_batches(self):
        """The held-out set in one fixed order -- the same for every provider."""
        from . import textdata

        return textdata.TextBatches(self._text()["validation"], batch_size=self.batch_size,
                                    seed=0, device=self.device)

    def eager_evaluator(self):
        """A fresh unpatched model: the trusted evaluator for validation NLL."""
        model = build_model(self.spec)
        model.eval()
        return model

    def data_identity(self) -> dict[str, Any]:
        return dict(self._text()["identity"])

    def data_identity_digest(self) -> str:
        from . import textdata

        return textdata.identity_digest(self.data_identity())

    def loss(self, model, batch) -> torch.Tensor:
        input_ids, labels = batch
        # `use_cache=False` at the call site as well as on the config: the
        # forward argument is what actually decides.
        return model(input_ids=input_ids, labels=labels, use_cache=False).loss

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "qwen3_next_token",
            "name": self.name,
            "workload_id": self.spec.workload_id,
            "workload_hash": self.spec.workload_hash,
            "config_hash": self.spec.config_hash,
            "model": self.spec.model_name,
            "layers": self.spec.arch["num_hidden_layers"],
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
            "expected_site_counts": expected_counts(
                self.spec.arch["num_hidden_layers"]
            ),
        }

    # ── what a report should be able to state ────────────────────────────

    def input_checksum(self, *, seed: int = 0) -> str:
        """A cheap fingerprint of the token stream, so a report can prove it.

        Computed on CPU from the ids alone; it identifies the batch without
        storing it, which is the whole point -- a run that quietly changed its
        data would otherwise look identical in the report.
        """
        spec = self.spec.replace(seed=self.data_seed + seed, device="cpu")
        ids, _labels = make_inputs(spec)
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        return f"{digest[:16]}:{tuple(ids.shape)}"

    def runtime_report(self, model) -> dict[str, Any]:
        """What the built model actually reports, as distinct from the request."""
        inner = getattr(model, "_orig_mod", model)  # a torch.compile'd model wraps the original
        report = effective_settings(inner, self.spec)
        if inner is not model:
            report["whole_model_torch_compile"] = dict(WHOLE_MODEL_COMPILE_SETTINGS)
        return report

    @property
    def last_build(self) -> PatchedModel | None:
        """Provenance and invocation counters from the most recent build."""
        return self._last


def _empty_provenance():
    from evograd.evaluation.tier3.patch import PatchProvenance

    return PatchProvenance(
        method="module_surgery", requested_sites=(), actual_sites=(), paths={}
    )


def from_config(config: dict[str, Any]) -> Qwen3Workload:
    """Rebuild a workload in a child process from its serialized fields."""
    return Qwen3Workload.from_config(config)
