"""The canonical Qwen3-0.6B training step, as a tier-3 workload.

    evograd tier3-bench --model qwen3_0_6b --structural-identity

A top-level class, not a ``ModuleWorkload`` built from closures, for one
reason: tier 3 runs each provider in a killable child process, and a closure
cannot cross that boundary. Everything this workload needs is a handful of
serializable fields; :meth:`Qwen3Workload.to_config` writes them out and
:func:`from_config` rebuilds an identical workload on the other side.

The execution is the canonical one this repository has been measuring since
level 4: Qwen3-0.6B, batch 2, sequence 2048, BF16, CUDA, SDPA, ``use_cache``
off, gradient checkpointing off, ``model.train()``, randomly initialised from
config with no weight or tokenizer download. It is not re-specified here -- it
is ``spec.CANONICAL``, and ``build_model``/``make_inputs`` are the same
functions the level-4 smoke, the harvest and the replay all use.

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

from ...levels.level4.model import build_model, effective_settings, make_inputs
from ...levels.level4.spec import CANONICAL, WorkloadSpec
from .sites import (
    PatchedModel,
    SiteCounters,
    expected_counts,
    patch_model,
    qwen3_sites,
)

#: The name the tier-3 CLI selects this workload by.
MODEL_KEY = "qwen3_0_6b"


def _spec_from_config(config: dict[str, Any]) -> WorkloadSpec:
    overrides = {
        key: config[key]
        for key in ("batch_size", "seq_len", "dtype", "device",
                    "attn_implementation", "seed")
        if key in config
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
    #: Where the numerics calibration lives. ``None`` uses the default path.
    calibration_path: str | None = None

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
        workload's own identity or its weights.
        """
        spec = self.spec.replace(seed=self.data_seed + seed)
        return make_inputs(spec)

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
        return effective_settings(model, self.spec)

    # ── the whole-model gate ─────────────────────────────────────────────

    def site_preflight(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Does each patched site hold at the shapes this model will supply?

        Tier 3's runner already asks this before it reaches the gate. Asking
        again here costs one small run per site and makes the gate complete on
        its own, so a standalone verdict names the same first stage the runner
        would have failed at.
        """
        from evograd.bench.tier3_runner import preflight as run_preflight
        from evograd.ops import OPS

        try:
            run_preflight(kernels, OPS, device=device)
        except Exception as exc:
            return {"ok": False, "sites": list(kernels.patched),
                    "reason": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "sites": list(kernels.patched),
                "checked": "declared correctness grid plus the observed shapes"}

    def model_correctness(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Tier 3 calls this before it times a patched provider.

        The thresholds are a measurement of this machine, stored in a
        calibration artifact and bound to the environment they were taken in. If
        none exists, or one exists from elsewhere, this refuses rather than
        quoting somebody else's noise floor -- an ungated timing is not cheaper
        than no timing, it is worse.
        """
        from .gate import CalibrationUnavailable, check_model_correctness, load_policy

        try:
            policy = load_policy(self.calibration_path)
        except CalibrationUnavailable as exc:
            return {"gate": "qwen3_model_correctness", "ok": False,
                    "reason": str(exc).splitlines()[0], "detail": str(exc)}
        if policy.workload_id != self.spec.workload_id:
            return {
                "gate": "qwen3_model_correctness", "ok": False,
                "reason": (
                    f"the calibration is for {policy.workload_id}, this is "
                    f"{self.spec.workload_id}"
                ),
            }
        preflight = self.site_preflight(kernels, device=device)
        from .gate import summarize

        return summarize(check_model_correctness(
            self, kernels, policy=policy, data_seed=self.data_seed,
            preflight=preflight,
        ))

    @property
    def last_build(self) -> PatchedModel | None:
        """Provenance and invocation counters from the most recent build."""
        return self._last


def _empty_provenance():
    from evograd.bench.tier3_patch import PatchProvenance

    return PatchProvenance(
        method="module_surgery", requested_sites=(), actual_sites=(), paths={}
    )


def from_config(config: dict[str, Any]) -> Qwen3Workload:
    """Rebuild a workload in a child process from its serialized fields."""
    return Qwen3Workload.from_config(config)
