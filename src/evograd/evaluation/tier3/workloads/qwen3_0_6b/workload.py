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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import (
    build_model,
    effective_settings,
    make_inputs,
)
from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import (
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
    #: Where the numerics calibration lives. ``None`` uses the default path.
    calibration_path: str | None = None
    #: The compile-anchored simplified policy (schema evograd-qwen3-t3-numerics/3).
    #: When set, it *decides* the model-level stage and the detailed 54-group
    #: envelope is recorded beside it as a diagnostic. Absent, nothing changes:
    #: the detailed policy decides, exactly as before.
    simple_calibration_path: str | None = None
    #: Parameter and token provenance. Defaults reproduce the synthetic canonical
    #: workload byte-for-byte; a pinned checkpoint and real text give the
    #: workload a distinct identity (WorkloadSpec.weights / .data) that no
    #: synthetic calibration can bind to.
    weights: str = "random"
    data: str = "synthetic"
    #: The four-part protocol (schema evograd-qwen3-t3-protocol/4). The
    #: calibration supplies the frozen thresholds; the verdict file supplies
    #: part D for the provider being timed, because a 1,000-step training run
    #: does not belong inside a timing runner's gate.
    protocol4_calibration_path: str | None = None
    protocol4_verdict_path: str | None = None
    #: Time a provider whose frozen screening FAILED, labelled diagnostic. Never a pass.
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

            from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import (
                pretrained_snapshot,
            )
            from . import textdata

            snapshot = pretrained_snapshot(self.spec)
            tokenizer = AutoTokenizer.from_pretrained(snapshot)
            cache_dir = os.environ.get("EVOGRAD_HF_CACHE") or None
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

    # ── the whole-model gate ─────────────────────────────────────────────

    def site_preflight(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Does each patched site hold at the shapes this model will supply?

        Tier 3's runner already asks this before it reaches the gate. Asking
        again here costs one small run per site and makes the gate complete on
        its own, so a standalone verdict names the same first stage the runner
        would have failed at.
        """
        from evograd.evaluation.tier3.runner import preflight as run_preflight
        from evograd.benchmark import TASKS

        try:
            run_preflight(kernels, TASKS, device=device)
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

        # The protocol-4 policy binds to its own workload identity (real text);
        # the legacy canonical calibration below is synthetic-only and must not
        # be consulted first, or every real-text provider is refused for a
        # calibration it was never meant to use (found 2026-09-06).
        if self.protocol4_calibration_path:
            return self._protocol4_hook(kernels, device)

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
        simple_policy = None
        if self.simple_calibration_path:
            from . import simple as simple_gate

            payload = json.loads(
                Path(self.simple_calibration_path).read_text(encoding="utf-8"))
            try:
                simple_policy = simple_gate.SimplePolicy.from_dict(payload["policy"])
            except simple_gate.PolicyMismatch as exc:
                return {"gate": "qwen3_model_correctness", "ok": False,
                        "reason": f"simplified calibration rejected: {exc}"}

        preflight = self.site_preflight(kernels, device=device)
        from .gate import summarize

        return summarize(check_model_correctness(
            self, kernels, policy=policy, data_seed=self.data_seed,
            preflight=preflight,
            simple_policy=simple_policy,
            simple_primary=simple_policy is not None,
        ))

    def _protocol4_hook(self, kernels, device: str) -> dict[str, Any]:
        """A, B and C in-process; D from the frozen holdout verdict for this provider.

        Refuses -- and therefore the runner does not time -- when the policy does
        not bind, when B or C exceed their thresholds, or when no protocol-4
        holdout verdict exists for the provider being timed.
        """
        from . import boundary, protocol4, purity
        from .gate import summarize
        from evograd.evaluation.tier3.gate.numerics import environment_fingerprint, fingerprint_hash
        from .simple import PatchSet

        gate_name = "qwen3_protocol4"
        try:
            patch_set = PatchSet.of(kernels, layers=self.spec.arch["num_hidden_layers"])
            calibration_path, verdict_path = self._protocol4_files_for(patch_set)
            policy = protocol4.load_policy(calibration_path)
            plan = policy.training_plan
            policy.require_binding(
                workload_id=self.spec.workload_id, workload_hash=self.spec.workload_hash,
                dtype=str(self.spec.dtype).replace("torch.", ""),
                environment_hash=fingerprint_hash(environment_fingerprint()),
                patch_set=patch_set, data_identity_digest=self.data_identity_digest(),
                training_plan=plan)
        except Exception as exc:  # binding is the first gate
            return {"gate": gate_name, "ok": False, "failed_at": "policy_binding",
                    "reason": str(exc)}

        preflight = self.site_preflight(kernels, device=device)
        if not preflight.get("ok", True):
            return {"gate": gate_name, "ok": False, "failed_at": "site_preflight",
                    "reason": str(preflight.get("reason"))}
        pure = purity.run_for(kernels, self, device=device)
        if not pure.get("ok", False):
            return {"gate": gate_name, "ok": False, "failed_at": "provider_purity",
                    "reason": "provider is not a function of its arguments"}
        # The reference-calibrated local envelope, when the frozen calibration
        # derived one (protocol4_cli calibrate --local-envelope). Read from the
        # same file the policy binds from, so the two cannot come apart.
        envelope = json.loads(Path(calibration_path).read_text(
            encoding="utf-8")).get("local_envelope") or None
        local = boundary.validate_all_invocations(self, kernels, data_seed=self.data_seed,
                                                  envelope=envelope)
        if not local.get("ok", False):
            from .gate import _boundary_reason
            return {"gate": gate_name, "ok": False, "failed_at": "live_boundary",
                    "reason": _boundary_reason(local), "live_boundary": local}

        from evograd.evaluation.tier3.patch import KernelSet
        ids, labels = self.batch_for(seed=0)
        reference = protocol4.capture_first_step(self, KernelSet(registry=self.site_registry), ids, labels)
        provider = protocol4.capture_first_step(self, kernels, ids, labels)
        measured = protocol4.step_distances(provider, reference, labels, device=device)
        del reference, provider

        # Part D: the frozen holdout verdict for this provider, by kernel origin.
        # A screening policy (SCREENING_PLAN) enforces A/B/C only; its holdout
        # verdict is still required to exist for this provider, so nothing is
        # timed that the frozen policy has not judged on independent seeds.
        origin = tuple(sorted(s.origin for s in kernels.sources))
        if protocol4.requires_training_part(policy):
            d_part = _protocol4_training_part(verdict_path, origin)
            if d_part is None:
                return {"gate": gate_name, "ok": False, "failed_at": "training_behaviour",
                        "reason": (f"no protocol-4 holdout verdict for provider origin {origin} "
                                   f"at {verdict_path}; run protocol4_cli holdout"),
                        "live_boundary": local, "step": measured}
            measured.update(d_part)
        else:
            judged = _protocol4_screening_rows(verdict_path, origin)
            if not judged:
                return {"gate": gate_name, "ok": False, "failed_at": "screening_holdout",
                        "reason": (f"no A/B/C screening holdout verdict for provider origin "
                                   f"{origin} at {verdict_path}; run "
                                   "protocol4_cli holdout --screening"),
                        "live_boundary": local, "step": measured}
            failed = [r for r in judged if not r["ok"]]
            if failed and not self.protocol4_diagnostic_timing:
                return {"gate": gate_name, "ok": False, "failed_at": "screening_holdout",
                        "reason": (f"the frozen A/B/C screening FAILED for this provider on "
                                   f"holdout seeds {sorted(r['seed'] for r in failed)}; "
                                   "timing refused (pass --protocol4-diagnostic-timing to time "
                                   "it anyway, labelled diagnostic)"),
                        "live_boundary": local, "step": measured, "screening_holdout": judged}
            measured["screening_holdout"] = judged
            if failed:
                measured["diagnostic_only"] = (
                    f"screening FAILED on holdout seeds {sorted(r['seed'] for r in failed)}; "
                    "timed only because --protocol4-diagnostic-timing was given")
        verdict = protocol4.check(policy, measured)
        return {
            "gate": gate_name, "ok": verdict["ok"], "failed_at": verdict["failed_at"],
            "reason": verdict["reason"], "protocol4": verdict,
            "policy_file": str(calibration_path), "verdict_file": str(verdict_path),
            "local_check_mode": local.get("local_check_mode"),
            "diagnostic_only": measured.get("diagnostic_only"),
            "live_boundary": {k: local.get(k) for k in ("ok", "failure_count", "sites",
                                                        "local_check_mode", "envelope_admitted")},
            "provider_purity": {"ok": True}, "site_preflight": preflight,
        }

    def _protocol4_files_for(self, patch_set) -> tuple[str, str | None]:
        """The frozen policy and verdict files that describe this patch set.

        ``protocol4_calibration_path`` is either one policy file (bound below
        to whatever patch set the provider has, exactly as before) or a *policy
        index*: ``{"policy_index": [{"calibration": ..., "verdict": ...}, ...]}``,
        from which the entry whose frozen patch set matches the provider's is
        selected. One timing run can then hold providers with different patch
        sets -- four single sites and their composition -- in one randomized
        order against one eager baseline, each judged by its own calibration.
        A provider whose patch set no entry describes is refused, never judged
        by a neighbour's thresholds.
        """
        from . import protocol4

        payload = json.loads(Path(self.protocol4_calibration_path).read_text(encoding="utf-8"))
        if "policy_index" not in payload:
            return self.protocol4_calibration_path, self.protocol4_verdict_path
        base = Path(self.protocol4_calibration_path).parent
        for entry in payload["policy_index"]:
            calibration = Path(entry["calibration"])
            calibration = calibration if calibration.is_absolute() else base / calibration
            candidate = protocol4.load_policy(calibration)
            if patch_set.matches(candidate.patch_set):
                verdict = entry.get("verdict")
                if verdict is not None:
                    verdict = Path(verdict)
                    verdict = str(verdict if verdict.is_absolute() else base / verdict)
                return str(calibration), verdict
        raise protocol4.PolicyMismatch(
            f"no entry of the policy index describes patch set {patch_set.to_dict()}")

    @property
    def last_build(self) -> PatchedModel | None:
        """Provenance and invocation counters from the most recent build."""
        return self._last


def _protocol4_training_part(path: str | None, origin: tuple[str, ...]):
    """Part-D distances for one provider from a frozen holdout verdict file.

    Matched by kernel origin (``trusted_torch_compile``,
    ``candidate:direct_deployment``); the worst holdout seed is what is judged.
    """
    if not path or not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r for r in payload.get("results", [])
            if tuple(sorted(r["measured"].get("kernel_origin", []))) == origin]
    if not rows:
        return None
    keys = ("train_window_nll_max_abs_delta", "val_nll_max_abs_delta")
    worst = {k: max(float(r["measured"][k]) for r in rows) for k in keys}
    worst["non_finite_training_steps"] = [
        s for r in rows for s in r["measured"].get("non_finite_training_steps", [])]
    worst["training_seeds"] = sorted({r["seed"] for r in rows})
    return worst


def _protocol4_screening_rows(path: str | None, origin: tuple[str, ...]) -> list[dict[str, Any]]:
    """The frozen screening verdicts (seed, ok, ratios) for one provider origin."""
    if not path or not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"seed": r["seed"], "ok": bool(r["verdict"].get("ok")),
             "ratios": r["verdict"].get("ratios", {})}
            for r in payload.get("results", [])
            if tuple(sorted(r["measured"].get("kernel_origin", []))) == origin]


def _empty_provenance():
    from evograd.evaluation.tier3.patch import PatchProvenance

    return PatchProvenance(
        method="module_surgery", requested_sites=(), actual_sites=(), paths={}
    )


def from_config(config: dict[str, Any]) -> Qwen3Workload:
    """Rebuild a workload in a child process from its serialized fields."""
    return Qwen3Workload.from_config(config)
