"""The canonical OLMoE-1B-7B training step, as a tier-3 workload.

    python -m evograd.evaluation.tier3.cli --model olmoe_1b_7b --structural-identity

A top-level dataclass of JSON-serializable fields, for the same reason as the
other workloads': tier 3 runs each provider in a killable child process, and
:meth:`OlmoeWorkload.to_config` / :func:`from_config` rebuild an identical
workload on the other side.

The execution is ``levels.level3.spec.CANONICAL``: OLMoE-1B-7B, batch 2,
sequence 2048, BF16, CUDA, SDPA, grouped-GEMM experts, ``use_cache`` off, no
checkpointing, ``model.train()``, random weights through the model's own
``post_init``, synthetic tokens. The runner adds AdamW
(``tier3.runner.OPTIMIZER_DEFAULTS``) around ``loss.backward()``. Sites are
patched on all 16 layers; the counts are held to the depth.

**No whole-model gate yet.** :meth:`model_correctness` exists -- the runner
calls it, and a whole-model gate will be bridged in later -- but it reports
the gate as not implemented: ``evaluation_complete: false`` with the reason,
``ok`` so the provider is still timed. Site preflight (declared correctness
grid) still runs before anything is timed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.model import (
    build_model,
    make_inputs,
    olmoe_effective_settings,
)
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.spec import CANONICAL, WorkloadSpec

from .sites import PatchedModel, SiteCounters, expected_counts, olmoe_sites, patch_model

#: The name the tier-3 CLI selects this workload by.
MODEL_KEY = "olmoe_1b_7b"

#: Whole-model ``torch.compile`` of the unpatched model: a baseline, not a site
#: provider (the other workloads' settings, recorded in every build's provenance).
WHOLE_MODEL_COMPILE_ORIGIN = "whole_model_torch_compile"
WHOLE_MODEL_COMPILE_SITE = "__whole_model__"
WHOLE_MODEL_COMPILE_SETTINGS: dict[str, Any] = {
    "backend": "inductor", "mode": None, "dynamic": False, "fullgraph": False,
}

#: Why the whole-model gate does not run, stated in every provider row.
MODEL_GATE_NOT_IMPLEMENTED = (
    "the OLMoE-1B-7B whole-model gate is not implemented yet; only site preflight ran")


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
                    "experts_implementation")
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
    return {"gate": "olmoe_model_gate", "implemented": False, "ok": True, "skipped": True,
            "execution_ok": True, "numerical_ok": None, "numerical_status": "unavailable",
            "numerical_unavailable_at": ["model_gate"], "evaluation_complete": False,
            "evaluation_incomplete_because": [reason], "reason": reason, "findings": []}


@dataclass
class OlmoeWorkload:
    """OLMoE-1B-7B next-token training, one step at a time (tier 3's ``TrainingWorkload``)."""

    batch_size: int = CANONICAL.batch_size
    seq_len: int = CANONICAL.seq_len
    dtype: str = CANONICAL.dtype
    device: str = CANONICAL.device
    attn_implementation: str = CANONICAL.attn_implementation
    #: Which kernel runs the experts (hashed into the identity; see the spec).
    experts_implementation: str = CANONICAL.experts_implementation
    seed: int = CANONICAL.seed
    #: Moves the token stream without touching the workload identity.
    data_seed: int = 0
    #: Reduced architectures for tests and debug depths (``--layers``).
    arch_overrides: dict[str, Any] = field(default_factory=dict)

    unit_name = "tokens"

    def __post_init__(self) -> None:
        self.spec = _spec_from_config(self.to_config())
        self.spec.validate()
        self.name = self.spec.workload_id
        self.site_registry = olmoe_sites()
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
            "experts_implementation": self.experts_implementation,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "arch_overrides": dict(self.arch_overrides),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OlmoeWorkload":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in config.items() if k in known})

    # ── the tier-3 protocol ──────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return int(self.spec.arch["num_hidden_layers"])

    def units_per_step(self) -> int:
        return self.spec.token_count

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def build_patched(self, kernels):
        """A fresh model with this provider's sites installed on every layer.
        Deterministic: weights are seeded on the CPU."""
        from evograd.evaluation.tier3.patch import PatchProvenance

        model = build_model(self.spec)
        expected = expected_counts(self.num_layers)
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
            model, kernels, expected_layers=self.num_layers)
        self._last = PatchedModel(model=model, provenance=provenance, counters=counters,
                                  carrier=carrier, expected=expected)
        return model, provenance

    def batch_for(self, *, seed: int):
        """Deterministic ``(input_ids, labels)``; ``labels = input_ids.clone()``."""
        spec = self.spec.replace(seed=self.data_seed + seed)
        return make_inputs(spec)

    def loss(self, model, batch) -> torch.Tensor:
        """The causal-LM loss plus the load-balancing auxiliary loss: what
        ``OlmoeForCausalLM`` returns with ``output_router_logits`` pinned on."""
        input_ids, labels = batch
        return model(input_ids=input_ids, labels=labels, use_cache=False).loss

    def describe(self) -> dict[str, Any]:
        arch = self.spec.arch
        return {
            "workload": "olmoe_next_token",
            "name": self.name,
            "workload_id": self.spec.workload_id,
            "workload_hash": self.spec.workload_hash,
            "config_hash": self.spec.config_hash,
            "model": self.spec.model_name,
            "layers": self.num_layers,
            "experts": {"num_experts": arch["num_experts"], "top_k": arch["num_experts_per_tok"],
                        "intermediate_size": arch["intermediate_size"],
                        "router_aux_loss_coef": arch["router_aux_loss_coef"],
                        "output_router_logits": arch["output_router_logits"]},
            "batch": self.spec.batch_size,
            "tokens": self.spec.seq_len,
            "units_per_step": self.units_per_step(),
            "unit_name": self.unit_name,
            "dtype": self.spec.dtype,
            "attn_implementation": self.spec.attn_implementation,
            "experts_implementation": self.spec.experts_implementation,
            "use_cache": self.spec.use_cache,
            "gradient_checkpointing": self.spec.gradient_checkpointing,
            "training": self.spec.training,
            "seed": self.spec.seed,
            "data_seed": self.data_seed,
            "canonical": self.spec.is_canonical,
            "config": self.to_config(),
            "input_checksum": self.input_checksum(),
            "expected_site_counts": expected_counts(self.num_layers),
            "model_gate": {"implemented": False, "reason": MODEL_GATE_NOT_IMPLEMENTED},
        }

    def input_checksum(self, *, seed: int = 0) -> str:
        spec = self.spec.replace(seed=self.data_seed + seed, device="cpu")
        ids, _labels = make_inputs(spec)
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        return f"{digest[:16]}:{tuple(ids.shape)}"

    def runtime_report(self, model) -> dict[str, Any]:
        """What the built model reports (experts dispatch included), not what was requested."""
        inner = getattr(model, "_orig_mod", model)
        report = olmoe_effective_settings(inner, self.spec)
        if inner is not model:
            report["whole_model_torch_compile"] = dict(WHOLE_MODEL_COMPILE_SETTINGS)
        return report

    def site_preflight(self, kernels, *, device: str = "cuda") -> dict[str, Any]:
        """Does each patched site hold on its declared correctness grid?

        No observed shapes are attached (provisional tolerances; see
        ``sites.olmoe_sites``), so this is the grid only.
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
        from .sites import live_sites

        model, _ = self.build_patched(kernels)
        chosen = range(self.num_layers) if layers is None else list(layers)
        expected = {site: len(chosen) for site in live_sites(kernels.patched)}
        return validate_invocations(model, kernels, self.batch_for(seed=seed), self.loss,
                                    registry=self.site_registry, dtype=self.spec.dtype,
                                    layers=layers, expected=expected)

    @property
    def last_build(self) -> PatchedModel | None:
        return self._last


def from_config(config: dict[str, Any]) -> OlmoeWorkload:
    """Rebuild a workload in a child process from its serialized fields."""
    return OlmoeWorkload.from_config(config)


__all__ = ["MODEL_GATE_NOT_IMPLEMENTED", "MODEL_KEY", "OlmoeWorkload",
           "WHOLE_MODEL_COMPILE_SETTINGS", "from_config", "model_gate_not_implemented",
           "whole_model_compile_kernels"]
