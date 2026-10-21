"""Reading a tier-3 report: what it measured, and what it may claim.

Tier 3 is *integrated* evaluation: a candidate installed at declared sites of a
real model's whole training step (benchmark Level 3). Reports carry three
identity fields:

    evaluation_tier   3
    execution_scope   "model"
    benchmark_level   3

This module is the one place a report is identified. Two rules:

**Only the current protocol is read** (``evograd-tier3-model-v3``, with all
three identity fields); any other protocol is refused by name.

**Nothing numeric is read with a default.** A report that lacks a key raises
with the path it was looking for -- the ``.get(name, 0.0)`` that once published
a 1.000 memory ratio for every operator is exactly what this refuses to repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evograd.evaluation.common.report import ReportFieldError, _require

#: The whole-model training-step protocol. Equal to
#: ``runner.TIER3_PROTOCOL_VERSION``; spelled here so this module does not
#: import the runner (and torch) to read a JSON file.
TIER3_MODEL_PROTOCOL_VERSION = "evograd-tier3-model-v3"

SCOPE_MODEL = "model"
SCOPES = (SCOPE_MODEL,)

#: The benchmark level a model-scope report measures.
MODEL_LEVEL = 3
LEVEL_OF_SCOPE = {SCOPE_MODEL: MODEL_LEVEL}

#: What the timing loop wraps. Recorded so a speedup is never read without the
#: boundary it was measured across.
BOUNDARY_OF_SCOPE = {SCOPE_MODEL: "loss.backward() + optimizer step"}


class UnknownTier3Report(ValueError):
    """Not a tier-3 report this reader knows how to interpret."""


@dataclass(frozen=True)
class Tier3Identity:
    """What one tier-3 report measured, as far as aggregation needs to know."""

    protocol: str
    evaluation_tier: int
    execution_scope: str
    benchmark_level: int
    #: The workload's ``describe()`` key (``qwen3_next_token``).
    workload: str
    #: The benchmark levels of the tasks the candidates implement -- ``(2,)``
    #: for a composition of L2 sites. Distinct from ``benchmark_level``.
    candidate_task_levels: tuple[int, ...]
    #: The timing boundary the latency numbers span.
    timing_boundary: str

    @property
    def group(self) -> str:
        """The aggregation key."""
        return f"tier3/{self.execution_scope}/{self.workload}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "evaluation_tier": self.evaluation_tier,
            "execution_scope": self.execution_scope,
            "benchmark_level": self.benchmark_level,
            "workload": self.workload,
            "candidate_task_levels": list(self.candidate_task_levels),
            "timing_boundary": self.timing_boundary,
            "group": self.group,
        }


def identify_report(report: Mapping[str, Any]) -> Tier3Identity:
    """What a tier-3 report measured. Refuses what it cannot place."""
    protocol = _require(report, "protocol")
    if protocol != TIER3_MODEL_PROTOCOL_VERSION:
        raise UnknownTier3Report(
            f"unknown tier-3 protocol {protocol!r}; known: {[TIER3_MODEL_PROTOCOL_VERSION]}")
    scope = _require(report, "execution_scope")
    if scope != SCOPE_MODEL:
        raise UnknownTier3Report(
            f"{protocol} is the whole-model training-step protocol; it cannot "
            f"carry execution_scope={scope!r}")
    level = _require(report, "benchmark_level")
    if level != MODEL_LEVEL:
        raise UnknownTier3Report(
            f"a {protocol} report measures benchmark level {MODEL_LEVEL}, not {level!r}")
    return Tier3Identity(
        protocol=str(protocol),
        evaluation_tier=int(_require(report, "evaluation_tier")),
        execution_scope=SCOPE_MODEL,
        benchmark_level=MODEL_LEVEL,
        workload=str(_require(report, "workload")),
        candidate_task_levels=_declared_task_levels(report),
        timing_boundary=(
            _require(report, "timing_protocol", "boundary")
            if "boundary" in (report.get("timing_protocol") or {})
            else BOUNDARY_OF_SCOPE[SCOPE_MODEL]
        ),
    )


def _declared_task_levels(report: Mapping[str, Any]) -> tuple[int, ...]:
    """The levels a model report declares, or ``()`` when it does not.

    Deriving them from ``kernel_sources`` needs the task registry, which this
    reader deliberately does not import. ``()`` means "not declared", never "none".
    """
    declared = report.get("candidate_task_levels")
    if declared is None:
        return ()
    return tuple(int(x) for x in declared)


# ── per-provider rows ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderRow:
    """One provider of one report.

    ``latency_ms`` spans the training step. Memory fields are ``None`` where the
    report says they were not measured; a report that does not carry the field
    at all is refused by :func:`provider_rows`.
    """

    provider: str
    ok: bool
    failed_at: str | None
    error: str | None
    patched: tuple[str, ...]
    latency_ms: float | None
    reference_latency_ms: float | None
    execution_peak_bytes: int | None
    saved_state_bytes: int | None
    validation_peak_bytes: int | None
    #: ``accepted`` / ``rejected`` / ``incomplete``; ``None`` for a report that
    #: predates the field.
    acceptance: str | None = None
    evaluation_complete: bool | None = None
    numerical_status: str | None = None

    @property
    def speedup_vs_reference(self) -> float | None:
        if self.latency_ms is None or self.reference_latency_ms is None:
            return None
        if self.latency_ms <= 0.0:
            return None
        return self.reference_latency_ms / self.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "failed_at": self.failed_at,
            "error": self.error,
            "patched": list(self.patched),
            "latency_ms": self.latency_ms,
            "reference_latency_ms": self.reference_latency_ms,
            "speedup_vs_reference": self.speedup_vs_reference,
            "execution_peak_bytes": self.execution_peak_bytes,
            "saved_state_bytes": self.saved_state_bytes,
            "validation_peak_bytes": self.validation_peak_bytes,
            "acceptance": self.acceptance,
            "evaluation_complete": self.evaluation_complete,
            "numerical_status": self.numerical_status,
        }


#: The provider each scope measures speedups against.
REFERENCE_OF_SCOPE = {SCOPE_MODEL: "eager"}


def provider_rows(report: Mapping[str, Any]) -> tuple[ProviderRow, ...]:
    """Every provider of a report as a :class:`ProviderRow`, in report order."""
    identity = identify_report(report)
    providers = _require(report, "providers")
    reference = REFERENCE_OF_SCOPE[identity.execution_scope]
    base = providers.get(reference)
    reference_latency = (
        _latency(identity.execution_scope, base) if base and base.get("ok") else None
    )
    rows = []
    for name, entry in providers.items():
        ok = bool(_require(entry, "ok"))
        verdict = {
            "acceptance": (entry.get("acceptance") or {}).get("status"),
            "evaluation_complete": entry.get("evaluation_complete"),
            "numerical_status": entry.get("numerical_status"),
        }
        if not ok:
            rows.append(ProviderRow(
                provider=name, ok=False,
                failed_at=str(_require(entry, "failed_at")),
                error=str(_require(entry, "error")),
                patched=tuple(entry.get("patched") or ()),
                latency_ms=None, reference_latency_ms=reference_latency,
                execution_peak_bytes=None, saved_state_bytes=None,
                validation_peak_bytes=None, **verdict,
            ))
            continue
        memory = _memory(identity.execution_scope, entry)
        rows.append(ProviderRow(
            provider=name, ok=True, failed_at=None, error=None,
            patched=tuple(_require(entry, "patched")),
            latency_ms=_latency(identity.execution_scope, entry),
            reference_latency_ms=reference_latency,
            **memory, **verdict,
        ))
    return tuple(rows)


def _latency(scope: str, entry: Mapping[str, Any]) -> float:
    return float(_require(entry, "step_ms"))


def _optional_int(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportFieldError(f"{where}: expected a byte count or null, got {value!r}")
    return int(value)


def _memory(scope: str, entry: Mapping[str, Any]) -> dict[str, int | None]:
    # The model runner observes memory from the outside: one peak over a
    # settled step. Saved state lives inside autograd's ctx there and is not
    # measurable, so it is None rather than 0 -- and the validation-process
    # peak is not something the model runner ever recorded.
    return {
        "execution_peak_bytes": _optional_int(
            _require(entry, "peak_memory_bytes"), where="peak_memory_bytes"),
        "saved_state_bytes": None,
        "validation_peak_bytes": None,
    }


# ── aggregation groups ──────────────────────────────────────────────────────


def group_reports(
    reports: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[Tier3Identity, ...]]:
    """Reports keyed by :attr:`Tier3Identity.group` (scope and workload)."""
    groups: dict[str, list[Tier3Identity]] = {}
    for report in reports:
        identity = identify_report(report)
        groups.setdefault(identity.group, []).append(identity)
    return {key: tuple(value) for key, value in groups.items()}


__all__ = [
    "BOUNDARY_OF_SCOPE",
    "LEVEL_OF_SCOPE",
    "REFERENCE_OF_SCOPE",
    "SCOPE_MODEL",
    "SCOPES",
    "MODEL_LEVEL",
    "TIER3_MODEL_PROTOCOL_VERSION",
    "ProviderRow",
    "Tier3Identity",
    "UnknownTier3Report",
    "group_reports",
    "identify_report",
    "provider_rows",
]
