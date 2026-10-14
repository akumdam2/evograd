"""Reading a tier-3 report: what it measured, and what it may claim.

Tier 3 is *integrated* evaluation: a candidate installed at declared sites of a
real model's whole training step (benchmark Level 3). Reports carry three
identity fields:

    evaluation_tier   3
    execution_scope   "model"
    benchmark_level   3

This module is the one place a report is identified. Three rules:

**The benchmark has three levels since 2026-10-13** (L1 primitive, L2
composite, L3 whole model). A report of the previous protocol
(``evograd-tier3-model-v2``) declared the whole model as level 4; it is read as
level 3 with ``renumbered_from=4``. A report written before the identity fields
existed is model-scope by construction and is marked ``legacy``.

**Block-scope reports are refused.** The block scope (one decoder layer,
forward plus VJP) was removed with the old Level 3; its reports
(``evograd-tier3-block-v1``) are historical and are never aggregated.

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
#: The previous whole-model protocol: identical measurement, level numbered 4.
TIER3_MODEL_PROTOCOL_V2 = "evograd-tier3-model-v2"
#: The removed block scope's protocol, named only to refuse it clearly.
REMOVED_BLOCK_PROTOCOL = "evograd-tier3-block-v1"

SCOPE_MODEL = "model"
SCOPES = (SCOPE_MODEL,)

#: The benchmark level a model-scope report measures.
MODEL_LEVEL = 3
#: What the v2 protocol called the same measurement.
MODEL_LEVEL_V2 = 4
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
    #: True when the scope was inferred from the protocol string because the
    #: report predates the explicit fields.
    legacy: bool
    #: The level number the report itself declared, when it differs from
    #: today's numbering (``4`` for an ``evograd-tier3-model-v2`` report).
    renumbered_from: int | None = None

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
            "renumbered_from": self.renumbered_from,
            "workload": self.workload,
            "candidate_task_levels": list(self.candidate_task_levels),
            "timing_boundary": self.timing_boundary,
            "legacy": self.legacy,
            "group": self.group,
        }


def identify_report(report: Mapping[str, Any]) -> Tier3Identity:
    """What a tier-3 report measured. Refuses what it cannot place."""
    protocol = _require(report, "protocol")
    if protocol == TIER3_MODEL_PROTOCOL_VERSION:
        return _identify_model_report(report, declared_level=MODEL_LEVEL)
    if protocol == TIER3_MODEL_PROTOCOL_V2:
        return _identify_model_report(report, declared_level=MODEL_LEVEL_V2)
    if protocol == REMOVED_BLOCK_PROTOCOL:
        raise UnknownTier3Report(
            f"{REMOVED_BLOCK_PROTOCOL} is the block scope, removed on 2026-10-13 with the "
            "old Level 3; its reports are historical and are not read")
    raise UnknownTier3Report(
        f"unknown tier-3 protocol {protocol!r}; known: "
        f"{[TIER3_MODEL_PROTOCOL_VERSION, TIER3_MODEL_PROTOCOL_V2]}"
    )


def _identify_model_report(report: Mapping[str, Any], *, declared_level: int) -> Tier3Identity:
    legacy = "execution_scope" not in report
    scope = SCOPE_MODEL if legacy else _require(report, "execution_scope")
    if scope != SCOPE_MODEL:
        raise UnknownTier3Report(
            f"{report['protocol']} is the whole-model training-step protocol; it cannot "
            f"carry execution_scope={scope!r}"
        )
    level = declared_level if legacy else _require(report, "benchmark_level")
    if level != declared_level:
        raise UnknownTier3Report(
            f"a {report['protocol']} report measures benchmark level {declared_level}, "
            f"not {level!r}"
        )
    tier = 3 if legacy else _require(report, "evaluation_tier")
    return Tier3Identity(
        protocol=str(report["protocol"]),
        evaluation_tier=int(tier),
        execution_scope=SCOPE_MODEL,
        benchmark_level=MODEL_LEVEL,
        renumbered_from=None if declared_level == MODEL_LEVEL else declared_level,
        workload=str(_require(report, "workload")),
        candidate_task_levels=_declared_task_levels(report),
        timing_boundary=(
            _require(report, "timing_protocol", "boundary")
            if "boundary" in (report.get("timing_protocol") or {})
            else BOUNDARY_OF_SCOPE[SCOPE_MODEL]
        ),
        legacy=legacy,
    )


def _declared_task_levels(report: Mapping[str, Any]) -> tuple[int, ...]:
    """The levels a model report declares, or ``()`` when it predates the field.

    A legacy report names its candidates' operators in ``kernel_sources`` but
    not their levels; deriving them needs the task registry, which this reader
    deliberately does not import. ``()`` means "not declared", never "none".
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
    "REMOVED_BLOCK_PROTOCOL",
    "TIER3_MODEL_PROTOCOL_V2",
    "TIER3_MODEL_PROTOCOL_VERSION",
    "ProviderRow",
    "Tier3Identity",
    "UnknownTier3Report",
    "group_reports",
    "identify_report",
    "provider_rows",
]
