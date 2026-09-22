"""Reading a tier-3 report: which scope it measured, and what it may claim.

Tier 3 is *integrated* evaluation -- a candidate installed at declared sites of
a real architecture. Until now the only integrated execution was a whole
model's training step (benchmark level 4, ``evograd-tier3-model-v2``). The
block scope (benchmark level 3: one architectural block, forward plus a
vector-Jacobian backward from supplied cotangents) is a different measurement
with a different timing boundary, and its reports say so in three fields:

    evaluation_tier   3
    execution_scope   "model" | "block"
    benchmark_level   4 | 3

This module is the one place a report is identified. Two rules:

**A report written before those fields existed is model-scope by
construction.** Its protocol string already says what it measured; the reader
fills the scope in from the protocol and marks the identity ``legacy`` so a
consumer can see that the field was inferred, not declared. It is never
re-read as a block measurement, and a block report is never aggregated as
full-model coverage: :attr:`Tier3Identity.group` puts the scope in the
aggregation key.

**Nothing numeric is read with a default.** A block report that did not
measure saved-state memory carries ``null`` there, and the row says ``None``.
A report that lacks the key altogether raises with the path it was looking
for -- the ``.get(name, 0.0)`` that once published a 1.000 memory ratio for
every operator is exactly what this refuses to repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evograd.evaluation.common.report import ReportFieldError, _require

#: The whole-model training-step protocol every existing tier-3 report carries.
#: Equal to ``runner.TIER3_PROTOCOL_VERSION``; spelled here so this module does
#: not import the runner (and torch) to read a JSON file.
TIER3_MODEL_PROTOCOL_VERSION = "evograd-tier3-model-v2"

#: The block-scope protocol emitted by the shared block executor.
TIER3_BLOCK_PROTOCOL_VERSION = "evograd-tier3-block-v1"

SCOPE_MODEL = "model"
SCOPE_BLOCK = "block"
SCOPES = (SCOPE_MODEL, SCOPE_BLOCK)

#: Benchmark level each scope measures. Fixed by the scope, never read from the
#: report alone: a "block" report claiming level 4 is a mislabel, not a choice.
LEVEL_OF_SCOPE = {SCOPE_MODEL: 4, SCOPE_BLOCK: 3}

#: What each scope's timing loop wraps. Recorded so a speedup is never read
#: without the boundary it was measured across.
BOUNDARY_OF_SCOPE = {
    SCOPE_MODEL: "loss.backward() + optimizer step",
    SCOPE_BLOCK: "forward + vjp",
}


class UnknownTier3Report(ValueError):
    """Not a tier-3 report this reader knows how to interpret."""


@dataclass(frozen=True)
class Tier3Identity:
    """What one tier-3 report measured, as far as aggregation needs to know."""

    protocol: str
    evaluation_tier: int
    execution_scope: str
    benchmark_level: int
    #: The architecture the report belongs to. For a model-scope report the
    #: workload's ``describe()`` key (``qwen3_next_token``); for a block report
    #: the case's architecture key (``qwen3_0_6b``).
    workload: str
    #: Block case id (``qwen3_0_6b/decoder_layer@14/captured/...``); ``None``
    #: for model scope, where the workload id plays that part.
    case: str | None
    #: The benchmark levels of the tasks the candidates implement -- ``(2,)`` for
    #: a composition of L2 sites. Distinct from ``benchmark_level``: a block
    #: (L3) run of L2 candidates is exactly the first version's shape, and a
    #: report must not let the two be confused.
    candidate_task_levels: tuple[int, ...]
    #: The timing boundary the latency numbers span.
    timing_boundary: str
    #: True when the scope was inferred from the protocol string because the
    #: report predates the explicit fields.
    legacy: bool

    @property
    def group(self) -> str:
        """The aggregation key. Scope is part of it, so a block speedup and a
        whole-model speedup on the same architecture never pool."""
        base = f"tier3/{self.execution_scope}/{self.workload}"
        return base if self.case is None else f"{base}/{self.case}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "evaluation_tier": self.evaluation_tier,
            "execution_scope": self.execution_scope,
            "benchmark_level": self.benchmark_level,
            "workload": self.workload,
            "case": self.case,
            "candidate_task_levels": list(self.candidate_task_levels),
            "timing_boundary": self.timing_boundary,
            "legacy": self.legacy,
            "group": self.group,
        }


def identify_report(report: Mapping[str, Any]) -> Tier3Identity:
    """Which scope a tier-3 report measured. Refuses what it cannot place."""
    protocol = _require(report, "protocol")
    if protocol == TIER3_MODEL_PROTOCOL_VERSION:
        return _identify_model_report(report)
    if protocol == TIER3_BLOCK_PROTOCOL_VERSION:
        return _identify_block_report(report)
    raise UnknownTier3Report(
        f"unknown tier-3 protocol {protocol!r}; known: "
        f"{[TIER3_MODEL_PROTOCOL_VERSION, TIER3_BLOCK_PROTOCOL_VERSION]}"
    )


def _identify_model_report(report: Mapping[str, Any]) -> Tier3Identity:
    legacy = "execution_scope" not in report
    scope = SCOPE_MODEL if legacy else _require(report, "execution_scope")
    if scope != SCOPE_MODEL:
        raise UnknownTier3Report(
            f"{TIER3_MODEL_PROTOCOL_VERSION} is the whole-model training-step "
            f"protocol; it cannot carry execution_scope={scope!r}"
        )
    level = LEVEL_OF_SCOPE[SCOPE_MODEL] if legacy else _require(report, "benchmark_level")
    if level != LEVEL_OF_SCOPE[SCOPE_MODEL]:
        raise UnknownTier3Report(
            f"a model-scope report measures benchmark level "
            f"{LEVEL_OF_SCOPE[SCOPE_MODEL]}, not {level!r}"
        )
    tier = 3 if legacy else _require(report, "evaluation_tier")
    return Tier3Identity(
        protocol=TIER3_MODEL_PROTOCOL_VERSION,
        evaluation_tier=int(tier),
        execution_scope=SCOPE_MODEL,
        benchmark_level=LEVEL_OF_SCOPE[SCOPE_MODEL],
        workload=str(_require(report, "workload")),
        case=None,
        candidate_task_levels=_declared_task_levels(report),
        timing_boundary=(
            _require(report, "timing_protocol", "boundary")
            if "boundary" in (report.get("timing_protocol") or {})
            else BOUNDARY_OF_SCOPE[SCOPE_MODEL]
        ),
        legacy=legacy,
    )


def _identify_block_report(report: Mapping[str, Any]) -> Tier3Identity:
    scope = _require(report, "execution_scope")
    if scope != SCOPE_BLOCK:
        raise UnknownTier3Report(
            f"{TIER3_BLOCK_PROTOCOL_VERSION} is the block-scope protocol; it "
            f"cannot carry execution_scope={scope!r}"
        )
    level = _require(report, "benchmark_level")
    if level != LEVEL_OF_SCOPE[SCOPE_BLOCK]:
        raise UnknownTier3Report(
            f"a block-scope report measures benchmark level "
            f"{LEVEL_OF_SCOPE[SCOPE_BLOCK]}, not {level!r}"
        )
    levels = _require(report, "candidate_task_levels")
    return Tier3Identity(
        protocol=TIER3_BLOCK_PROTOCOL_VERSION,
        evaluation_tier=int(_require(report, "evaluation_tier")),
        execution_scope=SCOPE_BLOCK,
        benchmark_level=LEVEL_OF_SCOPE[SCOPE_BLOCK],
        workload=str(_require(report, "case", "architecture")),
        case=str(_require(report, "case", "case_id")),
        candidate_task_levels=tuple(int(x) for x in levels),
        timing_boundary=str(_require(report, "timing_protocol", "boundary")),
        legacy=False,
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
    """One provider of one report, in scope-neutral terms.

    ``latency_ms`` spans the scope's timing boundary: the training step for a
    model report, forward + VJP for a block report. Memory fields are ``None``
    where the report says they were not measured; a report that does not carry
    the field at all is refused by :func:`provider_rows`.
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
        }


#: The provider each scope measures speedups against.
REFERENCE_OF_SCOPE = {SCOPE_MODEL: "eager", SCOPE_BLOCK: "native"}


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
        if not ok:
            rows.append(ProviderRow(
                provider=name, ok=False,
                failed_at=str(_require(entry, "failed_at")),
                error=str(_require(entry, "error")),
                patched=tuple(entry.get("patched") or ()),
                latency_ms=None, reference_latency_ms=reference_latency,
                execution_peak_bytes=None, saved_state_bytes=None,
                validation_peak_bytes=None,
            ))
            continue
        memory = _memory(identity.execution_scope, entry)
        rows.append(ProviderRow(
            provider=name, ok=True, failed_at=None, error=None,
            patched=tuple(_require(entry, "patched")),
            latency_ms=_latency(identity.execution_scope, entry),
            reference_latency_ms=reference_latency,
            **memory,
        ))
    return tuple(rows)


def _latency(scope: str, entry: Mapping[str, Any]) -> float:
    if scope == SCOPE_MODEL:
        return float(_require(entry, "step_ms"))
    return float(_require(entry, "latency", "forward_backward_ms"))


def _optional_int(value: Any, *, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportFieldError(f"{where}: expected a byte count or null, got {value!r}")
    return int(value)


def _memory(scope: str, entry: Mapping[str, Any]) -> dict[str, int | None]:
    if scope == SCOPE_MODEL:
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
    memory = _require(entry, "memory")
    return {
        key: _optional_int(_require(memory, key), where=f"memory.{key}")
        for key in ("execution_peak_bytes", "saved_state_bytes", "validation_peak_bytes")
    }


# ── aggregation groups ──────────────────────────────────────────────────────


def group_reports(
    reports: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[Tier3Identity, ...]]:
    """Reports keyed by :attr:`Tier3Identity.group`.

    The point is what is *not* in one group: a block report and a model report
    of the same architecture land in different keys, so no aggregate can count
    a block-scope speedup as full-model coverage or the reverse.
    """
    groups: dict[str, list[Tier3Identity]] = {}
    for report in reports:
        identity = identify_report(report)
        groups.setdefault(identity.group, []).append(identity)
    return {key: tuple(value) for key, value in groups.items()}


__all__ = [
    "BOUNDARY_OF_SCOPE",
    "LEVEL_OF_SCOPE",
    "REFERENCE_OF_SCOPE",
    "SCOPE_BLOCK",
    "SCOPE_MODEL",
    "SCOPES",
    "TIER3_BLOCK_PROTOCOL_VERSION",
    "TIER3_MODEL_PROTOCOL_VERSION",
    "ProviderRow",
    "Tier3Identity",
    "UnknownTier3Report",
    "group_reports",
    "identify_report",
    "provider_rows",
]
