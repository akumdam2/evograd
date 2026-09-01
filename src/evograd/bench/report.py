"""One report shape, whatever protocol produced it.

Two timing protocols existed before this module and each invented its own
names for the same quantities: the low-overhead harness emits ``saved_bytes``
and ``speedup_vs_baseline_raw_full_step``; the final-report protocol emits
``logical_saved_bytes`` and ``speedup.pair_full``, and carries no input total
at all. The suite then reconciled them with two hand-written adapters, each of
which restated what the numbers *mean*.

That arrangement failed once already, and quietly. The fair adapter read the
harness's ``saved_bytes`` name, got nothing, and — because the missing key was
read with a ``0.0`` default and a geometric mean over an all-zero set is 1.0 —
published a plausible ``1.000`` saved-memory aggregate for every operator. A
wrong number that looks exactly like a right one is the expensive kind.

Two rules here prevent the recurrence, and both matter more than the schema
itself:

**Nothing is read with a default.** :func:`_require` raises and names the path
it was looking for. A protocol that stops emitting a field breaks loudly at the
boundary instead of contributing a zero to an average.

**Ratios are derived, never read.** A case stores times; ``speedup_full`` is a
property computed from them in one place. There is no speedup key to read, so
there is no wrong speedup key to read by accident — which is what the suite's
``FULL_STEP_SPEEDUP_KEY`` constant was defending against by hand.

Adding a protocol means writing one reader that fills :class:`CaseMetrics`.
Everything downstream — aggregation, pooling, coverage, the markdown table —
is inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Evaluation tier: what is being measured, as opposed to how carefully.
#: Deliberately distinct from ``OpDecl.level``, which is the *task* hierarchy
#: (1 primitive, 2 fused, 3 block). An operator of any level can be measured at
#: any tier.
TIER_PAIR = "pair"  # candidate forward/backward invoked directly
TIER_OPERATOR = "operator"  # one nn.Module through the autograd engine
TIER_MODEL = "model"  # a full training step

TIERS = (TIER_PAIR, TIER_OPERATOR, TIER_MODEL)

#: How carefully it was measured.
PROTOCOL_FAIR = "fair"
PROTOCOL_FAST = "fast"


class ReportFieldError(KeyError):
    """A protocol reader asked for a field its report does not carry."""


def _require(mapping: Any, *path: str) -> Any:
    """Fetch ``mapping[path[0]][path[1]]...``, raising with the full path.

    The whole point is the absence of a default. Every silent-zero bug this
    module exists to prevent began as a ``.get(name, 0.0)``.
    """
    cursor = mapping
    walked: list[str] = []
    for key in path:
        if not isinstance(cursor, dict):
            raise ReportFieldError(
                f"{'.'.join(walked) or '<root>'} is {type(cursor).__name__}, "
                f"not a mapping; cannot read {key!r}"
            )
        if key not in cursor:
            raise ReportFieldError(
                f"missing {'.'.join(walked + [key])!r}; "
                f"{'.'.join(walked) or '<root>'} has {sorted(cursor)}"
            )
        cursor = cursor[key]
        walked.append(key)
    return cursor


@dataclass(frozen=True)
class CaseMetrics:
    """One workload, one candidate, one baseline — times only.

    Every field is a measured quantity. Nothing here is a ratio, a score, or an
    aggregate; those are derived, so that their definitions live in exactly one
    place and cannot drift between protocols.
    """

    dims: dict[str, int]
    dtype: str
    ok: bool
    #: ``None`` where a protocol does not time that region. The pair protocol
    #: times all three; a future tier may time only the full step.
    candidate_forward_ms: float | None = None
    candidate_backward_ms: float | None = None
    candidate_full_ms: float | None = None
    baseline_forward_ms: float | None = None
    baseline_backward_ms: float | None = None
    baseline_full_ms: float | None = None
    #: Bytes the candidate retained for its backward, and the bytes of the
    #: declared memory inputs it was given.
    saved_bytes: float = 0.0
    input_bytes: float = 0.0
    error: str | None = None

    @staticmethod
    def _ratio(baseline: float | None, candidate: float | None) -> float | None:
        if baseline is None or candidate is None or candidate <= 0.0:
            return None
        return baseline / candidate

    @property
    def speedup_full(self) -> float | None:
        """The suite's metric: like-for-like forward+backward.

        Never the backward-only ratio. The eager baseline's "backward" runs the
        forward too, while a candidate's backward is timed from pre-saved
        state, so that comparison is inflated by roughly the forward's share of
        the step — about half for a level-3 block.
        """
        return self._ratio(self.baseline_full_ms, self.candidate_full_ms)

    @property
    def speedup_backward(self) -> float | None:
        return self._ratio(self.baseline_backward_ms, self.candidate_backward_ms)

    @property
    def speedup_forward(self) -> float | None:
        return self._ratio(self.baseline_forward_ms, self.candidate_forward_ms)


@dataclass(frozen=True)
class BenchReport:
    """A protocol-independent measurement of one operator."""

    op: str
    tier: str
    protocol: str
    baseline: str
    cases: tuple[CaseMetrics, ...] = ()
    environment: dict[str, Any] = field(default_factory=dict)
    #: Set when setup failed and no case ran.
    error: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; known: {list(TIERS)}")

    @property
    def ok_cases(self) -> tuple[CaseMetrics, ...]:
        return tuple(c for c in self.cases if c.ok and c.speedup_full is not None)


def summarize_error(error: Any, *, limit: int = 200) -> str:
    """One line naming why a case did not run, without the traceback.

    Distinct reasons are collected per operator, so several shapes failing for
    the same reason report it once rather than repeating it per case.
    """
    if not error:
        return ""
    if isinstance(error, dict):
        kind = error.get("error_type") or "error"
        message = " ".join(str(error.get("error_message", "")).split())
        text = f"{kind}: {message}" if message else str(kind)
    else:
        text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def declared_input_bytes(op, case: dict[str, Any]) -> float:
    """Bytes of the inputs the saved-memory ratio is measured against.

    The fair protocol records what a provider retained but not what it was
    given, so the denominator is reconstructed from the declaration's shapes
    bound to this case's dims — the same set ``memory_input_names()`` selects,
    so integer labels and rotary tables stay out of it exactly as they do on
    the low-overhead path.
    """
    if op is None:
        return 0.0
    memory_inputs = tuple(op.memory_input_names())
    if not memory_inputs:
        return 0.0
    from evograd.opdecl.activity import bind_shape

    dims = case.get("dims") or {}
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}
    element = dtype_bytes.get(case.get("dtype", ""), 4)
    by_name = {arg.name: arg for arg in op.args}
    total = 0.0
    for name in memory_inputs:
        arg = by_name.get(name)
        if arg is None or not getattr(arg, "shape", None):
            continue
        try:
            shape = bind_shape(arg.shape, dims)
        except Exception:
            continue
        count = 1
        for extent in shape:
            count *= extent
        total += count * element
    return float(total)


# ── protocol readers ─────────────────────────────────────────────────────────
# Each reader is a field mapping and nothing else: no arithmetic, no policy, no
# defaults on measured values. Keeping them this thin is what makes them
# auditable at a glance, which two hundred lines of adapter were not.
#
# Structural metadata (was there a setup error, did this case run) may be
# absent, because absence there is unambiguous. A missing *measurement* is not:
# it is the thing that becomes a zero and then an average.


def from_harness_report(
    op_name: str,
    report: dict[str, Any],
    *,
    tier: str = TIER_PAIR,
    protocol: str = PROTOCOL_FAST,
) -> BenchReport:
    """Read a ``harness.run_benchmarks(..., on_error='record')`` report."""
    cases: list[CaseMetrics] = []
    for case in report.get("cases") or []:
        if not case.get("ok"):
            cases.append(
                CaseMetrics(
                    dims=dict(case.get("dims") or {}),
                    dtype=str(case.get("dtype", "")),
                    ok=False,
                    error=summarize_error(case.get("error")) or None,
                )
            )
            continue
        cases.append(
            CaseMetrics(
                dims=dict(_require(case, "dims")),
                dtype=str(_require(case, "dtype")),
                ok=True,
                candidate_forward_ms=float(_require(case, "forward_ms")),
                candidate_backward_ms=float(_require(case, "backward_from_saved_ms")),
                candidate_full_ms=float(
                    _require(case, "raw_forward_backward_full_step_ms")
                ),
                # The harness does not time the baseline's forward alone.
                baseline_forward_ms=None,
                baseline_backward_ms=float(_require(case, "baseline_backward_ms")),
                baseline_full_ms=float(_require(case, "baseline_raw_full_step_ms")),
                saved_bytes=float(_require(case, "saved_bytes")),
                input_bytes=float(_require(case, "input_bytes")),
            )
        )
    setup_error = report.get("error")
    return BenchReport(
        op=op_name,
        tier=tier,
        protocol=protocol,
        baseline=str(_require(report, "performance_baseline")),
        cases=tuple(cases),
        error=summarize_error(setup_error) or None,
    )


def from_fair_report(
    op_name: str,
    report: dict[str, Any],
    *,
    baseline: str,
    op=None,
    tier: str = TIER_PAIR,
    protocol: str = PROTOCOL_FAIR,
    candidate_name: str = "candidate",
) -> BenchReport:
    """Read a ``fair.run_fair_benchmarks`` report.

    ``baseline`` names the baseline provider, whose per-case block is keyed by
    that name. Deliberately required rather than inferred as "whichever
    provider is not the candidate": a wrong guess would report the baseline's
    own retained memory as the candidate's, and the number would look entirely
    plausible.

    ``op`` supplies the saved-memory denominator, which this protocol does not
    record; without it ``input_bytes`` stays 0 and the ratio reads as 0.
    """
    cases: list[CaseMetrics] = []
    for case in report.get("cases") or []:
        # Always walk from the case, never from an already-extracted provider
        # block: the value of a strict reader is the path in its error message,
        # and "missing 'saved_state'" without the provider that lacks it is a
        # worse diagnostic than the one this module replaced.
        def candidate(*path: str, _case=case) -> Any:
            return _require(_case, "providers", candidate_name, *path)

        def reference(*path: str, _case=case) -> Any:
            return _require(_case, "providers", baseline, *path)

        cases.append(
            CaseMetrics(
                dims=dict(_require(case, "dims")),
                dtype=str(_require(case, "dtype")),
                ok=True,  # the fair protocol raises rather than recording a failure
                candidate_forward_ms=float(candidate("forward", "median_ms")),
                candidate_backward_ms=float(candidate("backward", "median_ms")),
                candidate_full_ms=float(candidate("pair_full", "median_ms")),
                baseline_forward_ms=float(reference("forward", "median_ms")),
                baseline_backward_ms=float(reference("backward", "median_ms")),
                baseline_full_ms=float(reference("pair_full", "median_ms")),
                saved_bytes=float(candidate("saved_state", "logical_saved_bytes")),
                input_bytes=declared_input_bytes(op, case),
            )
        )
    return BenchReport(
        op=op_name,
        tier=tier,
        protocol=protocol,
        baseline=baseline,
        cases=tuple(cases),
        environment=dict(report.get("environment") or {}),
    )
