"""Two questions a Tier-3 gate answers, kept apart.

    1. Did the numerical comparisons meet their declared or frozen limits?
    2. May this provider go on to training and timing?

They used to be one question: any failure stopped the run, so a kernel that
disagreed with the reference in the last bfloat16 place was never measured end
to end, and "failed" meant both "wrong" and "not measured". That conflation is
what this module removes. A verdict now carries a *numerical* answer and an
*execution* answer, and a mode decides which findings block:

``strict`` (the historical behaviour, and the default)
    every finding blocks. A numerical mismatch stops the provider before
    training and timing, exactly as before.

``report_first``
    a finding whose kind is numerical (a finite comparison that exceeded its
    limit) or unavailable (a comparison that could not be made) is *recorded*
    and does not stop execution. Everything else still stops it.

What never becomes a numerical finding, in either mode, because none of it is
a question about tolerance:

    structural  a required gradient or output missing, a shape or dtype
                mismatch, parameter misalignment, patch coverage or invocation
                counts, purity / input mutation, permissions.
    execution   a non-finite output, gradient or loss; an exception; a crash.
    unavailable a policy that does not bind, a missing holdout verdict, a
                comparison that could not be run. Never reported as a pass.

The numerical verdict is preserved verbatim whichever mode is active: nothing
here can turn a numerical failure into a numerical pass. ``report_first`` only
changes whether the runner is allowed to continue past it.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Mapping

STRICT = "strict"
REPORT_FIRST = "report_first"
MODES = (STRICT, REPORT_FIRST)
#: Historical behaviour is the default; report-first is opt-in, per run.
DEFAULT_MODE = STRICT
ENV = "EVOGRAD_TIER3_ENFORCEMENT"
_ALIASES = {"report-first": REPORT_FIRST, "reportfirst": REPORT_FIRST, "": DEFAULT_MODE}

#: Finding kinds.
NUMERICAL = "numerical"
UNAVAILABLE = "unavailable"
STRUCTURAL = "structural"
EXECUTION = "execution"
KINDS = (NUMERICAL, UNAVAILABLE, STRUCTURAL, EXECUTION)

#: Kinds that stop a provider whatever the mode.
ALWAYS_BLOCKING = frozenset({STRUCTURAL, EXECUTION})

DESCRIPTION = {
    STRICT: ("every finding blocks: a numerical mismatch stops the provider before "
             "training and timing (historical behaviour)"),
    REPORT_FIRST: ("numerical mismatches and unavailable comparisons are recorded and do "
                   "not stop training or timing; structural and execution failures still do"),
}


def mode(name: str | None) -> str:
    key = _ALIASES.get((name or "").strip().lower().replace("-", "_"),
                       (name or "").strip().lower().replace("-", "_") or DEFAULT_MODE)
    if key not in MODES:
        raise ValueError(f"unknown Tier-3 enforcement mode {name!r}; known: {MODES}")
    return key


def active_mode() -> str:
    """The mode selected for this process (``EVOGRAD_TIER3_ENFORCEMENT``).

    Read on every call so a parent CLI can select it once and every isolated
    child process it spawns inherits the same choice.
    """
    return mode(os.environ.get(ENV))


def set_active_mode(name: str | None) -> str:
    chosen = mode(name)
    os.environ[ENV] = chosen
    return chosen


def add_enforcement_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--numerical-enforcement", dest="numerical_enforcement", default=None,
        choices=("strict", "report-first", REPORT_FIRST),
        help=("what a numerical mismatch does at Tier 3. 'strict' (default): it stops the "
              "provider before training and timing, as before. 'report-first': finite "
              "numerical mismatches and unavailable comparisons are recorded in full and "
              "execution continues, so an end-to-end training and timing result exists "
              "beside them; missing gradients, non-finite values, purity, permissions, "
              "patch coverage and runtime failures still stop the provider. The numerical "
              "verdict itself is identical in both modes"))


def apply_enforcement_argument(args: argparse.Namespace) -> str:
    """Select the mode this run asked for, and export it to child processes."""
    requested = getattr(args, "numerical_enforcement", None)
    if requested:
        return set_active_mode(requested)
    return active_mode()


def finding(stage: str, kind: str, reason: str, **extra: Any) -> dict[str, Any]:
    """One recorded problem: which stage, what kind, why."""
    if kind not in KINDS:
        raise ValueError(f"unknown finding kind {kind!r}; known: {KINDS}")
    return {"stage": stage, "kind": kind, "reason": reason, **extra}


def boundary_finding_kind(report: Mapping[str, Any]) -> str:
    """What kind of problem a live-boundary report found: numerical, or not.

    Anything other than a finite tolerance disagreement -- a shadow that raised,
    a non-finite result, wrong invocation coverage, a duplicate record, a shared
    parameter, a lost record -- is not a tolerance question and must not be
    waved through as one. Unrecognised shapes fall through to structural.
    """
    if report.get("errors") or report.get("non_finite_results"):
        return EXECUTION
    if (report.get("missing_or_extra") or report.get("unexpected_sites")
            or report.get("duplicate_ids") or report.get("record_desync")
            or report.get("shared_parameter_boundaries")):
        return STRUCTURAL
    if report.get("numerical_only"):
        return NUMERICAL
    return STRUCTURAL


def preflight_numerical_only(preflight: Mapping[str, Any]) -> bool:
    """Did the tier-1 gate fail on tolerance alone, or did a case raise?

    Without the structured detail the answer is unknown, and unknown is treated
    as an execution failure: a kernel that raised must never be recorded as a
    numerical disagreement.
    """
    detail = preflight.get("detail")
    failures = (detail or {}).get("failures") if isinstance(detail, Mapping) else None
    if not failures:
        return False
    return all(not case.get("error") and case.get("failed") for case in failures)


def blocks(kind: str, active: str | None = None) -> bool:
    """Does a finding of this kind stop the provider under this mode?"""
    active = active or active_mode()
    if kind in ALWAYS_BLOCKING:
        return True
    return active == STRICT


def summarize(findings: Iterable[dict[str, Any]], *, gate: str, active: str | None = None,
              **detail: Any) -> dict[str, Any]:
    """Fold findings into the two answers, plus the compatibility fields.

    ``ok`` keeps its old meaning *for the active mode*: may this provider be
    timed. In ``strict`` that is the historical verdict unchanged. The numerical
    answer is always reported separately and is not affected by the mode.
    """
    active = active or active_mode()
    found = list(findings)
    # `waived` is an explicit, recorded exception (today: the historical
    # --protocol4-diagnostic-timing escape). It never changes the numerical
    # answer below; it only says this finding was not allowed to stop the run.
    blocking = [f for f in found if blocks(f["kind"], active) and not f.get("waived")]
    numerical = [f for f in found if f["kind"] == NUMERICAL]
    unavailable = [f for f in found if f["kind"] == UNAVAILABLE]
    structural = [f for f in found if f["kind"] in ALWAYS_BLOCKING]

    if numerical:
        status = "mismatches_recorded"
        numerical_ok: bool | None = False
    elif unavailable:
        status = "unavailable"
        numerical_ok = None
    else:
        status = "within_limits"
        numerical_ok = True

    execution_ok = not structural
    # Report-first lets an unavailable comparison continue, which is right for
    # execution and wrong for the claim: a run that could not make a required
    # comparison has not been fully checked, whatever it managed to measure.
    # This flag is what a report must read before calling an evaluation
    # complete; `ok` only ever meant "admissible to training and timing".
    evaluation_complete = execution_ok and not unavailable
    verdict: dict[str, Any] = {
        "gate": gate,
        "enforcement": {"mode": active, "rule": DESCRIPTION[active],
                        "blocking_kinds": sorted(ALWAYS_BLOCKING) if active == REPORT_FIRST else sorted(KINDS)},
        # `ok` = admissible to training and timing under the active mode.
        "ok": not blocking,
        "execution_ok": execution_ok,
        "execution_failed_at": structural[0]["stage"] if structural else None,
        "numerical_ok": numerical_ok,
        "numerical_status": status,
        "numerical_failed_at": numerical[0]["stage"] if numerical else None,
        "numerical_unavailable_at": [f["stage"] for f in unavailable],
        "evaluation_complete": evaluation_complete,
        "evaluation_incomplete_because": [f["stage"] for f in unavailable] or None,
        "failed_at": blocking[0]["stage"] if blocking else None,
        "reason": blocking[0]["reason"] if blocking else None,
        "findings": found,
        "counts": {k: sum(1 for f in found if f["kind"] == k) for k in KINDS},
    }
    verdict.update(detail)
    return verdict


__all__ = [
    "ALWAYS_BLOCKING", "DEFAULT_MODE", "DESCRIPTION", "ENV", "EXECUTION", "KINDS",
    "MODES", "NUMERICAL", "REPORT_FIRST", "STRICT", "STRUCTURAL", "UNAVAILABLE",
    "active_mode", "add_enforcement_argument", "apply_enforcement_argument",
    "blocks", "boundary_finding_kind", "finding", "mode", "preflight_numerical_only",
    "set_active_mode", "summarize",
]
