"""Verdicts, findings and JSON-safe reports shared by every gate.

A verification produces exactly one of three statuses:

``passed``      every required check ran and none failed;
``failed``      the candidate broke a contract or exceeded a limit;
``incomplete``  some required evidence is missing (an invalid reference, no
                policy, no cases, a check that could not run).

A report keeps every :class:`Finding` it saw. When a candidate both failed and
left evidence missing, the status is ``failed`` and ``complete`` is ``False``;
neither fact hides the other. ``report-first`` collection is a caller decision
(:func:`continue_after`) and never rewrites the status.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

PASSED = "passed"
FAILED = "failed"
INCOMPLETE = "incomplete"
STATUSES = (PASSED, FAILED, INCOMPLETE)

#: The two outcomes a finding can carry. A finding never says "passed".
OUTCOMES = (FAILED, INCOMPLETE)

STRICT = "strict"
REPORT_FIRST = "report_first"


@dataclass(frozen=True)
class Finding:
    """One reason a check did not pass, and whose problem it is.

    ``outcome`` is ``failed`` when the candidate is at fault and ``incomplete``
    when the evidence needed to judge it is missing or invalid.
    """

    code: str
    outcome: str
    message: str
    subject: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"finding outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if not self.code:
            raise ValueError("a finding needs a code")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "outcome": self.outcome,
                               "message": self.message}
        if self.subject is not None:
            out["subject"] = self.subject
        if self.details:
            out["details"] = dict(self.details)
        return out


def failure(code: str, message: str, subject: str | None = None, **details: Any) -> Finding:
    return Finding(code, FAILED, message, subject, details or None)


def missing(code: str, message: str, subject: str | None = None, **details: Any) -> Finding:
    return Finding(code, INCOMPLETE, message, subject, details or None)


def summarize(findings: Iterable[Finding]) -> tuple[str, bool]:
    """``(status, complete)`` for a set of findings.

    Failure outranks incompleteness for the status; completeness is reported
    separately so a failed run that also lacked evidence says both.
    """
    findings = list(findings)
    failed = any(f.outcome == FAILED for f in findings)
    complete = not any(f.outcome == INCOMPLETE for f in findings)
    if failed:
        return FAILED, complete
    return (PASSED if complete else INCOMPLETE), complete


@dataclass
class ResultCheck:
    """One required result of one case: its metrics, its limits, its verdict."""

    name: str
    kind: str
    metrics: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    #: ``False`` when the numeric comparison did not run (contract failure,
    #: invalid reference, missing result). Metrics are then absent, not zero.
    compared: bool = False

    @property
    def status(self) -> str:
        status, complete = summarize(self.findings)
        if status == PASSED and not self.compared:
            # Every result check either compares or says why it could not.
            return INCOMPLETE
        return status

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "status": self.status,
                "compared": self.compared, "metrics": dict(self.metrics),
                "limits": dict(self.limits), "contract": dict(self.contract),
                "findings": [f.to_dict() for f in self.findings]}


@dataclass
class CaseReport:
    """One case (Gate 1 workload/variant, Gate 2 sample): every result and why."""

    case_id: str
    results: list[ResultCheck] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    #: ``required`` cases decide the verdict; a ``probe`` whose reference is
    #: outside the valid domain is recorded as skipped and decides nothing.
    role: str = "required"
    skipped: str | None = None

    def all_findings(self) -> list[Finding]:
        out = list(self.findings)
        for result in self.results:
            out.extend(result.findings)
            if not result.findings and not result.compared:
                out.append(missing("not_compared", "the result was not compared",
                                   subject=result.name))
        return out

    @property
    def status(self) -> str:
        if self.skipped is not None:
            return "skipped"
        return summarize(self.all_findings())[0]

    @property
    def complete(self) -> bool:
        if self.skipped is not None:
            return False
        return summarize(self.all_findings())[1]

    def result(self, name: str) -> ResultCheck | None:
        for result in self.results:
            if result.name == name:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "role": self.role, "status": self.status,
                "complete": self.complete, "skipped": self.skipped,
                "results": [r.to_dict() for r in self.results],
                "findings": [f.to_dict() for f in self.findings],
                "evidence": dict(self.evidence)}


@dataclass
class VerificationReport:
    """What one gate concluded about one candidate, and every piece of why."""

    gate: str
    protocol: str
    subject: dict[str, Any]
    cases: list[CaseReport] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def decisive_cases(self) -> list[CaseReport]:
        return [case for case in self.cases if case.skipped is None]

    def all_findings(self) -> list[Finding]:
        out = list(self.findings)
        decisive = self.decisive_cases()
        if not decisive:
            out.append(missing("no_cases", "no case was checked; an empty set never passes"))
        for case in decisive:
            out.extend(case.all_findings())
        return out

    @property
    def status(self) -> str:
        return summarize(self.all_findings())[0]

    @property
    def complete(self) -> bool:
        return summarize(self.all_findings())[1]

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    def counts(self) -> dict[str, int]:
        out = {PASSED: 0, FAILED: 0, INCOMPLETE: 0, "skipped": 0}
        for case in self.cases:
            out[case.status] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        findings = self.all_findings()
        status, complete = summarize(findings)
        payload = {
            "gate": self.gate, "protocol": self.protocol, "status": status,
            "complete": complete, "subject": self.subject,
            "case_counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
            "incomplete_because": sorted({f.code for f in findings if f.outcome == INCOMPLETE}),
            "failed_because": sorted({f.code for f in findings if f.outcome == FAILED}),
            "cases": [c.to_dict() for c in self.cases],
            "evidence": self.evidence,
        }
        return jsonable(payload)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, **kwargs)


def continue_after(report: VerificationReport, mode: str = STRICT) -> bool:
    """Whether a caller should keep collecting data (e.g. timing) after ``report``.

    ``strict`` continues only after a pass. ``report_first`` continues whenever
    the candidate executed, so a measurement exists beside its failure. The
    report's status is not touched either way.
    """
    if mode == STRICT:
        return report.passed
    if mode == REPORT_FIRST:
        return not any(f.code in NOT_EXECUTED for f in report.all_findings())
    raise ValueError(f"unknown continuation mode {mode!r}; use {STRICT!r} or {REPORT_FIRST!r}")


#: Findings meaning the candidate never produced results to measure.
NOT_EXECUTED = frozenset({"candidate_runtime_error", "output_structure", "build_failed"})


def jsonable(value: Any, *, _path: str = "$", _removed: list | None = None) -> Any:
    """A JSON-safe copy: tensors become metadata, non-finite floats become ``None``.

    A non-finite float is never written as ``NaN``/``Infinity``. Where one was
    replaced the top-level mapping carries ``non_finite_values_removed`` with
    its paths, so a replaced number cannot pass for a measurement.
    """
    top = _removed is None
    removed: list[str] = [] if top else _removed
    out = _jsonable(value, _path, removed)
    if top and removed and isinstance(out, dict):
        out["non_finite_values_removed"] = removed
    return out


def _jsonable(value: Any, path: str, removed: list[str]) -> Any:
    import torch

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            removed.append(path)
            return None
        return value
    if isinstance(value, Finding):
        return _jsonable(value.to_dict(), path, removed)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict(), path, removed)
    if torch.is_tensor(value):
        if value.numel() == 1 and not value.is_floating_point():
            return {"tensor": _meta(value), "value": value.item()}
        return {"tensor": _meta(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, f"{path}.{k}", removed) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(v, f"{path}[{i}]", removed) for i, v in enumerate(items)]
    return repr(value)


def _meta(tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "device": tensor.device.type}
