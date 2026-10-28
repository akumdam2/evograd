"""Gate 1 cases from recorded training payloads.

Each case is one recorded invocation: the candidate runs its own forward on the
payload's arguments, keeps its own saved state, and runs its own backward on
the payload's recorded output cotangents. The reference (the Gate 1 oracle,
i.e. the pinned model reference under autograd) runs on an isolated copy of
the same payload. Captured outputs and saved tensors are never inputs to the
candidate; only arguments and cotangents are loaded.

Values are materialized when the Gate 1 core runs the case
(:class:`PayloadValues`), so a split of hundreds of invocations never sits in
device memory at once.
"""

from __future__ import annotations

from typing import Any, Mapping

from evograd.opdecl.activity import OpDecl, Workload
from evograd.verification.payloads.corpus import Corpus, CorpusError

#: Where the inputs of a real-payload case came from, as reports record it.
INPUT_KIND = "real_training_payload"


class PayloadValues:
    """A payload's values, loaded on demand (``materialize``) and not cached."""

    def __init__(self, corpus: Corpus, record: Mapping[str, Any], *, device: str):
        self.corpus = corpus
        self.record = dict(record)
        self.device = device

    def materialize(self) -> dict[str, Any]:
        return self.corpus.load(self.record, device=self.device)

    # Mapping-like access for callers that inspect names without loading.
    def keys(self):
        return list(self.record["args"]) + list(self.record["cotangents"])


def required_names(op: OpDecl) -> list[str]:
    return [a.name for a in op.args] + list(op.upstream_grad_names)


def check_payload_interface(op: OpDecl, record: Mapping[str, Any]) -> list[str]:
    """A payload must supply exactly the declaration's arguments and output cotangents."""
    problems = []
    supplied = set(record["args"]) | set(record["cotangents"])
    for name in required_names(op):
        if name not in supplied:
            default = next((getattr(a, "default", None) for a in op.args if a.name == name), None)
            if default is None:
                problems.append(f"payload {record['payload_id']} lacks {name!r}")
    extra = sorted(supplied - set(required_names(op)))
    if extra:
        problems.append(f"payload {record['payload_id']} supplies undeclared {extra}")
    if record.get("task") != op.name:
        problems.append(f"payload {record['payload_id']} is for task {record.get('task')!r}, "
                        f"not {op.name!r}")
    return problems


def payload_cases(op: OpDecl, workload: Workload, corpus: Corpus, *, case_id: str, split: str,
                  device: str = "cuda", payload_ids=None) -> list:
    """Every payload of ``case_id`` in ``split`` (or those of ``payload_ids``) as
    a required Gate 1 case.

    Raises :class:`CorpusError` when the split holds no payload for the case, a
    named payload is not among them, or a payload does not match the
    declaration's interface: the caller reports a setup failure; there is no
    fallback to generated inputs.
    """
    from evograd.verification.gate1.runner import REQUIRED_CASE, Gate1Case

    records = corpus.payloads(case_id=case_id, split=split)
    if payload_ids is not None:
        by_id = {r["payload_id"]: r for r in records}
        missing_ids = [i for i in payload_ids if i not in by_id]
        if missing_ids:
            raise CorpusError(f"corpus {corpus.corpus_id} has no {split!r} payload "
                              f"{missing_ids[:3]} for {case_id!r}")
        records = [by_id[i] for i in payload_ids]
    if not records:
        raise CorpusError(f"corpus {corpus.corpus_id} has no {split!r} payload for {case_id!r}")
    problems = [p for r in records for p in check_payload_interface(op, r)]
    if problems:
        raise CorpusError("; ".join(problems[:5]))
    binding = corpus.binding(split)
    cases = []
    for record in records:
        provenance = {"inputs": INPUT_KIND, "corpus": binding, "payload_id": record["payload_id"],
                      "case": case_id, "batch": record.get("batch"), "call": record.get("call"),
                      "cotangent_stats": record.get("cotangent_stats")}
        cases.append(Gate1Case(record["payload_id"], workload, PayloadValues(corpus, record, device=device),
                               REQUIRED_CASE, provenance))
    return cases


#: The search-feedback rule of a task pack's ``search_correctness``: Gate 1's
#: rule on :func:`search_payload_ids`, with the envelope calibrated for the case.
SEARCH_RULE = "gate1_calibration_payloads"
SEARCH_SELECTION = "first, middle and last call of the case in each calibration order"
#: Set by the ADBench evaluator to the pack's ``task.json`` when its rule is
#: :data:`SEARCH_RULE`; the evolution evaluator's correctness check then calls
#: :func:`search_correctness` instead of judging generated inputs.
SEARCH_CORRECTNESS_ENV = "EVOGRAD_SEARCH_CORRECTNESS"


def search_payload_ids(corpus: Corpus, case_id: str) -> list[str]:
    """The payloads search feedback is judged on (:data:`SEARCH_SELECTION`).

    Only the calibration split: the acceptance payloads stay held out for the
    gates that judge the final submission.
    """
    by_order: dict[Any, list] = {}
    for record in corpus.payloads(case_id=case_id, split="calibration"):
        by_order.setdefault(record["batch"]["order"], []).append(record)
    if not by_order:
        raise CorpusError(f"corpus {corpus.corpus_id} has no calibration payload for {case_id!r}")
    ids = []
    for order in sorted(by_order):
        calls = sorted(by_order[order], key=lambda r: r["call"]["index_in_step"])
        ids += [calls[i]["payload_id"] for i in sorted({0, len(calls) // 2, len(calls) - 1})]
    return ids


def search_correctness(op: OpDecl, module, spec: Mapping[str, Any], *,
                       device: str = "cuda") -> dict[str, Any]:
    """Search feedback for one candidate under ``spec`` (a task pack's
    ``search_correctness``): Gate 1 on the named calibration payloads, in the
    evolution evaluator's correctness summary.

    A report that is not ``passed`` never reads as correct, even when every
    decisive case passed (a report-level finding) or there was none.
    """
    from evograd.verification import gate1

    if spec.get("rule") != SEARCH_RULE:
        raise ValueError(f"not a {SEARCH_RULE!r} specification: {spec.get('rule')!r}")
    corpus = Corpus.open(spec["corpus"]["manifest"])
    if corpus.corpus_id != spec["corpus"]["corpus_id"]:
        raise CorpusError(f"{spec['corpus']['manifest']} is {corpus.corpus_id}, "
                          f"not {spec['corpus']['corpus_id']}")
    cases = payload_cases(op, op.correctness[0], corpus, case_id=spec["case_id"],
                          split="calibration", device=device, payload_ids=spec["payload_ids"])
    report = gate1.verify_pair(op, module, cases=cases, device=device,
                               envelope=spec.get("envelope") or {})
    summary = gate1.evolution_correctness(report, op)
    if report.status != "passed":
        summary["total"] = max(summary["total"], 1)
        summary["passed"] = min(summary["passed"], summary["total"] - 1)
        messages = [f.message for f in report.findings]
        if messages:
            summary["error_message"] = "; ".join(messages[:4])
    return summary


def coverage(corpus: Corpus, *, case_id: str, split: str) -> dict[str, Any]:
    """Exact invocation coverage of one case in one split (for every report)."""
    records = corpus.payloads(case_id=case_id, split=split)
    by_order: dict[str, int] = {}
    roles: dict[str, int] = {}
    zero = []
    for r in records:
        order = str((r.get("batch") or {}).get("order"))
        by_order[order] = by_order.get(order, 0) + 1
        role = (r.get("call") or {}).get("role", "-")
        roles[role] = roles.get(role, 0) + 1
        for name, stats in (r.get("cotangent_stats") or {}).items():
            if stats.get("all_zero") or not stats.get("present", True):
                zero.append(f"{r['payload_id']}:{name}")
    return {"case_id": case_id, "split": split, "split_binding": corpus.binding(split)["split"],
            "payloads": len(records), "by_batch_order": by_order, "by_role": roles,
            "zero_or_absent_cotangents": zero}
