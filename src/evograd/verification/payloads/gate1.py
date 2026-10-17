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
                  device: str = "cuda") -> list:
    """Every payload of ``case_id`` in ``split`` as a required Gate 1 case.

    Raises :class:`CorpusError` when the split holds no payload for the case or
    a payload does not match the declaration's interface: the caller reports a
    setup failure; there is no fallback to generated inputs.
    """
    from evograd.verification.gate1.runner import REQUIRED_CASE, Gate1Case

    records = corpus.payloads(case_id=case_id, split=split)
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
