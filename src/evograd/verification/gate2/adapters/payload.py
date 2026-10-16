"""Operator scope on recorded training payloads.

The same operator module and deployments as :class:`OperatorScope` (native
eager autograd, the artifact's ``DEPLOYMENT_ENTRY``, the native callable under
``torch.compile``); only the inputs differ. A sample is one recorded native
invocation from a real training batch: its argument values, parameter values,
scalar arguments and output cotangents are loaded from the corpus with their
exact layouts, and every branch at a sample (native, its repeats, the compile
control, the candidate) runs on isolated copies of the same payload.

Sample ids carry the corpus, batch, case and call identity. ``Sample.seed``
encodes ``data_order * 10**6 + index_in_step``: unique per invocation (the
operator protocol requires distinct calibration seeds) and disjoint between
batches, so the calibration/holdout separation (by id and by seed) keeps
batches 101/202 apart from 303/404. The
adapter identity binds the corpus content, so a policy calibrated on generated
inputs (or another corpus) is a binding mismatch here.
"""

from __future__ import annotations

from typing import Any

from evograd.opdecl.activity import OpDecl, Workload
from evograd.verification.common.execution import Sample
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.payloads.corpus import Corpus, CorpusError
from evograd.verification.payloads.gate1 import INPUT_KIND, check_payload_interface


#: ``Sample.seed = data_order * SEED_STRIDE + index_in_step``.
SEED_STRIDE = 10**6


def load_payload_values(corpus: Corpus, payload_id: str, device: str) -> dict[str, Any]:
    """The loader every real-payload sample goes through (fingerprinted into policies)."""
    return corpus.load(payload_id, device=device)


class PayloadOperatorScope(OperatorScope):
    scope = "operator"

    def __init__(self, op: OpDecl, workload: Workload, corpus: Corpus, case_id: str, *,
                 device: str = "cuda", autocast_dtype: str | None = None):
        super().__init__(op, workload, device=device, autocast_dtype=autocast_dtype)
        self.corpus = corpus
        self.case_id = case_id
        self._cached: tuple[str | None, dict[str, Any] | None] = (None, None)

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "inputs": INPUT_KIND, "case_id": self.case_id,
                "corpus": self.corpus.binding()}

    def reference_parts(self) -> dict[str, Any]:
        parts = super().reference_parts()
        parts["inputs"] = load_payload_values
        parts["sampling"] = load_payload_values
        return parts

    def samples(self, split: str) -> list[Sample]:  # type: ignore[override]
        """The case's payloads in ``split``; refuses an empty or mismatched split."""
        records = self.corpus.payloads(case_id=self.case_id, split=split)
        if not records:
            raise CorpusError(f"{self.corpus.corpus_id}: no {split!r} payload for {self.case_id}")
        problems = [p for r in records for p in check_payload_interface(self.op, r)]
        if problems:
            raise CorpusError("; ".join(problems[:5]))
        out = []
        for r in records:
            batch = r.get("batch") or {}
            index = int((r.get("call") or {}).get("index_in_step", 0))
            if not 0 <= index < SEED_STRIDE:
                raise CorpusError(f"{r['payload_id']}: invocation index {index} out of range")
            out.append(Sample(f"{self.corpus.corpus_id}/{r['payload_id']}",
                              int(batch["order"]) * SEED_STRIDE + index,
                              {"inputs": INPUT_KIND, "payload_id": r["payload_id"], "split": split,
                               "batch_order": int(batch["order"]),
                               "batch_sha256": batch.get("sha256"),
                               "call": (r.get("call") or {}).get("path")}))
        return out

    def values(self, sample: Sample) -> dict[str, Any]:
        payload_id = sample.config.get("payload_id")
        if payload_id is None:
            raise CorpusError(f"sample {sample.sample_id} is not a real-payload sample")
        cached_id, cached = self._cached
        if cached_id != payload_id:
            # One payload resident at a time: a split never sits in device memory at once.
            self._cached = (None, None)
            cached = load_payload_values(self.corpus, payload_id, self.device)
            self._cached = (payload_id, cached)
        return cached
