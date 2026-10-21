"""Real training payloads: recorded operator invocations as verification inputs.

    corpus.py   versioned, immutable, sharded corpus (exact layouts, content hashes, splits)
    gate1.py    Gate 1 cases from a corpus (lazy, never mixed with generated inputs)
    cli.py      ``python -m evograd.verification.payloads.cli gate1 ...``: one artifact, one split

A model workload provides the capture (for Qwen3-0.6B:
``evograd.verification.integrations.qwen3_0_6b``). The Gate 2 operator scope on
a corpus is ``evograd.verification.gate2.adapters.payload.PayloadOperatorScope``.

Input provenance vocabulary used in reports:

* model-derived configuration -- dims, dtypes and layouts taken from the model
  (the workload index); values may still be generated;
* real training payload -- values, parameters and cotangents recorded from a
  native training step on real TRAIN batches (this package);
* synthetic diagnostic -- generated values or structured cotangents, labelled
  as such and never part of a real-payload verdict;
* whole-model execution -- the model itself run on real batches (Gate 2 model
  scope, Gate 3).
"""

from evograd.verification.payloads.corpus import Corpus, CorpusError, CorpusWriter, UnsupportedLayout
from evograd.verification.payloads.gate1 import INPUT_KIND, PayloadValues, coverage, payload_cases

__all__ = ["Corpus", "CorpusError", "CorpusWriter", "INPUT_KIND", "PayloadValues", "UnsupportedLayout",
           "coverage", "payload_cases"]
