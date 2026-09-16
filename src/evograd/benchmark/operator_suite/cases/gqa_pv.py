"""Operator-suite performance case for ``gqa_pv``.

Derived from the attention decomposition of Qwen3-0.6B's observed
``qwen3_attention`` boundary (causal GQA SDPA + output projection, batch 2 x
2048 tokens, bf16): the scores -> causal softmax -> PV split of that one
observed call. The dims are re-derived from the published configuration
(``hf_config`` provenance), and the note says what this is: a case derived from
a decomposition, NOT an operator the Level-4 harvest observed on its own. It
therefore joins the contract here, as an operator-suite case under a
``*_derived`` suite name, and never through the model's observed bindings --
the public ``qwen3_0_6b_observed`` suites are untouched.
"""

from __future__ import annotations

from evograd.benchmark.operator_suite.cases import SuiteCases
from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import QWEN3_0_6B

#: The observed attention call this primitive is one third of.
DERIVED_FROM = "qwen3_attention"

_NOTE = (
    "derived from the attention decomposition (scores -> causal softmax -> PV) "
    "of the observed qwen3_attention call at batch 2 x seq 2048; not an "
    "independently observed operator in the Qwen3-0.6B capture"
)

_BENCHMARK = (
    Workload(
        dims=QWEN3_0_6B.causal_gqa_sdpa_dims(batch=2, seq=2048),
        dtype="bfloat16",
        provenance=Provenance(
            model=QWEN3_0_6B.name,
            component="causal_gqa_sdpa",
            free={"batch": 2, "seq": 2048},
            source="hf_config",
            layout="head_major_view",
            note=_NOTE,
        ),
    ),
)

cases = SuiteCases(
    benchmark=_BENCHMARK,
    coverage=_BENCHMARK,
    suites={"qwen3_0_6b_attention_derived": _BENCHMARK},
)
