"""Gate 1 -- local pair verification against the model reference on recorded calls.

    from evograd.verification import gate1
    from evograd.verification.payloads import payload_cases
    cases = payload_cases(op, workload, corpus, case_id=..., split="acceptance")
    report = gate1.verify_pair(op, seed_module, cases=cases)
    report.status            # "passed" | "failed" | "incomplete"
"""

from evograd.verification.gate1.adapters import (
    as_pair,
    evolution_correctness,
    pair_from_baseline,
    pair_from_module,
    pair_from_provider,
    saved_state_observer,
)
from evograd.verification.gate1.runner import (
    ORACLE,
    PROBE_CASE,
    PROTOCOL,
    RECORD,
    REJECT,
    REQUIRED_CASE,
    SMOKE,
    Gate1Case,
    PairUnderTest,
    result_contract,
    verify_pair,
)

__all__ = [
    "ORACLE", "PROBE_CASE", "PROTOCOL", "RECORD", "REJECT", "REQUIRED_CASE", "SMOKE",
    "Gate1Case", "PairUnderTest", "as_pair", "evolution_correctness", "pair_from_baseline",
    "pair_from_module", "pair_from_provider", "result_contract", "saved_state_observer",
    "verify_pair",
]
