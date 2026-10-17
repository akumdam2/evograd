"""Gate 1 -- local pair verification against the declared PyTorch oracle.

    from evograd.verification import gate1
    report = gate1.verify_pair(op, seed_module, device="cpu")
    report.status            # "passed" | "failed" | "incomplete"
"""

from evograd.verification.gate1.adapters import (
    as_pair,
    declared_values,
    evolution_correctness,
    generated_case,
    pair_from_baseline,
    pair_from_module,
    pair_from_provider,
    saved_state_observer,
    stress_probes,
    upstream_variants,
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
    declared_cases,
    result_contract,
    verify_pair,
)

__all__ = [
    "ORACLE", "PROBE_CASE", "PROTOCOL", "RECORD", "REJECT", "REQUIRED_CASE", "SMOKE",
    "Gate1Case", "PairUnderTest", "as_pair", "declared_cases", "declared_values",
    "evolution_correctness", "generated_case", "pair_from_baseline", "pair_from_module", "pair_from_provider",
    "result_contract", "saved_state_observer", "stress_probes", "upstream_variants",
    "verify_pair",
]
