"""What is left of tier 3's whole-model gate: its enforcement mode.

The gate itself is gone; correctness is judged by the verification package's
Gates 1/2/3 on real payloads. :mod:`.enforcement` stays because
:func:`evograd.evaluation.tier3.runner.verification_policy` records the active
mode in every Tier-3 report.
"""
