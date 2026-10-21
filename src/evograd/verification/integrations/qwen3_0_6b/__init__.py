"""Qwen3-0.6B: real training payloads and the three gates on real data.

* :mod:`.capture` -- record every L1 call and L2 boundary of a real training
  step as payloads of the model's observed cases;
* :mod:`.reference` -- the model's own computation, the reference every gate uses;
* :mod:`.gates` -- Gate 1, Gate 2 (operator and model scope) and Gate 3 for
  candidate artifacts.

    python -m evograd.verification.integrations.qwen3_0_6b {capture,fidelity,verify} ...
"""
