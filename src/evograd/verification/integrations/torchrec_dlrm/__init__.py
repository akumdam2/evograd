"""DLRM (TorchRec example architecture): the three gates on real Criteo data.

Streamed Criteo Kaggle rows (``GeneratedWorkload(stream=True)``), random
weights as in the benchmark; the capture and model reference are shared with
DiT (:mod:`..generated`, :mod:`..generated_model`). ``embedding_bag`` is a
Level-1 task (Gate 1 and operator Gate 2) even though the model installs it
at a site; ``dlrm_interaction`` gets every gate.

    python -m evograd.verification.integrations torchrec_dlrm {capture,fidelity,verify} ...
"""

from ..generated_model import make_model

WORKLOAD_CONFIG = {"model": "torchrec_dlrm", "device": "cuda", "input_mode": "real",
                   "stream": True}
#: Gate 3 over 50 updates (2026-10-27): native repeats and compile are bitwise equal to
#: native, so the floor (1e-4) is the whole threshold. A float64-screened interaction
#: seed matches native's loss exactly for 75 updates and drifts past 2x the floor after
#: ~100 (5.5x at 125); a 5% fault there is >= 90x the floor from update 10 on.
MODEL = make_model("torchrec_dlrm", WORKLOAD_CONFIG, gate3_steps=50)
