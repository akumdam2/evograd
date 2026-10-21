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
MODEL = make_model("torchrec_dlrm", WORKLOAD_CONFIG)
