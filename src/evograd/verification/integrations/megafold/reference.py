"""What every MegaFold gate judges against: MegaFold's own computation at each site.

Each case's task is re-declared under :data:`CONTRACT` with, as its forward,
what the eager MegaFold tree computes at that call (``Attend``'s default path,
``LayernormLinear``'s torch path, ``nn.LayerNorm``, the transition's SwiGLU),
in MegaFold's precision. The two MegaFold contracts already are that
computation; the LayerNorm's runtime spelling is ``F.layer_norm``, which is what
``nn.LayerNorm`` calls; the SwiGLU is spelled as MegaFold spells it, without the
generic contract's float32 SiLU. The capture replays every recorded call
against these bit for bit.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import torch.nn.functional as F

CONTRACT = "evograd-model-reference/megafold/1"


def swiglu(a, b):
    """``FusedTransition``'s activation as MegaFold's eager spelling: ``silu(gates) * x``."""
    return F.silu(a) * b


#: Tasks whose declared forward is not MegaFold's computation.
_FORWARDS = {"swiglu": f"{__name__}:swiglu"}

#: MegaFold modules whose source is the reference (hashed into every identity).
MODELING_MODULES = ("megafold.model.megafold", "megafold.model.attention",
                    "megafold.model.FusedLayernormLinear.fused_layernorm_linear",
                    "megafold.model.FusedTransition.fused_transition", "megafold.utils.model_utils",
                    "megafold.inputs", "evograd.evaluation.tier3.workloads.megafold.workload")


def source_identity() -> dict:
    """The MegaFold checkout the reference follows: its commit and the modeling sources' hashes."""
    import importlib
    import subprocess

    import megafold

    root = Path(megafold.__file__).resolve().parent.parent
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    hashes = {}
    for name in MODELING_MODULES:
        with open(importlib.import_module(name).__file__, "rb") as handle:
            hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    return {"megafold_commit": commit, "sources_sha256": hashes}


def reference_op(task: str):
    """``task`` re-declared under :data:`CONTRACT`, with MegaFold's computation as forward."""
    from evograd.benchmark import get_task

    op = get_task(task)
    model = dataclasses.replace(op, forward=_FORWARDS.get(task) or op.runtime_forward or op.forward,
                                runtime_forward=None, reference_dtype=None, reference_contract=CONTRACT)
    model.validate()
    return model
