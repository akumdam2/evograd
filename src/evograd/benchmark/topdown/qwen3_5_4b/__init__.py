"""The Qwen3.5-4B benchmark workload (text model), organized by level and by tier.

One canonical execution -- ``Qwen3_5ForCausalLM`` built from the published
``text_config``, batch 1, sequence 2048, BF16, CUDA, SDPA, ``model.train()``,
``use_cache=False``, no gradient checkpointing, random weights from a fixed
seed, synthetic tokens. Twenty-four of its 32 layers are Gated DeltaNet (linear
attention) and eight gated full attention (every fourth); every layer has a
dense SwiGLU MLP. Batch 1 rather than the repository's usual 2 because the eager
training step at B2 does not fit one 96 GB card (see
:mod:`.levels.level3.spec`).

* :mod:`.architecture` -- ``text_config`` field by field;
* :mod:`.levels` -- the Level-3 step, the seven Level-2 sites, the Level-1
  bindings;
* :mod:`.layer` -- layers 14 (GDN) and 15 (full attention), captured or
  config-derived: where the Level-2 sites take their real invocations from;
* :mod:`.harvest` -- the observed whole-model step frozen into
  ``harvest/snapshot.json``.

Transformers is optional at import. Nothing in this package imports
:mod:`evograd.evaluation`; patching and judging is the Tier-3 adapter's
(:mod:`evograd.evaluation.tier3.workloads.qwen3_5_4b`).

    PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b --out report.json
"""

from .architecture import (
    MODEL_NAME,
    QWEN3_5_4B,
    REPRESENTATIVE_LAYERS,
    layer_kind,
    layer_types,
)
from .levels.level3.report import SmokeReport
from .levels.level3.spec import CANONICAL, WorkloadSpec, WorkloadSpecError

__all__ = ["CANONICAL", "MODEL_NAME", "QWEN3_5_4B", "REPRESENTATIVE_LAYERS", "SmokeReport",
           "WorkloadSpec", "WorkloadSpecError", "layer_kind", "layer_types"]

# There is deliberately no ``Workload`` accessor here: patching a model and
# judging what comes back is evaluation's. The canonical import is
# ``evograd.evaluation.tier3.workloads.qwen3_5_4b.workload``.
