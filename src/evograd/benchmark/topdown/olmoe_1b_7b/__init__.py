"""The OLMoE-1B-7B benchmark workload, organized by level and by tier.

One canonical execution -- ``OlmoeForCausalLM`` built from the published
configuration, batch 2, sequence 2048, BF16, CUDA, SDPA, grouped-GEMM experts,
``model.train()``, ``use_cache=False``, no gradient checkpointing, random
weights from a fixed seed, synthetic tokens. Sixteen decoder layers, each a
multi-head attention block and a routed mixture of 64 SwiGLU experts of which 8
serve every token: 6.92B parameters, 1.28B of them active per token.

* :mod:`.architecture` -- ``config.json`` field by field;
* :mod:`.levels` -- the Level-3 step, the four Level-2 sites, the Level-1
  bindings;
* :mod:`.layer` -- layer 8, captured or config-derived: where the Level-2
  sites take their real invocations from;
* :mod:`.harvest` -- the observed whole-model step frozen into
  ``harvest/snapshot.json``.

Transformers is optional at import. Nothing in this package imports
:mod:`evograd.evaluation`; patching and judging is the Tier-3 adapter's
(:mod:`evograd.evaluation.tier3.workloads.olmoe_1b_7b`).

    PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b --out report.json
"""

from .architecture import (
    BLOCK_KIND,
    MODEL_NAME,
    OLMOE_1B_7B,
    REPRESENTATIVE_LAYER,
    REPRESENTATIVE_LAYERS,
    layer_kind,
)
from .levels.level3.report import SmokeReport
from .levels.level3.spec import CANONICAL, WorkloadSpec, WorkloadSpecError

__all__ = ["BLOCK_KIND", "CANONICAL", "MODEL_NAME", "OLMOE_1B_7B", "REPRESENTATIVE_LAYER",
           "REPRESENTATIVE_LAYERS", "SmokeReport", "WorkloadSpec", "WorkloadSpecError",
           "layer_kind"]

# There is deliberately no ``Workload`` accessor here: patching a model and
# judging what comes back is evaluation's. The canonical import is
# ``evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload``.
