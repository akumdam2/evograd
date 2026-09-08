"""The Meta-Llama-3-8B benchmark workload, organized by level and by tier.

One canonical execution -- Llama-3-8B, batch 2, sequence 2048, BF16, CUDA, SDPA,
``model.train()``, ``use_cache=False``, no gradient checkpointing, deterministic
seed -- run as::

    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    loss.backward()

with no optimizer step. Everything in this package derives from that one
execution: :mod:`.harvest` records what it invokes, and :mod:`.levels`
decomposes it.

Weights are randomly initialised from a written-out configuration. Llama-3 is a
gated repository, but nothing here fetches a checkpoint, a config or a
tokenizer, so no Hub token is required.

Transformers is optional. Importing this package never imports it; the failure
appears at
:func:`~evograd.benchmark.topdown.llama3_8b.levels.level4.model.require_transformers`
with the extra to install.

    PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b --out report.json

**State.** Levels 4 through 1 and the tier-3 evaluation half are implemented,
and the shared machinery is Qwen3's. What does not exist is anything that must
be *derived from a run*: there is no tracked ``harvest/snapshot.json``, no
level-3 capture and no tier-3 numerics calibration, because each comes out of
executing the canonical step on a GPU and none has been executed. Every stage
that consumes one of those refuses by name rather than substituting a synthetic
stand-in; ``README.md`` lists the three commands, in the order they unblock each
other.
"""

from .levels.level4.report import SmokeReport
from .levels.level4.spec import (
    CANONICAL,
    LLAMA_3_8B,
    WorkloadSpec,
    WorkloadSpecError,
)

__all__ = [
    "CANONICAL",
    "LLAMA_3_8B",
    "SmokeReport",
    "WorkloadSpec",
    "WorkloadSpecError",
]


# There is deliberately no ``Workload`` accessor here. Patching a model and
# judging what comes back is evaluation's, and a convenience re-export would
# make this package import it -- the dependency the layering test forbids, and
# not one that becomes acceptable by being deferred, dynamic or string-based.
# The canonical import is
# ``evograd.evaluation.tier3.workloads.llama3_8b.workload``.
