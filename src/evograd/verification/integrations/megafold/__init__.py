"""MegaFold: Gate 1, operator Gate 2 and model Gate 2 on real data of one frozen PDB structure.

The workload is the tier-3 MegaFold training step (eager tree, ``configs/
megafold_1x1.yaml``, one recycling pass, confidence head called) on
7a4d-assembly1 featurized once by MegaFold's own training featurization
(:mod:`evograd.benchmark.topdown.megafold.data`). Every data order runs that
same structure, its step's randomness (dropout, diffusion noise,
augmentations) seeded by the order: every site's shapes follow the structure's
token, atom and MSA counts, so one structure is what makes a case the same
case in every order.

Two pins exist so that a gradient reaches every recorded call. The weights
are a frozen warm start (``integrations.warm_start``: 200 updates of
MegaFold's own trainer recipe on the structure, seeds of order 909): at the
initialization the zero-initialized output layers keep every gradient from
the MSA module, the diffusion module and the input embedder. The confidence
head is called on every step (MegaFold's trainer calls it on every tenth),
because a step that does not call it gives its sites no gradient. The
template embedder and the recycling projections get none in this step by
construction and hold no sites (``workload.NOT_SITES``).

The corpus keeps a declared subset of each case's calls (:mod:`.capture`).
Model Gate 2 runs on the warm start's forward and backward of the structure at
each data order: under the verification settings that step is bit-for-bit
repeatable (loss, distogram logits, every gradient), which its native repeats
check. Gate 3 does not apply (``gate3_applies``; user decision 2026-10-27: a
1000-update run of this step is ~2.3 h), so a Level-2 candidate and the
combination are accepted on model Gate 2.

    python -m evograd.verification.integrations megafold {capture,fidelity,verify} ...
"""

from __future__ import annotations

from evograd.benchmark.topdown.megafold import data as _data

from ..base import Model
from ..objectives import PrimaryOutput
from .capture import MegaFoldCapture, invocations, recorded_calls
from .reference import CONTRACT, MODELING_MODULES, reference_op, source_identity

WORKLOAD_CONFIG = {
    "model": "megafold",
    "megafold": "Supercomputing-System-AI-Lab/MegaFold@845905e7c707cf4004404354156df97f2efc4fde",
    "config": "configs/megafold_1x1.yaml",
    "sample": _data.SAMPLE,
    #: The frozen featurization every capture and gate runs on (``pool.json``).
    "sample_sha256": "611d1c75a2842e6e09654f04d851cc24b579007b26982a5b26972d5e46383104",
    "batch": 1, "residues": 384, "recycling": 1, "confidence_head": True, "init_seed": 0,
    "weights": "warm start: 200 updates of MegaFold's trainer recipe on the pinned structure, "
               "seeds of order 909",
    "init_checkpoint": str(_data.warm_start_path()),
    #: The warm start's file hash (``init_state.load``), pinned like the sample's.
    "init_checkpoint_sha256": "abf6060a3cc9f42b6b6da6b678a0495e15ae18d547ac5e948f54de89396bbf79",
    "orders": "the pinned structure at every order; the order seeds the step",
    "recording": "per case and order: first, middle and last training call; first only above 2 GiB",
}


def load_workload(*, warm: bool = True):
    """The pinned workload; ``warm`` False starts from the initialization (the warm start's own run)."""
    from evograd.benchmark.topdown.megafold import workload as declaration
    from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

    (case,) = declaration.benchmark
    workload = make_workload(case, device="cuda", seed=WORKLOAD_CONFIG["init_seed"],
                             include_confidence_head=WORKLOAD_CONFIG["confidence_head"],
                             init_checkpoint=WORKLOAD_CONFIG["init_checkpoint"] if warm else None)
    if warm:
        sha = workload.initial_state()[1]["sha256"]
        if sha != WORKLOAD_CONFIG["init_checkpoint_sha256"]:
            raise ValueError(f"{WORKLOAD_CONFIG['init_checkpoint']} is {sha}, not the pinned "
                             f"{WORKLOAD_CONFIG['init_checkpoint_sha256']}")
    return workload


def train_batches(workload, orders):
    from evograd.verification.common.contracts import stable_hash

    data = workload.data_identity()
    if data["sample_sha256"] != WORKLOAD_CONFIG["sample_sha256"]:
        raise ValueError(f"the pool's {data['sample']} is {data['sample_sha256']}, not the pinned "
                         f"{WORKLOAD_CONFIG['sample_sha256']}")
    out = {order: workload.batch_for(seed=order) for order in orders}
    identity = {"data": data, "rule": "workload.batch_for(seed=order)",
                "batches": {str(order): {"batch_sha256": stable_hash(
                    {"sample_sha256": data["sample_sha256"], "seed": order})} for order in orders}}
    return out, identity


def _objective() -> PrimaryOutput:
    from evograd.evaluation.tier3.workloads.megafold.workload import MegaFoldWorkload

    return PrimaryOutput(MegaFoldWorkload.primary_output_name, MegaFoldWorkload.objective_definition)


MODEL = Model(name="megafold", suite="megafold_observed", workload_config=WORKLOAD_CONFIG,
              load_workload=load_workload, train_batches=train_batches,
              live_capture=lambda model: MegaFoldCapture(model, _cases()), invocations=invocations,
              reference_op=reference_op, contract=CONTRACT, source_identity=source_identity,
              modeling_modules=MODELING_MODULES, objective=_objective(), gate3_applies=False,
              recorded_calls=recorded_calls)


def _cases():
    from ..base import observed_cases

    return observed_cases(MODEL)
