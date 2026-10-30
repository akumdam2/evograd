"""Workload declaration: MegaFold training step (level 3).

The whole-model level-3 task: one full MegaFold training step, forward to the
model's own combined loss, backward, optimizer step, with evolved kernels
patched into the module tree by surgery.

The model is `MegaFold <https://github.com/Supercomputing-System-AI-Lab/MegaFold>`_
(arXiv:2506.20686), which is where every AlphaFold3 shape in
:mod:`evograd.opdecl.models` came from. Using it here closes the provenance
chain at both ends: the level-1/2 operator grids were transcribed from this
model's configuration, and this task runs that same configuration, so
``evoattention``'s benchmark grid and the attention this model executes are the
same widths by construction rather than by coincidence.

MegaFold also supplies the comparison this level exists to make. It ships its
own fused kernels for four of the five sites below, so a run puts three
implementations of one architecture side by side: plain eager PyTorch,
MegaFold's hand-written Triton, and the evolved candidates. The level-3
question is whether an evolved kernel's local speedup survives a real training
step, and MegaFold's own kernels are the line it has to clear.

Nothing in this task is evolved. The sites accept the winners of the level-1/2
searches, and measurement is the tier-3 protocol
(``evaluation/tier3/runner.py``): identical weights and batches per provider,
step latency, peak memory, loss agreement, ``cpu_bound_fraction``.

The data is one real PDB structure (7a4d, 384 tokens after MegaFold's crop),
featurized once by MegaFold's own training featurization and frozen
(:mod:`.data`). One structure, because every site's shapes follow its token,
atom and MSA counts: steps differ by their seed alone (dropout, diffusion noise,
augmentations).
"""

from evograd.opdecl import declare_workload
from evograd.opdecl.models import ALPHAFOLD3
from evograd.ops._common import model_workloads

#: Patch sites, and the task each one accepts. Both attention flavours funnel to
#: ``megafold_evoattention``: pair-bias and triangle attention share ``Attend``,
#: at different head widths. The LayerNorm and the SwiGLU are the generic
#: tasks, because MegaFold computes exactly those there in either precision
#: region; the LayerNorm+Linear and the attention are MegaFold's own
#: mixed-precision contracts (:mod:`.contracts`).
MEGAFOLD_SITE_OPS = {
    "layer_norm": "layernorm",
    "layernorm_linear": "megafold_layernorm_linear",
    "transition": "swiglu",
    "pair_bias_attention": "megafold_evoattention",
    "triangle_attention": "megafold_evoattention",
}

workload = declare_workload(
    name="megafold",
    factory="evograd.evaluation.tier3.workloads.megafold.workload:make_workload",
    family="protein",
    model=ALPHAFOLD3.name,
    sites=dict(MEGAFOLD_SITE_OPS),
    benchmark=model_workloads(
        ALPHAFOLD3,
        "train_step",
        ({"batch": 1, "residues": 384},),
        ("float32",),
        note=(
            "MegaFold trains at batch 1 with activation checkpointing; 384 "
            "residues is its crop and the frozen structure's token count"
        ),
    ),
    exclusions=(
        "Sites cover what MegaFold's own fused kernels cover: normalization, "
        "the fused LayerNorm+Linear, the SwiGLU transition, and the two "
        "softmax-attention flavours. Everything else inside the timed step "
        "runs stock and dilutes the measured speedup accordingly, and that "
        "dilution is the level-3 result rather than a flaw: triangle "
        "multiplication (its combined projection carries a linear bias, which "
        "the LayerNorm+Linear contract does not), outer product mean, MSA "
        "pair-weighted averaging, the diffusion module internals and the "
        "confidence head internals are all unpatched. Attention-weight dropout is disabled for every provider "
        "alike, because no fused-attention contract models it (MegaFold's own "
        "EvoAttention and DS4Sci included) and a dropout firing in one "
        "provider but not another would also desynchronize their RNG streams. "
        "The data is one real structure featurized once and frozen, so nothing "
        "here measures data loading and no result speaks to prediction "
        "quality. Its templates are masked (MegaFold's training never passes a "
        "kalign binary) and it drew no constraints, as nine in ten of "
        "MegaFold's training examples do not. EMA, gradient clipping, "
        "DeepSpeed and sequence parallelism are excluded: the harness trains "
        "one model in one process with AdamW, in the single-rank process "
        "group MegaFold's own one-GPU config runs inside, and sequence "
        "parallelism is pinned to a group size of one, which makes MegaFold's "
        "collectives no-ops rather than removing them from the code path. The recycling "
        "count (one pass) and whether the confidence head is called are "
        "pinned rather than sampled, where MegaFold draws the first from "
        "randint(1, 4) each step and calls the second every tenth step: "
        "verification and the ADBench evaluation call it on every step, so "
        "that its sites get a gradient (the tier-3 CLI leaves it off unless "
        "--confidence-head). The template embedder (its output is masked "
        "without templates) and the recycling projections (multiplied by zero "
        "in the one pass) get no gradient and hold no sites. "
        "Weights are randomly initialized, except where verification and the "
        "ADBench evaluation load their warm start: 200 updates of MegaFold's "
        "own trainer recipe on the structure. Outside MegaFold's own tree the "
        "distogram loss's mean is taken as a sum over the counted pixels, the "
        "same mean without nll_loss2d's atomic reduction, which deterministic "
        "mode refuses. Step times are "
        "single-GPU compute per step and are not comparable to the MegaFold "
        "paper's reported throughput, which is a multi-GPU end-to-end number "
        "over the real data pipeline."
    ),
    requires="megafold",
    notes=(
        "The Level-1/2 cases are harvested from one real training step on the "
        "frozen structure (harvest/snapshot.json); the Level-3 case's dims "
        "re-derive from the registered AlphaFold3 configuration, and the "
        "workload re-derives them a second time at build. Weights are randomly initialized from a fixed "
        "seed; the task measures training-step speed and memory, not "
        "prediction quality, and identical weights across providers is what "
        "makes the comparison attributable."
    ),
)
