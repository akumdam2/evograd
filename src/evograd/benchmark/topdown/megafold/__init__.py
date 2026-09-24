"""Workload declaration: MegaFold training step (level 4).

The whole-model level-4 task: one full MegaFold training step, forward to the
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
MegaFold's hand-written Triton, and the evolved candidates. The level-4
question is whether an evolved kernel's local speedup survives a real training
step, and MegaFold's own kernels are the line it has to clear.

Nothing in this task is evolved. The sites accept the winners of the level-1/2
searches, and measurement is the tier-3 protocol
(``evaluation/tier3/runner.py``): identical weights and batches per provider,
step latency, peak memory, loss agreement, ``cpu_bound_fraction``.

Batches are fully synthetic (seeded random features and labels at a fixed crop
length). That is a requirement, not a shortcut: the protocol compares providers
on identical data, and an MSA/template pipeline would inject variance that is
not the kernels'.
"""

from evograd.opdecl import declare_workload
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import model_workloads

#: Patch sites, and the declared operator each one accepts. Both attention
#: flavours funnel to ``evoattention``: pair-bias and triangle attention share
#: one softmax-attention core in this implementation, at different head widths.
#:
#: ``layernorm_linear`` is a site here and was not one under
#: alphafold3-pytorch, which had no fused module boundary to swap. MegaFold
#: builds ``LayernormLinear`` directly into the tree in more than a dozen
#: places, so the boundary the declaration describes is a real module.
MEGAFOLD_SITE_OPS = {
    "layer_norm": "layernorm",
    "layernorm_linear": "layernorm_linear",
    "transition": "swiglu",
    "pair_bias_attention": "evoattention",
    "triangle_attention": "evoattention",
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
        tuple({"batch": 1, "residues": residues} for residues in AF3_RESIDUE_SWEEP),
        ("float32",),
        note=(
            "MegaFold trains at batch 1 with activation checkpointing; the "
            "residue sweep is its reported crop lengths"
        ),
    ),
    exclusions=(
        "Sites cover what MegaFold's own fused kernels cover: normalization, "
        "the fused LayerNorm+Linear, the SwiGLU transition, and the two "
        "softmax-attention flavours. Everything else inside the timed step "
        "runs stock and dilutes the measured speedup accordingly, and that "
        "dilution is the level-4 result rather than a flaw: triangle "
        "multiplication, outer product mean, MSA pair-weighted averaging, the "
        "diffusion module internals and the confidence head internals are all "
        "unpatched. Attention-weight dropout is disabled for every provider "
        "alike, because no fused-attention contract models it (MegaFold's own "
        "EvoAttention and DS4Sci included) and a dropout firing in one "
        "provider but not another would also desynchronize their RNG streams. "
        "The MSA and template data pipeline is replaced by seeded synthetic "
        "features at a fixed crop, so nothing here measures data loading and "
        "no result speaks to prediction quality. The pocket, contact and "
        "docking constraint channels are present but uniformly masked, which "
        "is what MegaFold writes for an example carrying no constraints and "
        "what nine in ten of its own training examples get. Every token in a "
        "synthetic example is a protein residue, so the ligand binding "
        "affinity head and the nucleotide loss weighting never fire; a random "
        "mixture of molecule types is not a harder case but an incoherent one, "
        "which MegaFold's own affinity pooling rejects. EMA, gradient clipping, "
        "DeepSpeed and sequence parallelism are excluded: the harness trains "
        "one model in one process with AdamW, in the single-rank process "
        "group MegaFold's own one-GPU config runs inside, and sequence "
        "parallelism is pinned to a group size of one, which makes MegaFold's "
        "collectives no-ops rather than removing them from the code path. The recycling "
        "count and whether the confidence head is called are pinned rather "
        "than sampled, where MegaFold draws the first from "
        "randint(1, num_recycling_steps) each step and calls the second every "
        "tenth step. Weights are randomly initialized. Step times are "
        "single-GPU compute per step and are not comparable to the MegaFold "
        "paper's reported throughput, which is a multi-GPU end-to-end number "
        "over the real data pipeline."
    ),
    requires="megafold",
    notes=(
        "The shapes here are declared from MegaFold's configuration, not "
        "harvested from a run: every benchmark case re-derives from the "
        "registered AlphaFold3 configuration, and the workload re-derives them "
        "a second time at build. Weights are randomly initialized from a fixed "
        "seed; the task measures training-step speed and memory, not "
        "prediction quality, and identical weights across providers is what "
        "makes the comparison attributable."
    ),
)
