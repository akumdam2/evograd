"""Workload declaration: AlphaFold3 training step (level 4).

The whole-model counterpart of the level-3 ``af3_single_repr_block``: one full
AlphaFold3 training step — forward to the model's own combined loss, backward,
optimizer step — with evolved kernels patched into the module tree by surgery.

The model is `alphafold3-pytorch <https://pypi.org/project/alphafold3-pytorch/>`_,
which is the implementation MegaFold builds on, and MegaFold is where every AF3
shape in :mod:`evograd.opdecl.models` came from. Using it here means the level-4
task trains the same architecture the level-1/2 operator shapes were derived
from, so the provenance chain closes: ``evoattention``'s benchmark grid and the
attention this model executes are the same widths by construction.

Nothing in this task is evolved. The sites below accept the winners of the
level-1/2 searches, and the task measures whether their kernel-level speedups
survive a real training step — the paper's level-4 question. Measurement is the
tier-3 protocol (``bench/tier3_runner``): identical weights and batches per
provider, step latency, peak memory, loss agreement, ``cpu_bound_fraction``.

Batches are fully synthetic (seeded random features and labels at a fixed crop
length). That is a requirement, not a shortcut: the protocol compares providers
on identical data, and an MSA/template pipeline would inject variance that is
not the kernels'.
"""

from evograd.opdecl import declare_workload
from evograd.opdecl.models import AF3_RESIDUE_SWEEP, ALPHAFOLD3
from evograd.ops._common import model_workloads

#: Patch sites, and the declared operator each one accepts. Both attention
#: flavours funnel to ``evoattention`` — pair-bias and triangle attention share
#: one softmax-attention core in this implementation, at different head widths.
AF3_SITE_OPS = {
    "layer_norm": "layernorm",
    "transition": "swiglu",
    "pair_bias_attention": "evoattention",
    "triangle_attention": "evoattention",
}

workload = declare_workload(
    name="alphafold3",
    factory="evograd.ops.level4.alphafold3.workload:make_workload",
    family="protein",
    model=ALPHAFOLD3.name,
    sites=dict(AF3_SITE_OPS),
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
        "Sites cover what MegaFold's own kernels cover: normalization, the "
        "SwiGLU transition, and the two softmax-attention flavours. The "
        "triangle-multiplicative updates, outer-product mean, MSA "
        "pair-weighted averaging, diffusion module internals, and confidence "
        "heads run stock eager inside the timed step and dilute the measured "
        "speedup accordingly — that dilution is the level-4 result, not a "
        "flaw. Attention-weight dropout is disabled for every provider alike: "
        "alphafold3-pytorch places the pair-stack dropout inside the "
        "attention core, which no fused-attention contract models (MegaFold's "
        "EvoAttention and DS4Sci included), and a dropout firing in one "
        "provider but not another would also desynchronize their RNG streams. "
        "MegaFold's fused LayerNorm+Linear (`layernorm_linear`) is not "
        "yet a site: alphafold3-pytorch has no fused module boundary to swap, "
        "so patching it needs composite surgery across PreLayerNorm and the "
        "first projection."
    ),
    requires="alphafold3-pytorch",
    notes=(
        "Weights are randomly initialized from a fixed seed; the task measures "
        "training-step speed and memory, not prediction quality, and identical "
        "weights across providers is what makes the comparison attributable."
    ),
)
