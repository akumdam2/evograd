"""Workload declaration: MegaFold training step (level 3).

The whole-model level-3 task: one full MegaFold training step, forward to the
model's own combined loss, backward, optimizer step, with evolved kernels
patched into the module tree by surgery.

The model is `MegaFold <https://github.com/Supercomputing-System-AI-Lab/MegaFold>`_
(arXiv:2506.20686), which is where every AlphaFold3 shape in
:mod:`evograd.opdecl.models` came from. Using it here closes the provenance
chain at both ends: the level-1/2 operator grids were transcribed from this
model's configuration, and this task runs that same configuration, so the
``layernorm`` and ``swiglu`` benchmark grids and the sites this model executes
are the same widths by construction rather than by coincidence.

Its tasks come from what AF3's training step computes (``docs/AF3.md``), not from
which kernels MegaFold ships, and MegaFold's own fused kernels are an expert
baseline beside eager PyTorch and ``torch.compile``. The level-3 question is whether
an evolved kernel's local speedup survives a real training step.

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

#: Patch sites, and the task each one accepts. Both flavours of ``Attend`` funnel to
#: ``megafold_evoattention``: pair-bias and triangle attention share it, at different
#: head widths. The LayerNorm and the SwiGLU are the generic tasks, because MegaFold
#: computes exactly those there in either precision region. The rest are its own
#: mixed-precision contracts (:mod:`.contracts`, ``levels/level2``), one task key per
#: role no shaped argument separates (``docs/AF3.md`` section 3).
MEGAFOLD_SITE_OPS = {
    "layer_norm": "layernorm",
    "layernorm_linear": "megafold_layernorm_linear",
    "transition": "swiglu",
    "pair_bias_attention": "megafold_evoattention",
    "triangle_attention": "megafold_evoattention",
    "triangle_multiplication_outgoing": "megafold_triangle_multiplication_outgoing",
    "triangle_multiplication_incoming": "megafold_triangle_multiplication_incoming",
    "triangle_attention_starting": "megafold_triangle_attention_starting",
    "triangle_attention_ending": "megafold_triangle_attention_ending",
    "transition_block": "megafold_transition",
    "outer_product_mean": "megafold_outer_product_mean",
    "msa_pair_weighted_averaging": "megafold_msa_pair_weighted_averaging",
}

#: Pairs of sites that nest: the first site's boundary holds the second's calls,
#: so they are never installed together (``SiteRegistry.overlaps``).
MEGAFOLD_SITE_OVERLAPS: tuple[tuple[str, str], ...] = (
    ("triangle_multiplication_outgoing", "layernorm_linear"),
    ("triangle_multiplication_incoming", "layernorm_linear"),
    ("triangle_attention_starting", "layer_norm"),
    ("triangle_attention_starting", "triangle_attention"),
    ("triangle_attention_ending", "layer_norm"),
    ("triangle_attention_ending", "triangle_attention"),
    ("transition_block", "layernorm_linear"),
    ("transition_block", "transition"),
    ("outer_product_mean", "layernorm_linear"),
    ("msa_pair_weighted_averaging", "layernorm_linear"),
)

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
        "Sites are the boundaries docs/AF3.md lists. Nested sites are never "
        "installed together, and everything outside the installed sites runs "
        "stock and dilutes the measured speedup accordingly, which is the "
        "level-3 result rather than a flaw. Windowed atom attention, attention "
        "with pair bias as a whole, the transition without its LayerNorm, AdaLN "
        "and its output gate, the embedding and pooling layers, the output heads, "
        "the losses and the alignment hold no site. Attention-weight dropout is disabled for every provider "
        "alike: the warm start was trained without it, and a dropout firing in "
        "one provider but not another would also desynchronize their RNG streams. "
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
