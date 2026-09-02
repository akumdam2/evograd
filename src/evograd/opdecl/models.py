"""Frozen configurations of the real models the benchmark draws its shapes from.

Every timed workload in the suite is supposed to be traceable to a layer of a
real network. Before this module the traceability existed only in comments —
``matmul`` benchmarked ``(M=8192, K=4096, N=14336)`` and ``cross_entropy``
benchmarked ``(4096, 128256)``, which *are* Llama-3-8B's MLP and vocabulary, but
nothing recorded that and nothing would have caught an edit that drifted away
from them.

So shapes are derived here rather than typed into declarations, and
:class:`evograd.opdecl.activity.Provenance` records which config and which
component a workload came from. ``tests/test_provenance`` re-derives every
benchmark workload through these methods and fails if the numbers no longer
agree, which turns provenance from a comment into an assertion.

The registry deliberately imports nothing: declarations must stay importable on
machines without torch (see :mod:`evograd.opdecl`), and this module is imported
by every declaration that derives a shape.

Naming: a ``*_dims`` method takes only the dimensions a model configuration does
*not* fix — batch size, token count, crop length — and returns the complete dim
dict for one operator. Those free arguments are what a ``Provenance`` stores in
its ``free`` field, so the round-trip is ``cfg.<component>_dims(**free) == dims``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """One decoder-only transformer, as its published config defines it.

    Field values are copied from the reference the ``source`` names; do not
    "round" them. ``intermediate`` is not ``4 * hidden`` for any modern model,
    and ``vocab`` is the number that makes the loss kernels interesting.
    """

    name: str  # registry key, also what Provenance.model stores
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    layers: int
    rope_theta: float
    dtype: str
    source: str = ""

    # ── derived widths ────────────────────────────────────────────────────
    # Shape strings in a declaration cannot contain arithmetic (only declared
    # dims or integer literals), so a product like n_heads*head_dim has to enter
    # the declaration as its own symbolic dim. These compute the values.

    @property
    def q_out(self) -> int:
        """Query projection fan-out. Equals ``hidden`` for Llama-style models."""
        return self.n_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        """Key/value projection fan-out; smaller than ``q_out`` under GQA."""
        return self.n_kv_heads * self.head_dim

    # ── per-operator dim builders ─────────────────────────────────────────

    def rmsnorm_dims(self, *, tokens: int) -> dict[str, int]:
        """RMSNorm over the residual stream: one row per token."""
        return {"rows": tokens, "hidden": self.hidden}

    def layernorm_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "hidden": self.hidden}

    def mlp_activation_dims(self, *, tokens: int) -> dict[str, int]:
        """SwiGLU/GeGLU operate on the MLP's intermediate width, not ``hidden``."""
        return {"rows": tokens, "cols": self.intermediate}

    def elementwise_dims(self, *, tokens: int) -> dict[str, int]:
        """Pointwise ops on the residual stream (relu_squared, dyt, poly_norm)."""
        return {"rows": tokens, "cols": self.hidden}

    def logits_dims(self, *, tokens: int) -> dict[str, int]:
        """Loss inputs: one row per token, one column per vocabulary entry."""
        return {"rows": tokens, "cols": self.vocab}

    def flce_dims(self, *, tokens: int) -> dict[str, int]:
        """Fused linear + cross entropy: the lm_head GEMM and the loss together.

        The point of the fusion is that ``[tokens, vocab]`` logits are never
        materialized, so the declaration takes ``hidden`` and ``vocab``
        separately rather than a logits shape.
        """
        return {"rows": tokens, "hidden": self.hidden, "vocab": self.vocab}

    def lm_head_dims(self, *, tokens: int) -> dict[str, int]:
        """The unfused lm_head projection, as a plain GEMM."""
        return {"M": tokens, "K": self.hidden, "N": self.vocab}

    def mlp_up_dims(self, *, tokens: int) -> dict[str, int]:
        """gate/up projection: [tokens, hidden] @ [hidden, intermediate]."""
        return {"M": tokens, "K": self.hidden, "N": self.intermediate}

    def mlp_down_dims(self, *, tokens: int) -> dict[str, int]:
        """down projection: [tokens, intermediate] @ [intermediate, hidden]."""
        return {"M": tokens, "K": self.intermediate, "N": self.hidden}

    def attn_qkv_dims(self, *, tokens: int) -> dict[str, int]:
        """The query projection, as a plain GEMM."""
        return {"M": tokens, "K": self.hidden, "N": self.q_out}

    def rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """RoPE applied to one projected tensor, in [B, T, heads, head_dim]."""
        return {
            "B": batch,
            "T": seq,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
        }

    def rope_kv_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """RoPE on the key tensor, which has fewer heads under GQA."""
        return {
            "B": batch,
            "T": seq,
            "n_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
        }

    def decoder_layer_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Every dim a whole decoder layer needs.

        ``n_heads``/``n_kv_heads`` are deliberately absent: a declared dim that
        appears in no shape string cannot be recovered from tensor shapes, and
        ``dispatch._route`` rebuilds the dim dict from shapes alone at deploy
        time. The layer's reference recovers them as ``q_out // head_dim`` and
        ``kv_out // head_dim`` instead.
        """
        return {
            "B": batch,
            "T": seq,
            "hidden": self.hidden,
            "head_dim": self.head_dim,
            "q_out": self.q_out,
            "kv_out": self.kv_out,
            "intermediate": self.intermediate,
        }


@dataclass(frozen=True)
class AlphaFoldConfig:
    """AlphaFold3-style attention widths, as MegaFold's implementation sets them.

    AF3 does not have a single "hidden size": each attention flavour carries its
    own head count and head dim, and the residue axis is the crop length rather
    than a token count. ``pair_bias`` is a trainable per-head bias over residue
    pairs, which is why it is an ``Active`` input everywhere it appears.
    """

    name: str
    c_s: int  # single representation width
    c_z: int  # pair representation width
    c_m: int  # MSA representation width
    pair_bias_attn_heads: int
    pair_bias_attn_dim_head: int
    tri_attn_heads: int
    tri_attn_dim_head: int
    msa_pwa_heads: int
    msa_pwa_dim_head: int
    source: str = ""

    def pair_bias_attention_dims(
        self, *, batch: int, n_seq: int, residues: int
    ) -> dict[str, int]:
        return {
            "B": batch,
            "S": n_seq,
            "H": self.pair_bias_attn_heads,
            "N": residues,
            "D": self.pair_bias_attn_dim_head,
        }

    def triangle_attention_dims(
        self, *, batch: int, residues: int
    ) -> dict[str, int]:
        """Triangle attention's MSA axis *is* the third residue index.

        Hence ``S == N``. The pre-v1 grid used ``S=64`` at ``N in {128, 256}``,
        which understates the real work by 2x and 4x respectively; deriving the
        dims here makes that class of mistake unrepresentable.
        """
        return {
            "B": batch,
            "S": residues,
            "H": self.tri_attn_heads,
            "N": residues,
            "D": self.tri_attn_dim_head,
        }

    def msa_attention_dims(
        self, *, batch: int, n_seq: int, residues: int
    ) -> dict[str, int]:
        return {
            "B": batch,
            "S": n_seq,
            "H": self.msa_pwa_heads,
            "N": residues,
            "D": self.msa_pwa_dim_head,
        }

    #: The transition layer widens the channel axis by this factor before its
    #: gated activation — the same ratio a transformer feed-forward block uses.
    transition_expansion: int = 4

    def single_repr_block_dims(
        self, *, batch: int, n_seq: int, residues: int
    ) -> dict[str, int]:
        """Every dim the single-representation block needs.

        The head dim is absent by design: it appears in no tensor shape, and
        ``dispatch._route`` rebuilds the dim dict from tensor shapes alone at
        deploy time. The block's reference recovers it as ``E // H``.
        """
        return {
            "B": batch,
            "S": n_seq,
            "N": residues,
            "H": self.pair_bias_attn_heads,
            "C": self.c_s,
            "E": self.pair_bias_attn_heads * self.pair_bias_attn_dim_head,
            "F": self.transition_expansion * self.c_s,
        }

    def layernorm_linear_dims(
        self, *, batch: int, n_seq: int, residues: int
    ) -> dict[str, int]:
        """MegaFold's fused LayerNorm+Linear: normalize c_s, project to q/k/v.

        The declaration is 2-D, so the leading batch/MSA/residue axes flatten
        into ``M``. ``N`` is the attention fan-out ``heads * dim_head``.
        """
        return {
            "M": batch * n_seq * residues,
            "K": self.c_s,
            "N": self.pair_bias_attn_heads * self.pair_bias_attn_dim_head,
        }


LLAMA_3_8B = ModelConfig(
    name="llama_3_8b",
    hidden=4096,
    intermediate=14336,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    vocab=128256,
    layers=32,
    # 500000, not the 10000 Llama-2 uses. Getting this wrong produces a RoPE
    # kernel that is self-consistent and completely wrong.
    rope_theta=500000.0,
    dtype="bfloat16",
    source="meta-llama/Meta-Llama-3-8B; Liger benchmark/scripts/benchmark_model_configs.py",
)

#: Llama-3-8B's architecture with four layers instead of thirty-two.
#:
#: For measuring what a kernel does to a training step, layer count is the one
#: dimension that can be cut without changing the answer: every per-layer effect
#: scales linearly in it, and the loss kernel's memory — a ``[tokens, vocab]``
#: logits tensor at 128256 vocab — does not depend on it at all. Every other
#: field is Llama-3-8B's, so the shapes each kernel sees are the real ones.
#:
#: Use it to iterate. Report from ``LLAMA_3_8B``.
LLAMA_3_8B_4L = ModelConfig(
    name="llama_3_8b_4l",
    hidden=LLAMA_3_8B.hidden,
    intermediate=LLAMA_3_8B.intermediate,
    n_heads=LLAMA_3_8B.n_heads,
    n_kv_heads=LLAMA_3_8B.n_kv_heads,
    head_dim=LLAMA_3_8B.head_dim,
    vocab=LLAMA_3_8B.vocab,
    layers=4,
    rope_theta=LLAMA_3_8B.rope_theta,
    dtype=LLAMA_3_8B.dtype,
    source="LLAMA_3_8B with layers=4; iteration config, not a published model",
)

ALPHAFOLD3 = AlphaFoldConfig(
    name="alphafold3",
    c_s=384,
    c_z=128,
    c_m=64,
    pair_bias_attn_heads=16,
    pair_bias_attn_dim_head=64,
    tri_attn_heads=4,
    tri_attn_dim_head=32,
    msa_pwa_heads=8,
    msa_pwa_dim_head=32,
    source="MegaFold (arXiv:2506.20686) megafold/model/megafold.py",
)

#: Token counts the residual-stream and MLP operators sweep. Chosen to bracket
#: the regimes a real training step visits: a partial batch, Liger's own e2e
#: setting (batch 64 x seq 512 = 32768 tokens sharded over 4 GPUs, so ~8192 per
#: device), and a long-context step. At the widest of these an activation is
#: 16384 x 14336 x bf16 = 470 MB, which leaves room for forward, backward, and a
#: baseline on a 40 GB card.
LLAMA_TOKEN_SWEEP = (1024, 2048, 4096, 8192, 16384)

#: Token counts for the vocabulary-width operators (the losses, and softmax /
#: sparsemax over logits). These cannot reuse the sweep above: one
#: 16384 x 128256 bf16 logits tensor is 4.2 GB, and a loss step needs the
#: logits, their gradient, and a baseline's copy simultaneously. 8192 tokens
#: (2.1 GB per tensor) is the largest that fits comfortably.
LLAMA_VOCAB_TOKEN_SWEEP = (512, 1024, 2048, 4096, 8192)

#: float32 doubles every byte above, so the vocabulary sweep stops one step
#: earlier when a declaration benchmarks fp32.
LLAMA_VOCAB_TOKEN_SWEEP_FP32 = (512, 1024, 2048, 4096)

#: Where the small/large shape regimes split, for grids derived from the sweeps
#: above. Because a model configuration fixes the column width, the token count
#: *is* the workload size, so one rule covers every derived LLM grid: split at
#: the middle of the declared sweep. The pre-v1 declarations each carried their
#: own split against their own hand-picked grid (some on ``rows``, some on
#: ``rows*cols``, at values from 4096 to 1e6), which made "small" mean something
#: different in every operator.
LLAMA_REGIME_SPLIT = 4096
LLAMA_VOCAB_REGIME_SPLIT = 2048

#: Residue crops MegaFold reports. Its full sweep runs to 1024; the entries past
#: 384 belong in untimed coverage rather than the default timed suite.
AF3_RESIDUE_SWEEP = (128, 256, 384)

MODELS: dict[str, ModelConfig | AlphaFoldConfig] = {
    config.name: config for config in (LLAMA_3_8B, ALPHAFOLD3)
}


def config_for(provenance) -> ModelConfig | AlphaFoldConfig:
    """The configuration a :class:`Provenance` names, or a clear failure."""
    try:
        return MODELS[provenance.model]
    except KeyError:
        raise KeyError(
            f"provenance names unknown model {provenance.model!r}; "
            f"known: {sorted(MODELS)}"
        ) from None


def rederive_dims(provenance) -> dict[str, int]:
    """Recompute a workload's dims from its provenance alone.

    This is the whole point of the provenance field: if the stored shape and the
    model configuration ever disagree, one of them is wrong, and the test that
    calls this says so.
    """
    config = config_for(provenance)
    builder = getattr(config, f"{provenance.component}_dims", None)
    if builder is None:
        raise AttributeError(
            f"{type(config).__name__} {config.name!r} has no component "
            f"{provenance.component!r} (expected a {provenance.component}_dims method)"
        )
    return builder(**provenance.free)


__all__ = [
    "AF3_RESIDUE_SWEEP",
    "ALPHAFOLD3",
    "AlphaFoldConfig",
    "LLAMA_3_8B",
    "LLAMA_REGIME_SPLIT",
    "LLAMA_TOKEN_SWEEP",
    "LLAMA_VOCAB_REGIME_SPLIT",
    "LLAMA_VOCAB_TOKEN_SWEEP",
    "LLAMA_VOCAB_TOKEN_SWEEP_FP32",
    "MODELS",
    "ModelConfig",
    "config_for",
    "rederive_dims",
]
