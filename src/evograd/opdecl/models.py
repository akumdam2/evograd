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

from dataclasses import dataclass, replace


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

    def swiglu_mlp_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The whole gated MLP block, kept in [B, T, H] rather than flattened.

        The projections alone are ``mlp_up_dims``/``mlp_down_dims`` as plain
        GEMMs over ``tokens``. This block keeps the batch and sequence axes
        because it is declared from an observed module invocation, and the
        observation records the shape the model actually passed -- flattening
        would make the declared shape something no layer ever received.
        """
        return {"B": batch, "T": seq, "H": self.hidden, "I": self.intermediate}

    def residual_rmsnorm_dims(self, *, tokens: int) -> dict[str, int]:
        """The decoder's residual-add-then-RMSNorm fusion site.

        Dimensionally identical to ``elementwise_dims`` -- one row per token,
        one column per residual channel -- but named for the site it describes,
        because the shape alone does not say that this is where an add and a
        norm meet and can be fused.
        """
        return {"rows": tokens, "cols": self.hidden}

    def residual_rmsnorm_fusion_sites(self) -> dict[str, int]:
        """How many times per step that fusion site actually occurs.

        Not ``2 * layers``. Each decoder layer performs two residual adds, and
        each is followed by an RMSNorm -- but the norm that follows the *MLP*
        add belongs to the *next* layer (or, for the last layer, to
        ``model.norm``). The very first ``input_layernorm`` therefore has no
        preceding decoder residual add and is not a fusion site, which is why
        the count is one short of the number of residual-width RMSNorms the
        model runs.
        """
        attention_sites = self.layers
        mlp_sites_into_next_layer = self.layers - 1
        final_mlp_site_into_model_norm = 1
        return {
            "attention_residual_then_post_attention_layernorm": attention_sites,
            "mlp_residual_then_next_layer_input_layernorm": mlp_sites_into_next_layer,
            "final_mlp_residual_then_model_norm": final_mlp_site_into_model_norm,
            "excluded_layer0_input_layernorm": 1,
            "total": attention_sites + mlp_sites_into_next_layer + final_mlp_site_into_model_norm,
        }

    def qkv_norm_rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The projection-plus-normalize-plus-rotate prefix of attention.

        Everything ``causal_gqa_attention_dims`` receives already prepared. The
        two are complementary halves of one module, and share ``QO`` -- the
        query fan-out -- which is where they meet.
        """
        return {
            "B": batch,
            "T": seq,
            "H": self.hidden,
            "HQ": self.n_heads,
            "HK": self.n_kv_heads,
            "D": self.head_dim,
            "QO": self.q_out,
            "KVO": self.kv_out,
        }

    def causal_gqa_attention_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Causal grouped-query SDPA plus the output projection.

        The boundary an observed training step actually presents: the
        projections and the rotary embedding happen *before* it and the residual
        add after, so ``q``, ``k`` and ``v`` arrive already split into heads and
        already rotated. ``QO`` is the query fan-out ``n_heads * head_dim``,
        which is the output projection's input width and, for Qwen3, is not
        ``hidden``.
        """
        return {
            "B": batch,
            "T": seq,
            "HQ": self.n_heads,
            "HK": self.n_kv_heads,
            "D": self.head_dim,
            "QO": self.q_out,
            "H": self.hidden,
        }

    def attn_qkv_dims(self, *, tokens: int) -> dict[str, int]:
        """The query projection, as a plain GEMM."""
        return {"M": tokens, "K": self.hidden, "N": self.q_out}

    def attn_kv_proj_dims(self, *, tokens: int) -> dict[str, int]:
        """The key or value projection. Narrower than the query's under GQA."""
        return {"M": tokens, "K": self.hidden, "N": self.kv_out}

    def attn_out_proj_dims(self, *, tokens: int) -> dict[str, int]:
        """The output projection: back from the query fan-out to ``hidden``."""
        return {"M": tokens, "K": self.q_out, "N": self.hidden}

    def q_head_norm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Qwen3's per-head RMSNorm on the queries, as a 2-D normalization.

        The module sees ``[B, T, heads, head_dim]`` and reduces over the last
        axis, so as a generic RMSNorm it is ``B * T * heads`` rows of
        ``head_dim`` -- many short rows rather than few long ones, which is a
        materially different occupancy regime from the residual-stream norm.
        """
        return {"rows": batch * seq * self.n_heads, "hidden": self.head_dim}

    def k_head_norm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The same, on the keys, which have fewer heads under GQA."""
        return {"rows": batch * seq * self.n_kv_heads, "hidden": self.head_dim}

    def causal_gqa_sdpa_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Causal grouped-query attention alone, without the output projection.

        The Level-1 primitive inside ``causal_gqa_attention_dims``: q, k and v
        in, one attention output out. ``o_proj`` is a separate GEMM and stays in
        the Level-2 task.
        """
        return {
            "B": batch,
            "HQ": self.n_heads,
            "HK": self.n_kv_heads,
            "T": seq,
            "D": self.head_dim,
        }

    def causal_softmax_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The causal softmax inside ``causal_gqa_sdpa_dims``: one [T, T] row
        block per (batch, query head). The scores and the value product on
        either side of it carry the full ``causal_gqa_sdpa`` dims; this one has
        no KV-head or head-dim axis because the softmax never sees them."""
        return {"B": batch, "HQ": self.n_heads, "T": seq}

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

    # ── whole-model structure (level 3) ───────────────────────────────────
    # Stack depths and input-feature widths, as the AF3 paper specifies them
    # and as alphafold3-pytorch (the implementation MegaFold builds on) sets
    # its defaults. A level-3 workload builds the training model from these,
    # so a benchmark run is traceable to the published architecture rather
    # than to whatever a harness happened to construct.
    pairformer_depth: int = 48
    msa_module_depth: int = 4
    template_pairformer_depth: int = 2
    diffusion_token_transformer_depth: int = 24
    diffusion_atom_encoder_depth: int = 3
    diffusion_atom_decoder_depth: int = 3
    confidence_pairformer_depth: int = 4
    #: Per-atom input feature width (alphafold3-pytorch's ``dim_atom_inputs``).
    dim_atom_inputs: int = 77
    #: Per-atom-pair input feature width (``dim_atompair_inputs``).
    dim_atompair_inputs: int = 5
    #: Template feature width (``dim_template_feats``).
    dim_template_feats: int = 108
    #: Sequence-local atom-transformer window (``atoms_per_window``).
    atoms_per_window: int = 27

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

    def pair_transition_layernorm_linear_dims(self, *, residues: int) -> dict[str, int]:
        """The pair transition's fused LayerNorm+Linear: one row per residue
        pair, ``c_z`` in, the SwiGLU's two ``expansion * c_z`` halves out.
        MegaFold builds it as ``LayernormLinear(dim, dim_inner * 2)``."""
        return {
            "M": residues * residues,
            "K": self.c_z,
            "N": 2 * self.transition_expansion * self.c_z,
        }

    def triangle_mult_layernorm_linear_dims(self, *, residues: int) -> dict[str, int]:
        """Triangle multiplication's combined projection: ``c_z`` in, five
        ``c_z``-wide outputs (left, right, their gates, the output gate) in
        one fused LayerNorm+Linear, twice per pairformer block. MegaFold
        builds it as ``LayernormLinear(dim, dim_hidden * 5)``."""
        return {"M": residues * residues, "K": self.c_z, "N": 5 * self.c_z}

    def pair_layernorm_dims(self, *, residues: int) -> dict[str, int]:
        """LayerNorm over the pair representation: one row per residue pair.

        The pairformer trunk's dominant normalization site at a given crop --
        ``[1, N, N, c_z]`` flattened to rows -- which is where MegaFold's
        ``layer_norm`` patch site does most of its work.
        """
        return {"rows": residues * residues, "hidden": self.c_z}

    def pair_transition_swiglu_dims(self, *, residues: int) -> dict[str, int]:
        """The pair transition's gated activation: ``silu(a) * b`` over the
        expanded pair width, one row per residue pair."""
        return {"rows": residues * residues, "cols": self.transition_expansion * self.c_z}

    def train_step_dims(self, *, batch: int, residues: int) -> dict[str, int]:
        """Every dim a whole-model training-step workload needs (level 3).

        ``batch`` and ``residues`` (the crop length) are the free dimensions;
        widths and stack depths come from the configuration, so a level-3
        benchmark case cannot silently drift away from the architecture it
        claims to train.
        """
        return {
            "batch": batch,
            "residues": residues,
            "c_s": self.c_s,
            "c_z": self.c_z,
            "c_m": self.c_m,
            "pairformer_depth": self.pairformer_depth,
            "diffusion_depth": self.diffusion_token_transformer_depth,
        }


@dataclass(frozen=True)
class HybridDenseModelConfig:
    """A hybrid Gated-DeltaNet / gated-attention decoder with a *dense* MLP in
    every layer, as Qwen3.5 publishes it.

    Three of every four layers are a Gated DeltaNet (linear attention), the
    fourth gated full attention; every layer's MLP is one dense SwiGLU, with no
    router and no experts. The GDN input stage projects q|k|v, z, b and a with
    four separate linears, so its site dims name ``C`` (the q|k|v width the
    convolution runs over) and ``VD`` (z).

    Layer types are derived from ``full_attention_interval`` exactly as
    Transformers derives them; Qwen3.5's ``config.json`` also writes the list out,
    and a test holds the two equal.
    """

    name: str
    hidden: int
    intermediate: int
    layers: int
    full_attention_interval: int
    # gated full attention
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    # Gated DeltaNet
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    vocab: int
    rms_norm_eps: float
    dtype: str
    tie_word_embeddings: bool = True
    #: Transformers' torch chunk size for the gated delta rule (a constant of the
    #: reference implementation, not a config field).
    chunk_size: int = 64
    source: str = ""

    # ── structure ────────────────────────────────────────────────────────
    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(
            "linear_attention" if (i + 1) % self.full_attention_interval else "full_attention"
            for i in range(self.layers)
        )

    def layer_count(self, kind: str) -> int:
        """How many layers of one kind the model stacks -- 24 GDN, 8 full for Qwen3.5-4B."""
        return sum(1 for t in self.layer_types if t == kind)

    # ── derived widths ───────────────────────────────────────────────────
    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        """``in_proj_qkv``'s fan-out, and the channels the short convolution runs over."""
        return 2 * self.key_dim + self.value_dim

    @property
    def q_out(self) -> int:
        """Query fan-out *without* the gate: what ``o_proj`` reads."""
        return self.n_heads * self.head_dim

    @property
    def q_gate_out(self) -> int:
        """``q_proj``'s fan-out: every head's query interleaved with its gate."""
        return 2 * self.q_out

    @property
    def kv_out(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    # ── Level-2 site dims ────────────────────────────────────────────────
    def gdn_in_proj_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The four GDN input projections, conv, split, gating and q/k repeat."""
        return {
            "B": batch, "T": seq, "H": self.hidden,
            "C": self.conv_dim, "VD": self.value_dim,
            "HV": self.linear_num_value_heads, "K": self.linear_conv_kernel_dim,
            "D": self.linear_key_head_dim,
        }

    def gated_delta_rule_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The chunked gated delta rule, q and k already repeated to the
        value-head count (Dk == Dv)."""
        return {"B": batch, "T": seq, "HV": self.linear_num_value_heads,
                "D": self.linear_value_head_dim}

    def gdn_norm_out_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "HV": self.linear_num_value_heads,
                "D": self.linear_value_head_dim, "VD": self.value_dim, "H": self.hidden}

    def qkv_norm_partial_rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {
            "B": batch, "T": seq, "H": self.hidden,
            "HQ": self.n_heads, "HK": self.n_kv_heads, "D": self.head_dim,
            "R": self.rotary_dim, "QG": self.q_gate_out, "KVO": self.kv_out,
            "QO": self.q_out,
        }

    def gated_attention_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "HQ": self.n_heads, "HK": self.n_kv_heads,
                "D": self.head_dim, "QO": self.q_out, "H": self.hidden}

    def residual_rmsnorm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "H": self.hidden}

    def swiglu_mlp_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The dense gated MLP, kept in ``[B, T, H]`` as the module receives it."""
        return {"B": batch, "T": seq, "H": self.hidden, "I": self.intermediate}

    # ── Level-1 component dims ───────────────────────────────────────────
    def gdn_conv1d_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "C": self.conv_dim, "T": seq, "K": self.linear_conv_kernel_dim}

    def gdn_qk_l2norm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """q and k are L2-normalized after the repeat to the value-head count."""
        return {"rows": batch * seq * self.linear_num_value_heads,
                "cols": self.linear_key_head_dim}

    def gdn_gating_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "heads": self.linear_num_value_heads}

    def gdn_gated_norm_dims(self, *, tokens: int) -> dict[str, int]:
        """``Qwen3_5RMSNormGated`` over one value head per row."""
        return {"rows": tokens * self.linear_num_value_heads,
                "hidden": self.linear_value_head_dim}

    def residual_norm_dims(self, *, tokens: int) -> dict[str, int]:
        """The residual-stream (1 + w) RMSNorm: both per-layer norms and the final one."""
        return {"rows": tokens, "hidden": self.hidden}

    def attn_q_norm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"rows": batch * seq * self.n_heads, "hidden": self.head_dim}

    def attn_k_norm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"rows": batch * seq * self.n_kv_heads, "hidden": self.head_dim}

    def partial_rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "H": self.n_heads, "T": seq, "D": self.head_dim,
                "R": self.rotary_dim}

    def partial_rope_kv_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "H": self.n_kv_heads, "T": seq, "D": self.head_dim,
                "R": self.rotary_dim}

    def causal_gqa_sdpa_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "HQ": self.n_heads, "HK": self.n_kv_heads, "T": seq,
                "D": self.head_dim}

    def mlp_activation_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "cols": self.intermediate}

    def logits_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "cols": self.vocab}

    # Biasless projections, as plain GEMMs: [M, K] @ [N, K]^T.
    def gdn_in_qkv_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.conv_dim}

    def gdn_in_z_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.value_dim}

    def gdn_in_ba_dims(self, *, tokens: int) -> dict[str, int]:
        """``in_proj_b`` and ``in_proj_a``: one scalar per value head each."""
        return {"M": tokens, "K": self.hidden, "N": self.linear_num_value_heads}

    def gdn_out_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.value_dim, "N": self.hidden}

    def attn_q_gate_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.q_gate_out}

    def attn_kv_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.kv_out}

    def attn_out_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.q_out, "N": self.hidden}

    def mlp_up_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.intermediate}

    def mlp_down_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.intermediate, "N": self.hidden}

    def lm_head_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.vocab}


@dataclass(frozen=True)
class MoeModelConfig:
    """A decoder-only transformer with a routed mixture of experts in the MLP
    position, as OLMoE publishes it.

    Attention is Llama's shape (``head_dim = hidden // n_heads``) with one
    difference that is OLMoE's own: the query and key projections are RMSNormed
    over their *full width* (``q_out`` and ``kv_out``) before the head split,
    not per head as Qwen3 does. The MLP position holds a router of ``[E, H]``
    and ``n_experts`` SwiGLU experts of width ``intermediate`` each, stored as
    one ``[E, 2I, H]`` gate/up tensor and one ``[E, H, I]`` down tensor
    (Transformers 5's layout); ``top_k`` of them serve each token.
    """

    name: str  # registry key, also what Provenance.model stores
    hidden: int
    intermediate: int  # per expert
    n_experts: int
    top_k: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    layers: int
    rope_theta: float
    rms_norm_eps: float
    dtype: str
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    source: str = ""

    # ── derived widths ────────────────────────────────────────────────────
    @property
    def q_out(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gate_up_out(self) -> int:
        """Each expert's fused gate/up fan-out: ``2 * intermediate``."""
        return 2 * self.intermediate

    # ── Level-2 site dims ────────────────────────────────────────────────
    def qkv_norm_rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Projections, the full-width q/k RMSNorms and RoPE: the attention prefix."""
        return {
            "B": batch, "T": seq, "H": self.hidden,
            "HQ": self.n_heads, "HK": self.n_kv_heads, "D": self.head_dim,
            "QO": self.q_out, "KVO": self.kv_out,
        }

    def attention_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """Causal (grouped-query) SDPA plus the output projection."""
        return {"B": batch, "T": seq, "HQ": self.n_heads, "HK": self.n_kv_heads,
                "D": self.head_dim, "QO": self.q_out, "H": self.hidden}

    def moe_experts_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        """The routed experts, token-flat as the block hands them over: ``T`` is
        ``batch * seq`` because ``OlmoeSparseMoeBlock`` flattens before routing."""
        return {"T": batch * seq, "H": self.hidden, "E": self.n_experts,
                "G": self.gate_up_out, "I": self.intermediate, "K": self.top_k}

    def residual_rmsnorm_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "H": self.hidden}

    # ── Level-1 component dims ───────────────────────────────────────────
    def residual_norm_dims(self, *, tokens: int) -> dict[str, int]:
        """The residual-stream RMSNorm: both per-layer norms and the final one."""
        return {"rows": tokens, "hidden": self.hidden}

    def attn_q_norm_dims(self, *, tokens: int) -> dict[str, int]:
        """OLMoE's query norm: one row per token over the whole projection."""
        return {"rows": tokens, "hidden": self.q_out}

    def attn_k_norm_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "hidden": self.kv_out}

    def rope_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "n_heads": self.n_heads, "head_dim": self.head_dim}

    def rope_kv_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "T": seq, "n_heads": self.n_kv_heads, "head_dim": self.head_dim}

    def causal_gqa_sdpa_dims(self, *, batch: int, seq: int) -> dict[str, int]:
        return {"B": batch, "HQ": self.n_heads, "HK": self.n_kv_heads, "T": seq,
                "D": self.head_dim}

    def moe_activation_dims(self, *, tokens: int) -> dict[str, int]:
        """The experts' SiLU under the grouped-GEMM path: one call per layer on
        the packed ``[tokens * top_k, intermediate]`` tensor."""
        return {"rows": tokens * self.top_k, "cols": self.intermediate}

    def logits_dims(self, *, tokens: int) -> dict[str, int]:
        return {"rows": tokens, "cols": self.vocab}

    # Biasless projections, as plain GEMMs: [M, K] @ [N, K]^T.
    def attn_qkv_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.q_out}

    def attn_kv_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.kv_out}

    def attn_out_proj_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.q_out, "N": self.hidden}

    def lm_head_dims(self, *, tokens: int) -> dict[str, int]:
        return {"M": tokens, "K": self.hidden, "N": self.vocab}


#: Meta-Llama-3-8B. The operator suite's timed grids (``rmsnorm``, ``swiglu``,
#: the vocabulary-regime loss kernels, ...) are derived from *these* widths, and
#: every historical suite report records them under the key ``llama_3_8b``.
#: The key therefore stays ``llama_3_8b`` even though the top-down workload
#: moved to :data:`LLAMA_3_2_1B`: relabelling 4096-wide shapes as a 2048-wide
#: model would make ``Provenance.model`` say something false about every
#: operator-suite case and about every result already on record.
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

#: Llama-3.2-1B, as ``meta-llama/Llama-3.2-1B``'s ``config.json`` publishes it.
#: This is the configuration the ``topdown.llama3_2_1b`` workload actually pins,
#: and the key its Level-2 tasks cite in their ``Provenance`` -- so it is
#: registered in :data:`MODELS` under ``llama_3_2_1b`` and ``rederive_dims`` can
#: check them. :data:`LLAMA_3_8B` above stays at the 8B numbers, under its own
#: key, because the operator suite's vocabulary-regime cases (``kl_div``,
#: ``fused_linear_cross_entropy``) were chosen to represent a *large* LLM's loss
#: shapes, and shrinking them would change what those tasks are for.
#:
#: Unlike Meta-Llama-3-8B: ``head_dim`` is 64 rather than 128, and the
#: embeddings are tied, so there is no second ``vocab x hidden`` matrix.
LLAMA_3_2_1B = ModelConfig(
    name="llama_3_2_1b",
    hidden=2048,
    intermediate=8192,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    vocab=128256,
    layers=16,
    # 500000, not the 10000 Llama-2 uses. Getting this wrong produces a RoPE
    # kernel that is self-consistent and completely wrong. Llama-3.2 also
    # *scales* RoPE, which moves the values in the cos/sin tables but not any
    # kernel's contract -- the boundary receives them as Inactive inputs.
    rope_theta=500000.0,
    dtype="bfloat16",
    source="meta-llama/Llama-3.2-1B config.json",
)

#: Qwen3-0.6B, as ``Qwen/Qwen3-0.6B``'s ``config.json`` publishes it. Unlike
#: Llama-3, ``n_heads * head_dim`` (2048) is *not* ``hidden`` (1024): Qwen3's
#: query projection fans out and ``o_proj`` brings it back, which is why the
#: harvested q_proj is 1024 -> 2048.
QWEN3_0_6B = ModelConfig(
    name="qwen3_0_6b",
    hidden=1024,
    intermediate=3072,
    n_heads=16,
    n_kv_heads=8,
    head_dim=128,
    vocab=151936,
    layers=28,
    rope_theta=1000000.0,
    dtype="bfloat16",
    source="Qwen/Qwen3-0.6B config.json; observed in the evograd Level-3 harvest",
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

#: Qwen3.5-4B's text model, as ``Qwen/Qwen3.5-4B``'s ``config.json`` publishes it
#: under ``text_config`` (the ViT and the MTP head are not part of this
#: workload). Every benchmark shape the ``qwen3_5_4b`` cases use is derived from
#: here; ``benchmark.topdown.qwen3_5_4b.architecture`` carries the same fields
#: verbatim and a test keeps the two in agreement.
QWEN3_5_4B = HybridDenseModelConfig(
    name="qwen3_5_4b",
    hidden=2560,
    intermediate=9216,
    layers=32,
    full_attention_interval=4,
    n_heads=16,
    n_kv_heads=4,
    head_dim=256,
    partial_rotary_factor=0.25,
    rope_theta=10000000.0,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    vocab=248320,
    rms_norm_eps=1e-6,
    dtype="bfloat16",
    tie_word_embeddings=True,
    source="Qwen/Qwen3.5-4B config.json (text_config)",
)

#: OLMoE-1B-7B, as ``allenai/OLMoE-1B-7B-0924``'s ``config.json`` publishes it.
#: Multi-head rather than grouped-query attention (16 query heads over 16 KV
#: heads of 128), 64 experts of width 1024 with 8 active per token, untied
#: embeddings over a 50,304-token vocabulary. ``intermediate`` is per expert;
#: ``norm_topk_prob`` is false, so the routing weights are the raw softmax
#: probabilities of the chosen experts and do not sum to one.
OLMOE_1B_7B = MoeModelConfig(
    name="olmoe_1b_7b",
    hidden=2048,
    intermediate=1024,
    n_experts=64,
    top_k=8,
    n_heads=16,
    n_kv_heads=16,
    head_dim=128,
    vocab=50304,
    layers=16,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    dtype="bfloat16",
    norm_topk_prob=False,
    tie_word_embeddings=False,
    source="allenai/OLMoE-1B-7B-0924 config.json",
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

#: AlphaFold3's architecture with the stacks cut to two pairformer blocks.
#:
#: The level-3 analogue of ``LLAMA_3_8B_4L``: every width, head count, and
#: input-feature dimension is the real configuration, only the stack depths are
#: reduced, so the kernels see the true shapes while a training step stays
#: cheap enough to iterate on. Use it to iterate; report from ``ALPHAFOLD3``.
ALPHAFOLD3_2L = replace(
    ALPHAFOLD3,
    name="alphafold3_2l",
    pairformer_depth=2,
    msa_module_depth=1,
    template_pairformer_depth=1,
    diffusion_token_transformer_depth=2,
    diffusion_atom_encoder_depth=1,
    diffusion_atom_decoder_depth=1,
    confidence_pairformer_depth=1,
    source="ALPHAFOLD3 with reduced stack depths; iteration config, not a published model",
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

MODELS: dict[str, ModelConfig | AlphaFoldConfig | HybridDenseModelConfig | MoeModelConfig] = {
    config.name: config
    for config in (LLAMA_3_8B, LLAMA_3_2_1B, QWEN3_0_6B, ALPHAFOLD3, QWEN3_5_4B, OLMOE_1B_7B)
}
# One key per model: a name collision here would let one configuration
# silently answer for another's provenance.
assert len(MODELS) == 6, sorted(MODELS)


def config_for(provenance) -> ModelConfig | AlphaFoldConfig | HybridDenseModelConfig | MoeModelConfig:
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
    "ALPHAFOLD3_2L",
    "AlphaFoldConfig",
    "HybridDenseModelConfig",
    "LLAMA_3_8B",
    "LLAMA_REGIME_SPLIT",
    "LLAMA_TOKEN_SWEEP",
    "LLAMA_VOCAB_REGIME_SPLIT",
    "LLAMA_VOCAB_TOKEN_SWEEP",
    "LLAMA_VOCAB_TOKEN_SWEEP_FP32",
    "MODELS",
    "ModelConfig",
    "MoeModelConfig",
    "OLMOE_1B_7B",
    "QWEN3_5_4B",
    "config_for",
    "rederive_dims",
]
