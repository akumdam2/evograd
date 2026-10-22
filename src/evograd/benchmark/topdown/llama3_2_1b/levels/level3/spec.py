"""The canonical Llama-3 training workload: its architecture and run settings.

What a workload spec *is* -- the frozen fields, the hashing, the three rules
``validate`` refuses to bend -- is the same for every model and lives in
:mod:`....common.spec`. What is Llama's is the architecture below and the
canonical batch, sequence, dtype, device and attention backend, which are the
field defaults of :class:`WorkloadSpec`.

The architecture is Llama-3.2-1B's published configuration, written out rather
than downloaded. Two reasons, and the second is the one that matters here: a
reference must be reproducible on a node with no network, and the Llama
repositories are **gated** -- fetching a config would require an authenticated
Hub token. The canonical workload never needs one, because it builds from this
configuration with random weights and no tokenizer. The separate real-text
identity (:data:`PRETRAINED` on :data:`REAL_TEXT`) loads the pinned checkpoint
from a local cache that a one-time preparation step fills.

**Why 1B rather than the 8B this package was written around.** Meta-Llama-3-8B
does not fit a single 95 GiB card under AdamW: 8.03B parameters cost ~64 GiB in
weights, gradients and two moments before any activation. Every Llama number
would have come from a ``--layers`` smoke, and a reduced-depth run cannot carry
a calibration -- the gate binds a policy to the workload id it was measured
for. Llama-3.2-1B is a *published* model, so its shapes are real ones somebody
trains, and at ~1.24B it is measurable at full depth and full sequence. That is
the difference between a benchmark number and an exploratory one.

It is also a genuinely different point from Llama-3-8B, which is what makes it
worth measuring rather than merely cheaper: ``head_dim`` is 64 where the 8B's is
128, and the embeddings are **tied**, so ``embed_tokens.weight`` accumulates
gradient from both the lookup and the head.

Sequence length 2048 rather than the 131072 the architecture permits, so the
observed shapes line up with the Qwen3 workload's and the two can be read side
by side. It is a canonical choice, not a limit, and ``validate`` enforces the
architectural ceiling.

The package is still named ``llama3_2_1b`` and its tasks ``llama3_*``. Those names
predate this configuration and are load-bearing -- they appear in report
schemas, result directories and tracked artifacts -- so only the configuration
moved. ``llama3`` remains accurate; the ``_8b`` does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....common.spec import (  # noqa: F401  (re-export)
    SUPPORTED_ATTENTION,
    SUPPORTED_DTYPES,
    WorkloadSpecError,
    analytic_parameter_count,
)
from ....common.spec import WorkloadSpec as _WorkloadSpec

#: Llama-3.2-1B, as ``meta-llama/Llama-3.2-1B``'s ``config.json`` publishes it.
#: ~1.24B parameters, of which 0.26B are the embedding -- counted once, because
#: ``tie_word_embeddings`` is true and the lm_head is the same matrix.
#: ``analytic_parameter_count`` reproduces the total from these numbers alone.
#:
#: ``head_dim`` is 64, and stated explicitly: ``hidden_size //
#: num_attention_heads`` gives the same answer here, but a derived value that
#: later moved would change every attention shape without changing this file.
LLAMA_3_2_1B: Mapping[str, Any] = {
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05,
    # 500000, not the 10000 Llama-2 uses. Getting this wrong produces a RoPE
    # kernel that is self-consistent and completely wrong.
    "rope_theta": 500000.0,
    # Llama-3.2 scales RoPE where Llama-3 did not. This changes the *values* in
    # the cos/sin tables, not any kernel's contract: `LlamaRotaryEmbedding`
    # builds them once per forward and the qkv_rope boundary receives them as
    # Inactive inputs. So the declared operator is unchanged and only the
    # numbers flowing through it move.
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    # Like Qwen3-0.6B and unlike Meta-Llama-3-8B, this model *ties* its
    # embeddings: the lm_head is the embedding matrix transposed, not a second
    # 128256x2048 one. `embed_tokens.weight` therefore accumulates gradient
    # from two paths, which is worth knowing when reading its drift.
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
}

MODEL_NAME = "Llama-3.2-1B"

#: The real-text workload's two pins. The checkpoint is gated: downloading it
#: needs a Hub account that accepted Meta's license, so only the preparation
#: step (``scripts/prepare_benchmark_inputs.py --benchmark llama_3_2_1b``) ever
#: needs a token. The canonical synthetic workload still needs none.
PRETRAINED = "pretrained:meta-llama/Llama-3.2-1B@4e20de362430cd3b72f300e6b0f18e50e7166e08"
#: The same WikiText-2 raw revision Qwen3 packs (``qwen3_0_6b.textdata``).
REAL_TEXT = "wikitext-2-raw:b08601e04326c79dfdd32d625aee71d232d685c3"


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined Llama-3.2-1B training execution.

    The defaults are the canonical run, which is what makes
    ``WorkloadSpec() == CANONICAL`` true and ``is_canonical`` meaningful.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(LLAMA_3_2_1B.items()))
    batch_size: int = 2
    seq_len: int = 2048
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    #: Where the parameters come from: ``"random"`` (``nn.init`` under ``seed``,
    #: the canonical workload) or :data:`PRETRAINED`. Part of the identity, so a
    #: calibration taken on random weights can never bind to pretrained ones.
    weights: str = "random"
    #: Where the tokens come from: ``"synthetic"`` (drawn from ``seed``) or
    #: :data:`REAL_TEXT`, tokenised with the checkpoint's own tokenizer.
    data: str = "synthetic"

    @property
    def pretrained(self) -> bool:
        return self.weights == PRETRAINED

    @property
    def real_text(self) -> bool:
        return self.data == REAL_TEXT

    @property
    def pretrained_repo(self) -> tuple[str, str]:
        """``(repo, revision)`` of the pinned checkpoint; raises otherwise."""
        if not self.pretrained:
            raise WorkloadSpecError(f"weights are {self.weights!r}, not pretrained")
        repo, _, revision = self.weights[len("pretrained:"):].partition("@")
        return repo, revision

    def validate(self) -> "WorkloadSpec":
        super().validate()
        # Exactly two identities. Real text needs the checkpoint's tokenizer, and
        # pretrained weights on random token ids measure nothing a report could
        # name, so the pins travel together.
        if (self.weights, self.data) not in (("random", "synthetic"), (PRETRAINED, REAL_TEXT)):
            raise WorkloadSpecError(
                f"weights={self.weights!r}, data={self.data!r}: Llama real text "
                f"requires the pinned checkpoint and WikiText-2 together "
                f"(weights={PRETRAINED!r}, data={REAL_TEXT!r}), or neither")
        if self.pretrained and self.arch != dict(LLAMA_3_2_1B):
            raise WorkloadSpecError(
                "the pinned checkpoint exists only at the published architecture; "
                "a reduced-layer or reshaped run cannot load it")
        return self

    @property
    def workload_id(self) -> str:
        """The shared slug, with ``.pretrained.realtext`` before the hash on real text."""
        base = super().workload_id
        if not self.real_text:
            return base
        head, _, digest = base.rpartition(".")
        return f"{head}.pretrained.realtext.{digest}"

    def to_dict(self) -> dict[str, Any]:
        # Only when non-default, so the canonical synthetic hash -- the binding of
        # the tracked harvest snapshot and every stored calibration -- is unchanged.
        payload = super().to_dict()
        if self.weights != "random":
            payload["weights"] = self.weights
        if self.data != "synthetic":
            payload["data"] = self.data
        return payload


#: The reference execution. Every Llama-3 number comes from a run whose spec
#: equals this one.
CANONICAL = WorkloadSpec().validate()
