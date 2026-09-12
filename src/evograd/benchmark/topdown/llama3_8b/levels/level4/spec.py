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
Hub token. The workload never needs one, because it builds from this
configuration with random weights and no tokenizer.

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

The package is still named ``llama3_8b`` and its tasks ``llama3_*``. Those names
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

#: The name this package has always used for its architecture. Kept so nothing
#: that imports it has to change; it now holds Llama-3.2-1B.
LLAMA_3_8B: Mapping[str, Any] = LLAMA_3_2_1B

MODEL_NAME = "Llama-3.2-1B"


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined Llama-3.2-1B training execution.

    The defaults are the canonical run, which is what makes
    ``WorkloadSpec() == CANONICAL`` true and ``is_canonical`` meaningful.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(LLAMA_3_8B.items()))
    batch_size: int = 2
    seq_len: int = 2048
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"


#: The reference execution. Every Llama-3 number comes from a run whose spec
#: equals this one.
CANONICAL = WorkloadSpec().validate()
