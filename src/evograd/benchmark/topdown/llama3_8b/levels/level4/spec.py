"""The canonical Llama-3 training workload: its architecture and run settings.

What a workload spec *is* -- the frozen fields, the hashing, the three rules
``validate`` refuses to bend -- is the same for every model and lives in
:mod:`....common.spec`. What is Llama's is the architecture below and the
canonical batch, sequence, dtype, device and attention backend, which are the
field defaults of :class:`WorkloadSpec`.

The architecture is Meta-Llama-3-8B's published configuration, written out
rather than downloaded. Two reasons, and the second is the one that matters
here: a reference must be reproducible on a node with no network, and
Llama-3 is a **gated repository** -- fetching its config would require an
authenticated Hub token. The workload never needs one, because it builds from
this configuration with random weights and no tokenizer.

Sequence length 2048 rather than the 8192 the architecture permits, so the
observed shapes line up with the Qwen3 workload's and the two can be read side
by side. It is a canonical choice, not a limit: ``max_position_embeddings`` is
8192 and ``validate`` enforces it.
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

#: Meta-Llama-3-8B, as ``meta-llama/Meta-Llama-3-8B``'s ``config.json`` publishes
#: it. 8.03B parameters total, 6.98B excluding the untied embedding and lm_head,
#: which ``analytic_parameter_count`` reproduces from these numbers alone.
#:
#: ``head_dim`` is stated explicitly. The published config omits it -- it is
#: ``hidden_size // num_attention_heads`` -- but the harvest records it per
#: configuration and a derived value that later moves would change shapes
#: without changing this file.
LLAMA_3_8B: Mapping[str, Any] = {
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-05,
    # 500000, not the 10000 Llama-2 uses. Getting this wrong produces a RoPE
    # kernel that is self-consistent and completely wrong.
    "rope_theta": 500000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    # Unlike Qwen3-0.6B, Llama-3-8B does *not* tie its embeddings: the lm_head
    # is a second 128256x4096 matrix, and it is a third of the non-layer
    # parameters.
    "tie_word_embeddings": False,
    "initializer_range": 0.02,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
}

MODEL_NAME = "Meta-Llama-3-8B"


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined Llama-3-8B training execution.

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
