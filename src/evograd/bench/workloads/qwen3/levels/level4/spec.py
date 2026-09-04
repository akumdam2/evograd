"""The canonical Qwen3 training workload: its architecture and its run settings.

What a workload spec *is* -- the frozen fields, the hashing, the three rules
``validate`` refuses to bend -- is the same for every model and lives in
:mod:`...common.spec`. What is Qwen3's is the architecture below and the
canonical batch, sequence, dtype, device and attention backend, which are the
field defaults of :class:`WorkloadSpec`.

The architecture is Qwen3-0.6B's published configuration, written out rather
than downloaded: the reference must be reproducible on a node with no network,
and a config fetched at runtime would make the workload identity depend on what
the Hub served that day.
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

#: Qwen3-0.6B, exactly as published (``Qwen/Qwen3-0.6B`` ``config.json``).
#: 596M parameters total, 440M excluding the tied embedding -- the "0.6B" and
#: "0.44B non-embedding" of the model card, which
#: ``analytic_parameter_count`` reproduces from these numbers alone.
QWEN3_0_6B: Mapping[str, Any] = {
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": None,
    "use_sliding_window": False,
    "max_window_layers": 28,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}

MODEL_NAME = "Qwen3-0.6B"


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined Qwen3-0.6B training execution.

    The defaults are the canonical run, which is what makes
    ``WorkloadSpec() == CANONICAL`` true and ``is_canonical`` meaningful.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(QWEN3_0_6B.items()))
    batch_size: int = 2
    seq_len: int = 2048
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"


#: The reference execution. Every number this milestone reports comes from a run
#: whose spec equals this one.
CANONICAL = WorkloadSpec().validate()
