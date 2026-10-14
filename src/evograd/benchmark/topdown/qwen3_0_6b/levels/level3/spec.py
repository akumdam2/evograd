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
    #: Where the parameters come from. ``"random"`` is the synthetic canonical
    #: workload: ``nn.init`` under ``seed``. ``"pretrained:<repo>@<revision>"``
    #: loads a pinned checkpoint. Part of the identity, so a calibration taken on
    #: random weights can never be applied to pretrained ones.
    weights: str = "random"
    #: Where the tokens come from. ``"synthetic"`` draws them from ``seed``;
    #: ``"wikitext-2-raw:<revision>"`` is real text, tokenised and packed by
    #: ``evaluation.tier3.textdata``. Also part of the identity.
    data: str = "synthetic"

    @property
    def pretrained(self) -> bool:
        return self.weights.startswith("pretrained:")

    @property
    def real_text(self) -> bool:
        return self.data != "synthetic"

    @property
    def pretrained_repo(self) -> tuple[str, str]:
        """``(repo, revision)`` of a pretrained checkpoint; raises otherwise."""
        if not self.pretrained:
            raise WorkloadSpecError(f"weights are {self.weights!r}, not pretrained")
        repo, _, revision = self.weights[len("pretrained:"):].partition("@")
        if not repo or not revision:
            raise WorkloadSpecError(
                f"pretrained weights must be 'pretrained:<repo>@<revision>', got "
                f"{self.weights!r}")
        return repo, revision

    @property
    def workload_id(self) -> str:
        """The shared slug, with ``.pretrained.realtext`` before the hash when the
        weights or data are not the synthetic canonical ones."""
        base = super().workload_id
        if not (self.pretrained or self.real_text):
            return base
        head, _, digest = base.rpartition(".")
        return (f"{head}.{'pretrained' if self.pretrained else 'random'}."
                f"{'realtext' if self.real_text else 'synthetic'}.{digest}")

    def to_dict(self) -> dict[str, Any]:
        # Only when non-default, so the canonical synthetic hash -- the binding of
        # every stored calibration -- is unchanged by these two fields.
        payload = super().to_dict()
        if self.weights != "random":
            payload["weights"] = self.weights
        if self.data != "synthetic":
            payload["data"] = self.data
        return payload


#: The reference execution. Every number this milestone reports comes from a run
#: whose spec equals this one.
CANONICAL = WorkloadSpec().validate()
