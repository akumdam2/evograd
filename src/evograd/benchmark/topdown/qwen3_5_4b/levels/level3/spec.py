"""The canonical Qwen3.5-4B training workload: its architecture and its run settings.

What a workload spec *is* -- the frozen fields, the hashing, the three rules
``validate`` refuses to bend -- is the same for every model and lives in
:mod:`....common.spec`. What is Qwen3.5's is the architecture (the published
``text_config``, :mod:`...architecture`) and the canonical batch, sequence,
dtype, device and attention backend, which are the field defaults of
:class:`WorkloadSpec`.

**Why batch 1, when every other workload here runs B2 x T2048 (decided
2026-10-10).** The eager training step does not fit one 96 GB GH200 at batch 2.
Measured on CPU as saved-for-backward bytes at B2 x T2048 in bfloat16: one
Gated DeltaNet layer saves 2.76 GiB (the torch chunk delta rule alone 1.67
GiB), one full-attention layer about 1.0 GiB, so the 32 layers' activations are
about 74 GiB; weights, gradients and AdamW's two moments in bfloat16 add 7.8 +
7.8 + 15.7 GiB, and the logits and cross entropy over the 248,320-wide
vocabulary (upcast to float32 by Transformers) another 6-8 GiB -- more than 105
GiB in all. B1 x T2048 is about 73 GiB. Offload and multiple GPUs were rejected
for the timed workload: activation offload puts host transfers inside the
timed step and moves a candidate's saved state off the card, and DDP would need
B1 per card anyway -- which makes single-card B1 x T2048 exactly the per-GPU
workload of a two-GPU data-parallel run at global batch 2, a real training
configuration in its own right. The batch and sequence stay parameters; the
first GPU session records one B2 attempt (expected out of memory) as evidence.

The architecture is written out rather than downloaded, so the workload identity
cannot depend on what the Hub served that day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....common.spec import (  # noqa: F401  (re-export)
    SUPPORTED_ATTENTION,
    SUPPORTED_DTYPES,
    WorkloadSpecError,
    analytic_parameter_count,
)
from ....common.spec import WorkloadSpec as _WorkloadSpec
from ...architecture import (
    CANONICAL_BATCH,
    CANONICAL_DTYPE,
    CANONICAL_SEQ,
    MODEL_NAME,
    QWEN3_5_4B,
    layer_types,
    with_layers,
)


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined Qwen3.5-4B (text) training execution.

    The defaults are the canonical run, which is what makes
    ``WorkloadSpec() == CANONICAL`` true and ``is_canonical`` meaningful.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(QWEN3_5_4B.items()))
    batch_size: int = CANONICAL_BATCH
    seq_len: int = CANONICAL_SEQ
    dtype: str = CANONICAL_DTYPE
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    #: Where the parameters come from. ``"random"`` is ``nn.init`` under
    #: ``seed``, through the model's own ``post_init``; ``"pretrained:<repo>@<revision>"``
    #: loads that pinned checkpoint's text model, as for Qwen3-0.6B.
    weights: str = "random"
    #: Where the tokens come from. ``"synthetic"`` draws them from ``seed``;
    #: ``"wikitext-2-raw:<revision>"`` is real text, packed as for Qwen3-0.6B.
    data: str = "synthetic"

    @property
    def pretrained(self) -> bool:
        return self.weights.startswith("pretrained:")

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
    def real_text(self) -> bool:
        return self.data != "synthetic"

    @property
    def workload_id(self) -> str:
        base = super().workload_id
        if not (self.pretrained or self.real_text):
            return base
        head, _, digest = base.rpartition(".")
        return (f"{head}.{'pretrained' if self.pretrained else 'random'}."
                f"{'realtext' if self.real_text else 'synthetic'}.{digest}")

    def to_dict(self) -> dict[str, Any]:
        # Only when non-default, so the canonical synthetic hash does not depend
        # on these two fields.
        payload = super().to_dict()
        if self.weights != "random":
            payload["weights"] = self.weights
        if self.data != "synthetic":
            payload["data"] = self.data
        return payload

    def replace(self, **overrides: Any) -> "WorkloadSpec":
        """As the shared ``replace``, except that a new ``num_hidden_layers``
        without ``layer_types`` cuts the written-out ``layer_types`` with it, so
        a debug depth keeps the published pattern instead of a config
        Transformers refuses."""
        arch = overrides.get("arch")
        if arch is not None and "num_hidden_layers" in arch and "layer_types" not in arch:
            rest = {k: v for k, v in dict(arch).items() if k != "num_hidden_layers"}
            overrides["arch"] = with_layers({**self.arch, **rest}, int(arch["num_hidden_layers"]))
        return super().replace(**overrides)

    def validate(self) -> "WorkloadSpec":
        super().validate()
        arch = self.arch
        try:
            layer_types(arch)
        except ValueError as exc:
            raise WorkloadSpecError(str(exc)) from None
        return self


#: The reference execution. Every Qwen3.5 Level-3 number comes from a run whose
#: spec equals this one.
CANONICAL = WorkloadSpec().validate()

__all__ = ["CANONICAL", "QWEN3_5_4B", "SUPPORTED_ATTENTION", "SUPPORTED_DTYPES",
           "WorkloadSpec", "WorkloadSpecError", "analytic_parameter_count"]
