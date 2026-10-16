"""The canonical OLMoE-1B-7B training workload: its architecture and its run settings.

What a workload spec *is* -- the frozen fields, the hashing, the three rules
``validate`` refuses to bend -- is the same for every model and lives in
:mod:`....common.spec`. What is OLMoE's is the architecture (the published
configuration, :mod:`...architecture`), the canonical batch, sequence, dtype,
device and attention backend, and one field the dense workloads do not have:

``experts_implementation``
    Which kernel runs the 64 experts. Transformers 5 dispatches the experts
    module the way it dispatches attention: ``"grouped_mm"`` sorts the routed
    tokens by expert and runs two grouped GEMMs (``torch.nn.functional.grouped_mm``,
    or torch's own fallback where it is unavailable); ``"eager"`` is the
    per-expert Python loop in ``OlmoeExperts.forward``. The default is the
    grouped GEMM because that is what an unmodified ``OlmoeForCausalLM`` runs
    on this torch, exactly as ``sdpa`` is the canonical attention backend and
    ``eager`` attention the alternative. The field is hashed into the workload
    identity and read back off the built model, so a run never silently
    measures the other kernel.

**Why batch 2 (decided 2026-10-16, confirmed on a GH200 the same day).** The
repository's usual B2 x T2048. Weights, gradients and AdamW's two bfloat16
moments are 13.8 + 13.8 + 27.7 GiB for 6.92B parameters. Measured in run
``20261016_gpu1`` on one 96 GB GH200: the Level-3 step (loss + backward, no
optimizer) peaks at 28.2 GiB allocated, and the Tier-3 step with AdamW at
64.5 GiB -- about 30 GiB of headroom, so B2 is the canonical batch and no
fallback to B1 was needed. The grouped-GEMM experts step runs at 241 ms, host-
submission-bound (``cpu_bound_fraction`` 0.97). Building the model initializes
13.8 GiB of weights on the host first.

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
    CANONICAL_EXPERTS,
    CANONICAL_SEQ,
    MODEL_NAME,
    OLMOE_1B_7B,
)

#: The experts kernels a Level-3 run may pin. ``batched_mm`` and the vendor
#: kernels Transformers also registers are not verified end to end here.
SUPPORTED_EXPERTS = ("grouped_mm", "eager")


@dataclass(frozen=True)
class WorkloadSpec(_WorkloadSpec):
    """One fully-determined OLMoE-1B-7B training execution.

    The defaults are the canonical run, which is what makes
    ``WorkloadSpec() == CANONICAL`` true and ``is_canonical`` meaningful.
    """

    model_name: str = MODEL_NAME
    arch_items: tuple[tuple[str, Any], ...] = tuple(sorted(OLMOE_1B_7B.items()))
    batch_size: int = CANONICAL_BATCH
    seq_len: int = CANONICAL_SEQ
    dtype: str = CANONICAL_DTYPE
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    #: Which kernel runs the experts; see the module docstring.
    experts_implementation: str = CANONICAL_EXPERTS

    @property
    def workload_id(self) -> str:
        """The shared id with the experts kernel spelled out before the digest,
        so two runs that differ only in it are told apart at a glance."""
        base = super().workload_id
        head, _, digest = base.rpartition(".")
        return f"{head}.{self.experts_implementation}.{digest}"

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["experts_implementation"] = self.experts_implementation
        return payload

    def validate(self) -> "WorkloadSpec":
        super().validate()
        if self.experts_implementation not in SUPPORTED_EXPERTS:
            raise WorkloadSpecError(
                f"experts_implementation={self.experts_implementation!r} is not a Level-3 "
                f"experts kernel for this milestone; choose one of {SUPPORTED_EXPERTS}. "
                f"The canonical workload is {CANONICAL_EXPERTS!r}."
            )
        arch = self.arch
        experts = arch.get("num_experts")
        top_k = arch.get("num_experts_per_tok")
        if not experts or not top_k:
            raise WorkloadSpecError("a routed-MoE architecture needs num_experts and "
                                    "num_experts_per_tok")
        if top_k > experts:
            raise WorkloadSpecError(f"num_experts_per_tok={top_k} exceeds num_experts={experts}")
        if arch.get("clip_qkv") is not None:
            raise WorkloadSpecError(
                "clip_qkv must be None: the qkv site is declared without the clamp, and "
                "OLMoE-1B-7B publishes none"
            )
        return self


#: The reference execution. Every OLMoE Level-3 number comes from a run whose
#: spec equals this one.
CANONICAL = WorkloadSpec().validate()

__all__ = ["CANONICAL", "OLMOE_1B_7B", "SUPPORTED_ATTENTION", "SUPPORTED_DTYPES",
           "SUPPORTED_EXPERTS", "WorkloadSpec", "WorkloadSpecError", "analytic_parameter_count"]
