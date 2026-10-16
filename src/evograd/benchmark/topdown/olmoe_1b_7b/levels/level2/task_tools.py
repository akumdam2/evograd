"""What every OLMoE Level-2 declaration shares.

The canonical case of each site is *config-derived*: its dims come from the
published configuration through :data:`evograd.opdecl.models.OLMOE_1B_7B`
(``hf_config`` provenance, re-derived by ``tests/test_provenance``) at the
canonical B2 x T2048, not from the harvest snapshot. The whole-model harvest
checks those dims against what the step actually ran
(``harvest.snapshot.check_declared_dims``); it does not supply them, so every
declaration imports on a machine that has never run a harvest.

Tolerances: **not calibrated yet.** Every site carries :data:`PROVISIONAL`, the
repository's standard float32/bfloat16 pair, and says so with
``PROVISIONAL_TOLERANCES = True``. The calibration inventory
(``evograd.evaluation.workloads.olmoe_1b_7b.level2.calibrate``) is the first
GPU session's job; the weight gradients of the experts site sum over hundreds
of tokens per expert at the observed size and are the ones most likely to
need a reduction-scaled gate.
"""

from __future__ import annotations

from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import OLMOE_1B_7B

WORKLOAD = "olmoe_1b_7b"
OBSERVED_SUITE = "olmoe_1b_7b_observed"
CONFIG = OLMOE_1B_7B

#: The canonical invocation every site case is cut from (see ``levels/level3/spec.py``).
BATCH = 2
SEQ = 2048
TOKENS = BATCH * SEQ

#: Not calibrated: the repository's standard
#: float32/bfloat16 defaults (``evograd.ops._common.STANDARD_TOLERANCES``).
PROVISIONAL = {"float32": (2e-5, 2e-5), "bfloat16": (8e-2, 8e-2)}


def observed(component: str, *, note: str = "", strides=None) -> tuple[Workload, ...]:
    """The site's canonical case, derived from the configuration.

    ``strides(dims) -> ((name, stride), ...)`` states the layouts the model
    hands this site (inputs and upstream gradients) at these dims.
    """
    free = {"batch": BATCH, "seq": SEQ}
    dims = getattr(CONFIG, f"{component}_dims")(**free)
    recorded = tuple(strides(dims)) if strides is not None else ()
    return (
        Workload(
            dims=dims,
            dtype="bfloat16",
            provenance=Provenance(model=WORKLOAD, component=component, free=free,
                                  source="hf_config", note=note, strides=recorded),
        ),
    )


def shrunk(what: str, strides=()) -> Provenance:
    """Provenance for a CPU-sized correctness case."""
    return Provenance(
        model=WORKLOAD, component=what, free={}, source="handpicked", scaled=True,
        note=(f"{what}: every width reduced from OLMoE-1B-7B's so the correctness cases "
              "run on CPU; head-count ratios, the expert count to top-k ratio, layouts and "
              "the absence of biases are preserved"),
        strides=tuple(strides),
    )


def token_major(batch: int, tokens: int, heads: int, dim: int) -> tuple[int, ...]:
    """``[B, H, T, D]`` viewed out of a contiguous ``[B, T, H, D]`` buffer."""
    return (tokens * heads * dim, dim, heads * dim, 1)


def seed_of(dims: dict[str, int], salt: int = 0) -> int:
    seed = salt
    for name in sorted(dims):
        seed = (seed * 131 + int(dims[name])) % (2**31 - 1)
    return seed


__all__ = ["BATCH", "CONFIG", "OBSERVED_SUITE", "PROVISIONAL", "SEQ", "TOKENS", "WORKLOAD",
           "observed", "seed_of", "shrunk", "token_major"]
