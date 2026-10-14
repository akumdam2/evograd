"""What every Qwen3.5 Level-2 declaration shares.

The canonical case of each site is *config-derived*: its dims come from the
published configuration through :data:`evograd.opdecl.models.QWEN3_5_4B`
(``hf_config`` provenance, re-derived by ``tests/test_provenance``) at the
canonical B1 x T2048, not from the harvest snapshot. The whole-model harvest
checks those dims against what the step actually ran
(``harvest.snapshot.check_declared_dims``); it does not supply them, so every
declaration imports on a machine that has never run a harvest.

Tolerances: every site is calibrated (the generated block in its ``task.py``,
``scripts/qwen3_5_4b/calibration.py``). :data:`PROVISIONAL` is the repository's
standard float32/bfloat16 pair a new site starts from before calibration.
"""

from __future__ import annotations

from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import QWEN3_5_4B

WORKLOAD = "qwen3_5_4b"
OBSERVED_SUITE = "qwen3_5_4b_observed"
CONFIG = QWEN3_5_4B

#: The canonical invocation every site case is cut from (batch 1; see ``levels/level3/spec.py``).
BATCH = 1
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
        note=(f"{what}: every width reduced from Qwen3.5-4B's so the correctness cases "
              "run on CPU; head-count ratios, layouts and the absence of biases are "
              "preserved"),
        strides=tuple(strides),
    )


def token_major(batch: int, tokens: int, heads: int, dim: int) -> tuple[int, ...]:
    """``[B, H, T, D]`` viewed out of a contiguous ``[B, T, H, D]`` buffer."""
    return (tokens * heads * dim, dim, heads * dim, 1)


def head_major(batch: int, tokens: int, heads: int, dim: int) -> tuple[int, ...]:
    """``[B, T, H, D]`` viewed out of a contiguous ``[B, H, T, D]`` buffer."""
    return (heads * tokens * dim, dim, tokens * dim, 1)


def seed_of(dims: dict[str, int], salt: int = 0) -> int:
    seed = salt
    for name in sorted(dims):
        seed = (seed * 131 + int(dims[name])) % (2**31 - 1)
    return seed


__all__ = ["BATCH", "CONFIG", "OBSERVED_SUITE", "PROVISIONAL", "SEQ", "TOKENS", "WORKLOAD",
           "head_major", "observed", "seed_of", "shrunk", "token_major"]
