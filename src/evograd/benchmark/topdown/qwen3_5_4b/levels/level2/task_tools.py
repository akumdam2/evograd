"""What every Qwen3.5 Level-2 declaration shares.

The canonical case of each site is *config-derived*: its dims come from the
published configuration through :data:`evograd.opdecl.models.QWEN3_5_4B`
(``hf_config`` provenance, re-derived by ``tests/test_provenance``) at the
canonical B1 x T2048, not from the harvest snapshot. The whole-model harvest
checks those dims against what the step actually ran
(``harvest.snapshot.check_declared_dims``); it does not supply them, so every
declaration imports on a machine that has never run a harvest.
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


__all__ = ["BATCH", "CONFIG", "OBSERVED_SUITE", "SEQ", "TOKENS", "WORKLOAD",
           "head_major", "observed", "seed_of", "token_major"]
