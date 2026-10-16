"""What every Gate 2 scope adapter shares: numerics settings and samples."""

from __future__ import annotations

import contextlib
from typing import Any, Iterable

import torch

from evograd.verification.common.execution import Sample


class ScopeBase:
    """Defaults for the parts of :class:`~evograd.verification.gate2.runner.ScopeAdapter`
    that are the same for every scope."""

    # Subclasses set ``scope`` and ``device``. Plain (unannotated) defaults
    # here, so a dataclass subclass does not inherit them as field defaults.
    #: Autocast dtype name, or ``None`` for none. Part of the policy binding.
    autocast_dtype = None
    training = True

    def numerics_context(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(self.device).type,
                              dtype=getattr(torch, self.autocast_dtype))

    def reference_parts(self) -> dict[str, Any]:
        """The objects that define the native reference: native sites, the
        module factory, output/loss selection, the input recipe. Subclasses
        name theirs; see :meth:`reference_identity`."""
        raise NotImplementedError(f"{type(self).__name__} does not declare its reference")

    def reference_identity(self) -> dict[str, Any]:
        """Implementation fingerprints of the reference, frozen into a policy.

        Raises ``UnidentifiableReference`` when a part cannot be identified.
        """
        from evograd.verification.common.fingerprint import fingerprint_parts

        return fingerprint_parts({"adapter": type(self), **self.reference_parts()})

    def base_numerics(self) -> dict[str, Any]:
        return {"device_type": torch.device(self.device).type, "autocast": self.autocast_dtype,
                "module_mode": "train" if self.training else "eval", "grad_mode": "enabled"}


def seeded_samples(prefix: str, seeds: Iterable[int], **config: Any) -> list[Sample]:
    """Samples named ``<prefix>/seed<k>``; the generator config travels with each."""
    return [Sample(f"{prefix}/seed{int(s)}", int(s), dict(config)) for s in seeds]


def seeded_cotangents(outputs: dict[str, torch.Tensor], seed: int, *, scale: float = 1.0
                      ) -> dict[str, torch.Tensor]:
    """One standard-normal cotangent per floating output, from a CPU generator."""
    generator = torch.Generator().manual_seed(int(seed) + 7919)
    out = {}
    for name in sorted(outputs):
        value = outputs[name]
        if torch.is_tensor(value) and value.is_floating_point() and value.requires_grad:
            draw = torch.randn(tuple(value.shape), generator=generator, dtype=torch.float32)
            out[name] = (draw * scale).to(device=value.device, dtype=value.dtype)
    return out
