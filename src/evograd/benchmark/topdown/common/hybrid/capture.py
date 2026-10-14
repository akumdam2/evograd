"""Taking one site invocation out of a live decoder block, for any hybrid model.

A Level-2 site's real case is what one invocation of it received, produced and
was handed back inside a block's forward and VJP. The model package says
*which* tensors those are (its sites' ``capture.py``); this module owns the
mechanics -- hooks that clone and never retain the graph, function patches that
are always restored, and the record of one call.

Nothing here changes what the block computes: hooks observe, gradient hooks
return ``None``, and a patched function calls the original with the original
arguments.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch


@dataclass
class DerivedCall:
    """One real invocation of one site, as the declared task names its parts."""

    site: str
    task: str
    layer_index: int
    #: Declared argument name -> tensor (detached copy) or scalar.
    inputs: dict[str, Any]
    #: Declared upstream-gradient name -> the gradient the block delivered.
    upstream: dict[str, torch.Tensor]
    #: Declared output name -> what the native block produced at this site.
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Where the block came from: ``"captured"`` (a layer artifact) or ``"config"``.
    source: str = "config"

    def shapes(self) -> dict[str, tuple[int, ...]]:
        found = {k: tuple(v.shape) for k, v in self.inputs.items() if torch.is_tensor(v)}
        found.update({k: tuple(v.shape) for k, v in self.upstream.items()})
        found.update({k: tuple(v.shape) for k, v in self.outputs.items()})
        return found


def keep(tensor: torch.Tensor) -> torch.Tensor:
    """A detached copy with the tensor's exact strides (layout is evidence).

    ``clone()`` keeps strides only for a dense tensor; a view with gaps (a
    channel slice of a larger buffer) would come back packed.
    """
    out = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()),
                              dtype=tensor.dtype, device=tensor.device)
    out.copy_(tensor.detach())
    return out


class Recorder:
    """Values and gradients captured during one block run, by key."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.grads: dict[str, torch.Tensor] = {}

    def value(self, key: str, tensor: Any) -> None:
        if key in self.values:
            raise RuntimeError(f"{key} was captured twice in one block call")
        self.values[key] = keep(tensor) if torch.is_tensor(tensor) else tensor

    def grad_of(self, key: str, tensor: torch.Tensor) -> None:
        """Record ``tensor``'s gradient when backward reaches it."""
        if not tensor.requires_grad:
            raise RuntimeError(f"{key} does not require grad; no gradient to capture")

        def hook(grad, _key=key):
            if _key in self.grads:
                raise RuntimeError(f"gradient of {_key} arrived twice")
            self.grads[_key] = grad.detach().clone()
            return None

        tensor.register_hook(hook)

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.values and k not in self.grads]
        if missing:
            raise RuntimeError(f"the block run did not produce {missing}")


@contextlib.contextmanager
def patched(namespace: Any, attribute: str, factory: Callable[[Callable], Callable]) -> Iterator[None]:
    """Replace ``namespace.attribute`` for the duration; always restored."""
    original = getattr(namespace, attribute)
    setattr(namespace, attribute, factory(original))
    try:
        yield
    finally:
        setattr(namespace, attribute, original)


@contextlib.contextmanager
def hooks(*handles) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


__all__ = ["DerivedCall", "Recorder", "hooks", "keep", "patched"]
