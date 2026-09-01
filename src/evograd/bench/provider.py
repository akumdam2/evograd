"""The provider boundary: one way to call a candidate or a baseline.

Every timing protocol in evograd measures at least two implementations of the
same mathematics and divides one time by the other. That ratio is only
meaningful when both sides are *invoked* the same way, so the thing being
compared is the kernel rather than the glue around it.

:class:`PairProvider` is that boundary. A provider exposes

    forward(values)             -> (output, saved)
    backward(dout, saved, values) -> gradients per ``op.grad_names()``

and nothing else. A candidate seed, a declared Liger pair, and eager PyTorch
autograd all reduce to it, after which the protocol driving them cannot tell
them apart — which is the property that makes the division fair.

This module holds no timing code on purpose. It was extracted from
``bench.tier1`` so that protocols other than the level-1 pair benchmark can
reuse the boundary and the input-mutation guards instead of restating them;
``fair`` re-exports every name for existing callers.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from evograd.bench.harness import describe_saved, normalize_saved
from evograd.opdecl.activity import OpDecl
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.oracle import resolve_runtime_forward


@dataclass(frozen=True)
class PairProvider:
    """A thin provider boundary consumed identically by the common runner."""

    name: str
    forward: Callable[[dict[str, Any]], tuple[Any, tuple[Any, ...]]]
    backward: Callable[[torch.Tensor, tuple[Any, ...], dict[str, Any]], Any]
    source_hash: str
    adapter_kind: str


@dataclass(frozen=True)
class TensorSnapshot:
    name: str
    #: ``None`` when the declaration permits the backward to overwrite this
    #: input, in which case only the metadata below is enforced.
    value: torch.Tensor | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int


def _callable_hash(*functions: Callable) -> str:
    digest = hashlib.sha256()
    for function in functions:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = repr(function)
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def candidate_provider(op: OpDecl, module) -> PairProvider:
    forward_fn, backward_fn = lookup_pair(op, module)
    arg_names = tuple(arg.name for arg in op.args)

    def forward(values):
        output, saved = forward_fn(*(values[name] for name in arg_names))
        return output, normalize_saved(saved)

    def backward(dout, saved, values):
        kwargs = backward_inactive_kwargs(op, backward_fn, values)
        return backward_fn(dout, saved, **kwargs)

    return PairProvider(
        name="candidate",
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_fn, backward_fn),
        adapter_kind="candidate_pair",
    )


def declared_provider(op: OpDecl, name: str) -> PairProvider:
    """Wrap any declaration-provided pair baseline in the common boundary.

    Named rather than hard-coded to ``liger``: the level-3 protein block is
    meant to be compared against MegaFold's kernels, and ``matmul`` already
    declares a cuBLAS pair, so the final-report protocol has to reach every
    declared baseline rather than one privileged name.
    """
    try:
        hook = op.performance_baselines[name]
    except KeyError:
        available = sorted(op.performance_baselines)
        raise ValueError(
            f"{op.name}: no {name!r} pair baseline is declared; "
            f"available: {available or 'none'}"
        ) from None
    factory = getattr(hook, "pair_factory", None)
    if factory is None:
        raise ValueError(
            f"{op.name}: the {name!r} baseline does not expose a pair factory, "
            "so it cannot be measured under the symmetric protocol"
        )
    forward_fn, backward_fn = factory()
    forward_args = tuple(getattr(hook, "forward_args", ()))
    backward_extras = tuple(getattr(hook, "backward_extras", ()))

    def forward(values):
        output, saved = forward_fn(*(values[name_] for name_ in forward_args))
        return output, normalize_saved(saved)

    def backward(dout, saved, values):
        return backward_fn(
            dout,
            saved,
            *(values[name_] for name_ in backward_extras),
        )

    # layernorm's adapter calls Liger's shipped entry points directly rather
    # than through a declaration-local wrapper; the report records which, so a
    # reader can tell how much glue sat between the kernel and the timer.
    if name == "liger":
        kind = "stock_liger_pair" if op.name == "layernorm" else "declared_liger_pair"
    else:
        kind = f"declared_{name}_pair"

    return PairProvider(
        name=name,
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_fn, backward_fn),
        adapter_kind=kind,
    )


def liger_provider(op: OpDecl) -> PairProvider:
    """Backwards-compatible alias for the Liger baseline."""
    return declared_provider(op, "liger")


def pytorch_autograd_provider(op: OpDecl) -> PairProvider:
    """Expose eager PyTorch autograd through the common provider boundary.

    Timed through ``runtime_forward`` when the declaration has one — see
    ``resolve_runtime_forward``. The oracle keeps using ``forward`` regardless;
    only what gets timed changes.
    """
    forward_ref = resolve_runtime_forward(op)
    arg_names = tuple(arg.name for arg in op.args)
    active_args = tuple(op.active_args())
    active_names = tuple(arg.name for arg in active_args)
    by_grad = {arg.grad_name: arg.name for arg in active_args}

    def forward(values):
        for name in active_names:
            value = values[name]
            if not value.requires_grad:
                value.requires_grad_(True)
        output = forward_ref(*(values[name] for name in arg_names))
        return output, (output,)

    def backward(dout, saved, values):
        (output,) = saved
        gradients = torch.autograd.grad(
            output,
            tuple(values[name] for name in active_names),
            dout,
            retain_graph=False,
            create_graph=False,
        )
        by_name = dict(zip(active_names, gradients))
        return tuple(by_name[by_grad[name]] for name in op.grad_names())

    return PairProvider(
        name="pytorch_autograd",
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_ref, forward, backward),
        adapter_kind="pytorch_eager_autograd",
    )


def renamed_provider(provider: PairProvider, name: str) -> PairProvider:
    """Use the exact same callables under another label for identity controls."""
    return PairProvider(
        name=name,
        forward=provider.forward,
        backward=provider.backward,
        source_hash=provider.source_hash,
        adapter_kind=provider.adapter_kind,
    )


def clone_values(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }


def snapshot_tensors(
    values: dict[str, Any], *, may_overwrite: tuple[str, ...] = ()
) -> tuple[TensorSnapshot, ...]:
    """Record inputs so a provider can be shown not to have changed them.

    Inputs the declaration lists in ``backward_may_overwrite`` are recorded by
    metadata only: their contents may legitimately be replaced by the gradient,
    but the buffer must keep its shape, strides, dtype and storage offset — a
    reused buffer is still the same buffer. Skipping their clone also removes
    the copy, which matters when the input is large.
    """
    return tuple(
        TensorSnapshot(
            name=name,
            value=None if name in may_overwrite else value.detach().clone(),
            shape=tuple(value.shape),
            stride=tuple(value.stride()),
            dtype=value.dtype,
            storage_offset=value.storage_offset(),
        )
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    )


def assert_tensors_unchanged(
    values: dict[str, Any],
    snapshots: tuple[TensorSnapshot, ...],
    *,
    provider: str,
) -> None:
    for snapshot in snapshots:
        current = values[snapshot.name]
        metadata_ok = (
            tuple(current.shape) == snapshot.shape
            and tuple(current.stride()) == snapshot.stride
            and current.dtype == snapshot.dtype
            and current.storage_offset() == snapshot.storage_offset
        )
        if not metadata_ok:
            raise RuntimeError(
                f"{provider}: changed the shape, strides, dtype or storage "
                f"offset of benchmark input {snapshot.name!r}"
            )
        # value is None for inputs the declaration allows the backward to
        # overwrite; their contents are exempt, their identity is not.
        if snapshot.value is not None and not torch.equal(current, snapshot.value):
            raise RuntimeError(
                f"{provider}: mutated benchmark input {snapshot.name!r}"
            )


def saved_state_report(
    saved: tuple[Any, ...], values: dict[str, Any]
) -> dict[str, Any]:
    input_storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    }
    unique_storages: dict[int, int] = {}
    retained_input_storages: dict[int, int] = {}
    logical_bytes = 0
    for value in saved:
        if not isinstance(value, torch.Tensor):
            continue
        logical_bytes += value.numel() * value.element_size()
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        unique_storages[pointer] = storage.nbytes()
        if pointer in input_storages:
            retained_input_storages[pointer] = input_storages[pointer]
    return {
        "logical_saved_bytes": logical_bytes,
        "unique_saved_storage_bytes": sum(unique_storages.values()),
        "retained_input_bytes": sum(retained_input_storages.values()),
        "saved_tensors": describe_saved(saved),
    }

