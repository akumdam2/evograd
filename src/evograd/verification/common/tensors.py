"""Tensor isolation, mutation evidence, and the two numeric comparators.

**Isolation.** :func:`isolate` gives each side of a comparison its own copy of
the inputs. Copies are made per *storage*, and every tensor is re-viewed onto
its storage's copy with the original offset and strides, so a strided view
keeps its gaps and two inputs that alias one buffer still alias one buffer.

**Comparators.** Both run in float64, chunked, whatever the input dtype:

* :func:`elementwise_error` -- Gate 1's per-element allowance
  ``|c_i - r_i| <= atol + rtol * |r_i|`` (complex: modulus, as ``allclose``);
* :func:`relative_error` -- Gate 2's whole-tensor pair

      E2   = ||c - r||_2   / max(||r||_2,  EPSILON)
      Emax = ||c - r||_inf / max(RMS(r),   EPSILON)

Complex tensors are compared as complex128 (both parts). Both require finite
inputs of equal shape; callers settle structure and
finiteness first (:func:`evograd.verification.common.contracts.check_structure`).
An empty tensor compares as zero error and says ``empty``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

import torch

#: Denominator guard in E2/Emax. Not a threshold floor (see Gate 2 calibration).
EPSILON = 1e-30

#: Elements per float64 chunk. Bounds the comparator's extra memory at 32 MiB.
CHUNK = 1 << 22


# ── isolation ────────────────────────────────────────────────────────────────


def isolate(tree: Any) -> Any:
    """A deep copy of every tensor in ``tree`` that preserves layout and aliasing.

    Containers (tuple, list, dict) are rebuilt; anything else is shared. The
    copies are detached and do not require grad.
    """
    storages: dict[tuple[str, int], Any] = {}
    memo: dict[int, torch.Tensor] = {}

    def copy(t: torch.Tensor) -> torch.Tensor:
        if id(t) in memo:
            return memo[id(t)]
        src = t.detach()
        if src.layout != torch.strided or src.untyped_storage().nbytes() == 0:
            out = src.clone()
        else:
            storage = src.untyped_storage()
            key = (str(src.device), storage.data_ptr())
            fresh = storages.get(key)
            if fresh is None:
                fresh = storage.clone()
                storages[key] = fresh
            out = torch.empty((0,), dtype=src.dtype, device=src.device)
            out.set_(fresh, src.storage_offset(), src.size(), src.stride())
        memo[id(t)] = out
        return out

    def walk(value: Any) -> Any:
        if torch.is_tensor(value):
            return copy(value)
        if isinstance(value, tuple):
            return tuple(walk(v) for v in value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(tree)


def tensors_in(tree: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    """Every tensor in a tree, by a stable path."""
    if torch.is_tensor(tree):
        yield prefix, tree
    elif isinstance(tree, (tuple, list)):
        for i, v in enumerate(tree):
            yield from tensors_in(v, f"{prefix}[{i}]")
    elif isinstance(tree, dict):
        for k, v in tree.items():
            yield from tensors_in(v, f"{prefix}.{k}" if prefix else str(k))


@dataclass(frozen=True)
class _Snap:
    value: torch.Tensor | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    offset: int


def snapshot(tree: Any, *, metadata_only: tuple[str, ...] = ()) -> dict[str, _Snap]:
    """Contents and layout of every tensor in ``tree``, for :func:`mutations`.

    Paths named in ``metadata_only`` are tensors a contract allows to be
    overwritten: their bytes are not kept, their layout still is.
    """
    out = {}
    for path, t in tensors_in(tree):
        keep = path not in metadata_only and path.split("[")[0] not in metadata_only
        out[path] = _Snap(t.detach().clone() if keep else None, tuple(t.shape),
                          tuple(t.stride()) if t.layout == torch.strided else (),
                          t.dtype, t.storage_offset() if t.layout == torch.strided else 0)
    return out


def mutations(tree: Any, before: Mapping[str, _Snap]) -> list[str]:
    """Paths whose layout changed, or whose bytes changed without permission."""
    from evograd.verification.common.contracts import bitwise_equal

    now = dict(tensors_in(tree))
    changed = []
    for path, snap in before.items():
        t = now.get(path)
        if t is None:
            changed.append(path)
            continue
        layout = (tuple(t.shape), tuple(t.stride()) if t.layout == torch.strided else (),
                  t.dtype, t.storage_offset() if t.layout == torch.strided else 0)
        if layout != (snap.shape, snap.stride, snap.dtype, snap.offset):
            changed.append(path)
        elif snap.value is not None and not bitwise_equal(t.detach(), snap.value):
            changed.append(path)
    return changed


def tensor_digest(t: torch.Tensor) -> str:
    """Content hash of one tensor (bytes as stored, plus shape and dtype)."""
    import hashlib

    h = hashlib.sha256(f"{tuple(t.shape)}|{t.dtype}".encode())
    flat = t.detach().reshape(-1).contiguous().cpu()
    if flat.numel():
        h.update(flat.view(torch.uint8).numpy().tobytes() if flat.dtype != torch.bool
                 else flat.numpy().tobytes())
    return h.hexdigest()


def module_state_digest(module: torch.nn.Module) -> dict[str, str]:
    """Content hash per unique parameter and buffer, by name."""
    out = {f"param:{n}": tensor_digest(p) for n, p in module.named_parameters()}
    out.update({f"buffer:{n}": tensor_digest(b) for n, b in module.named_buffers()})
    return out


# ── RNG ──────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def isolated_rng(seed: int | None, device: str | torch.device | None = None):
    """Seed every generator for the block, and restore all of them afterwards.

    What runs inside cannot move the caller's random stream, and two runs that
    start from the same seed consume identical randomness.
    """
    kind = torch.device(device).type if device is not None else "cpu"
    devices = [torch.device(device).index or 0] if kind == "cuda" and torch.cuda.is_available() else []
    # An empty device list forks only the CPU generator and never touches CUDA.
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(int(seed))
        yield


def synchronize(device: str | torch.device | None) -> None:
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


# ── comparators ──────────────────────────────────────────────────────────────


def _wide(t: torch.Tensor) -> torch.Tensor:
    """float64, or complex128 for a complex tensor (both parts are kept)."""
    return t.to(torch.complex128 if t.is_complex() else torch.float64)


def _chunks(c: torch.Tensor, r: torch.Tensor):
    cf = c.detach().reshape(-1)
    rf = r.detach().to(c.device).reshape(-1)
    for start in range(0, cf.numel(), CHUNK):
        yield _wide(cf[start:start + CHUNK]), _wide(rf[start:start + CHUNK]), start


@dataclass(frozen=True)
class RelativeError:
    """Gate 2's two statistics for one tensor, and what they were built from."""

    E2: float
    Emax: float
    diff_l2: float
    diff_max: float
    ref_l2: float
    ref_rms: float
    elements: int
    reference_zero: bool
    bitwise: bool
    empty: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> RelativeError:
    """``E2`` and ``Emax`` of ``candidate`` against ``reference``, in float64.

    ``Emax`` is normalized by the reference's RMS rather than by any single
    element, so a near-zero reference element cannot blow it up. A reference
    that is exactly zero divides by ``EPSILON``: any nonzero candidate then has
    an enormous error, which is the defined behaviour, and ``reference_zero``
    says so.
    """
    if candidate.shape != reference.shape:
        raise ValueError(f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}")
    n = reference.numel()
    if n == 0:
        return RelativeError(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True)
    # Two passes: the maxima first, then sums of squares scaled by them, so
    # finite float64 values near the overflow limit cannot overflow the norm.
    diff_max = ref_max = 0.0
    for c, r, _ in _chunks(candidate, reference):
        diff_max = max(diff_max, float((c - r).abs().max()))
        ref_max = max(ref_max, float(r.abs().max()))
    if not (math.isfinite(diff_max) and math.isfinite(ref_max)):
        raise ValueError("relative_error needs finite tensors; settle finiteness first")
    scale = max(diff_max, ref_max)
    diff_sq = ref_sq = 0.0
    if scale > 0:
        for c, r, _ in _chunks(candidate, reference):
            d = (c - r).abs() / scale
            a = r.abs() / scale
            diff_sq += float((d * d).sum())
            ref_sq += float((a * a).sum())
    diff_l2, ref_l2 = scale * math.sqrt(diff_sq), scale * math.sqrt(ref_sq)
    ref_rms = scale * math.sqrt(ref_sq / n)
    return RelativeError(
        E2=diff_l2 / max(ref_l2, EPSILON), Emax=diff_max / max(ref_rms, EPSILON),
        diff_l2=diff_l2, diff_max=diff_max, ref_l2=ref_l2, ref_rms=ref_rms, elements=n,
        reference_zero=ref_max == 0.0, bitwise=diff_max == 0.0, empty=False)


@dataclass(frozen=True)
class ElementwiseError:
    """Gate 1's per-element allowance for one tensor."""

    violations: int
    elements: int
    max_abs_err: float
    #: ``max_i |c_i - r_i| / max(|r_i|, 1e-8)`` -- the legacy evaluator's figure.
    max_rel_err: float
    #: ``max_i |c_i - r_i| / (atol + rtol |r_i|)``; ``<= 1`` exactly when no
    #: element violates (an allowance of zero with zero error counts as 0).
    headroom: float
    worst_index: list[int] | None
    atol: float
    rtol: float

    @property
    def ok(self) -> bool:
        return self.violations == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def elementwise_error(candidate: torch.Tensor, reference: torch.Tensor, *,
                      atol: float, rtol: float) -> ElementwiseError:
    """Count elements outside ``atol + rtol * |reference|``, computed in float64."""
    if candidate.shape != reference.shape:
        raise ValueError(f"shape {tuple(candidate.shape)} != {tuple(reference.shape)}")
    if not (math.isfinite(atol) and math.isfinite(rtol)) or atol < 0 or rtol < 0:
        raise ValueError(f"tolerances must be finite and non-negative, got atol={atol}, rtol={rtol}")
    violations = 0
    max_abs = max_rel = headroom = 0.0
    worst_flat = None
    for c, r, start in _chunks(candidate, reference):
        d = (c - r).abs()
        allowance = atol + rtol * r.abs()
        bad = d > allowance
        violations += int(bad.sum())
        if d.numel():
            max_abs = max(max_abs, float(d.max()))
            max_rel = max(max_rel, float((d / r.abs().clamp_min(1e-8)).max()))
            ratio = torch.where(allowance > 0, d / allowance.clamp_min(1e-300),
                                torch.where(d > 0, torch.full_like(d, math.inf), torch.zeros_like(d)))
            local = int(ratio.argmax())
            value = float(ratio[local])
            if value > headroom or worst_flat is None:
                headroom, worst_flat = max(headroom, value), start + local
    # A difference that overflows float64 is a violation, counted as such;
    # its magnitude is reported as unbounded rather than raised.
    worst = None
    if worst_flat is not None and reference.numel():
        worst = [int(i) for i in torch.unravel_index(torch.tensor(worst_flat), tuple(reference.shape))]
    return ElementwiseError(violations=violations, elements=reference.numel(),
                            max_abs_err=max_abs, max_rel_err=max_rel,
                            headroom=headroom if math.isfinite(headroom) else float("inf"),
                            worst_index=worst, atol=float(atol), rtol=float(rtol))
