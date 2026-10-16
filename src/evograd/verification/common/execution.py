"""The execution vocabulary Gate 2 and Gate 3 share.

* :class:`Sample` -- one input draw, with a stable id and seed. Calibration and
  validation samples are told apart by id and seed, never by position.
* :class:`Deployment` -- which implementation each replaceable site runs:
  ``native`` (no site replaced), a trusted ``control``, or a ``candidate``.
  The patch set is the set of replaced sites, nothing else.
* :class:`SiteHooks` -- wraps each replaced site's callable to count its calls
  and, when asked, record every call at its real boundary (inputs as passed,
  outputs, and the upstream gradient that reached them).
* :class:`Invocation` and tensor paths (``args[0]``, ``kwargs.x``) -- how a
  scope names its inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from evograd.verification.common.tensors import isolate, tensors_in

NATIVE = "native"
CONTROL = "control"
CANDIDATE = "candidate"
ROLES = (NATIVE, CONTROL, CANDIDATE)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    seed: int
    #: How the sample was generated (recipe, offsets, data source). Recorded.
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "seed": int(self.seed), "config": dict(self.config)}


@dataclass(frozen=True)
class Deployment:
    name: str
    role: str
    #: site -> the callable installed there. Empty for the native deployment.
    sites: Mapping[str, Callable] = field(default_factory=dict)
    identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"deployment role must be one of {ROLES}, got {self.role!r}")
        if self.role == NATIVE and self.sites:
            raise ValueError("the native deployment replaces no site")
        if self.role != NATIVE and not self.sites:
            raise ValueError(f"a {self.role} deployment must replace at least one site")

    @property
    def patch_set(self) -> tuple[str, ...]:
        return tuple(sorted(self.sites))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "patch_set": list(self.patch_set),
                "identity": dict(self.identity)}

    @staticmethod
    def native(identity: Mapping[str, Any] | None = None) -> "Deployment":
        return Deployment("native", NATIVE, {}, dict(identity or {"kind": "native_eager"}))


def candidate(name: str, sites: Mapping[str, Callable], **identity: Any) -> Deployment:
    return Deployment(name, CANDIDATE, dict(sites), identity)


# ── site hooks ───────────────────────────────────────────────────────────────


@dataclass
class BoundaryCall:
    """One call of a replaced site, as the surrounding computation made it."""

    site: str
    index: int
    #: ``(args, kwargs)`` exactly as passed, isolated before the call.
    inputs: tuple[tuple[Any, ...], dict[str, Any]]
    #: Paths into ``inputs`` of floating tensors that required grad at the call.
    grad_paths: tuple[str, ...]
    #: The call's outputs, detached copies.
    outputs: Any
    #: Output path -> the gradient that reached it in the backward.
    upstream: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Input path -> the gradient this call's own backward actually emitted
    #: for that argument (captured on a per-call identity view, so other
    #: branches and other calls of a shared input are not mixed in).
    emitted: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Input paths whose emitted gradient could not be isolated.
    unattributable: tuple[str, ...] = ()
    #: Whether any output was differentiable at the call.
    differentiable_outputs: bool = False
    #: Outputs whose external cotangent could not be isolated.
    unattributable_outputs: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.site}#{self.index}"


class SiteHooks:
    """Counts calls of installed site callables; optionally records boundaries."""

    def __init__(self, *, record: bool = False):
        self.record = record
        self.counts: dict[str, int] = {}
        self.calls: list[BoundaryCall] = []
        self.enabled = True

    def wrap(self, site: str, fn: Callable) -> Callable:
        hooks = self

        def site_call(*args, **kwargs):
            if not hooks.enabled:
                return fn(*args, **kwargs)
            index = hooks.counts.get(site, 0)
            hooks.counts[site] = index + 1
            if not hooks.record:
                return fn(*args, **kwargs)
            roots: dict[str, Any] = {"args": tuple(args), "kwargs": dict(kwargs)}
            grad_paths = tuple(path for path, t in tensors_in(roots)
                               if t.requires_grad and (t.is_floating_point() or t.is_complex()))
            inputs = isolate((tuple(args), dict(kwargs)))
            emitted: dict[str, torch.Tensor] = {}
            unattributable = []
            for path in grad_paths:
                # Each differentiable argument reaches the site through its own
                # identity view. The view's gradient is exactly what this call's
                # backward emitted for that argument; ViewBackward then hands it
                # on unchanged, so the surrounding gradient flow, the layout and
                # any declared in-place reuse of the storage are as before.
                try:
                    local = resolve(roots, path).view_as(resolve(roots, path))
                except Exception:
                    unattributable.append(path)
                    continue
                local.register_hook(capture_grad(emitted, path))
                roots = replace_at(roots, path, local)
            out = fn(*roots["args"], **roots["kwargs"])
            call = BoundaryCall(site, index, inputs, grad_paths, isolate(_detached(out)),
                                emitted=emitted, unattributable=tuple(unattributable),
                                differentiable_outputs=any(
                                    t.requires_grad for _, t in tensors_in(out)))
            # A returned tensor may also feed another output *inside* an eager
            # site. Hooking that original tensor observes internal + external
            # gradients, and seeding a replay with that total counts the internal
            # path twice. Return identity views and hook only those boundary
            # edges. Repeated references to the same output reuse one view.
            views: dict[int, torch.Tensor] = {}
            unsupported = []
            returned = {"out": out}
            for path, t in tensors_in(out, "out"):
                if not t.requires_grad:
                    continue
                if id(t) not in views:
                    try:
                        view = t.view_as(t)
                    except Exception:
                        unsupported.append(path)
                        continue
                    view.register_hook(capture_grad(call.upstream, path))
                    views[id(t)] = view
                returned = replace_at(returned, path, views[id(t)])
            call.unattributable_outputs = tuple(unsupported)
            hooks.calls.append(call)
            return returned["out"]

        site_call.__name__ = f"site_{site}"
        site_call.__wrapped__ = fn
        return site_call

    def reset(self) -> None:
        self.counts.clear()
        self.calls.clear()

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def capture_grad(store: dict[str, torch.Tensor], path: str):
    """A tensor hook storing (accumulating) the gradient under ``path``."""
    def hook(grad):
        if grad is None:   # an unused output of an autograd.Function
            return None
        store[path] = grad.detach().clone() if path not in store else store[path] + grad.detach()
        return None
    return hook


def _detached(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detached(v) for v in value)
    if isinstance(value, list):
        return [_detached(v) for v in value]
    if isinstance(value, dict):
        return {k: _detached(v) for k, v in value.items()}
    return value


# ── invocations and paths ────────────────────────────────────────────────────


@dataclass
class Invocation:
    """One call of a scope's module.

    ``differentiable_inputs`` maps a stable result name to a path into
    ``args``/``kwargs``; each becomes a fresh leaf and its gradient is a
    required result. ``mutable_inputs`` are paths whose *contents* the
    contract allows the computation to overwrite (``cotangent:<output>`` for
    an upstream gradient); their layout must still survive.
    """

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    differentiable_inputs: Mapping[str, str] = field(default_factory=dict)
    mutable_inputs: tuple[str, ...] = ()
    #: Anything else the scope needs to read its outputs (labels, metadata).
    context: dict[str, Any] = field(default_factory=dict)


_TOKEN = re.compile(r"\.(\w+)|\[(\d+)\]")


def parse_path(path: str) -> tuple[str, tuple[Any, ...]]:
    match = re.match(r"^(\w+)", path)
    if not match:
        raise ValueError(f"malformed path {path!r}")
    steps: list[Any] = []
    rest, position = path[match.end():], 0
    for token in _TOKEN.finditer(rest):
        if token.start() != position:
            raise ValueError(f"malformed path {path!r}")
        steps.append(token.group(1) if token.group(1) is not None else int(token.group(2)))
        position = token.end()
    if position != len(rest):
        raise ValueError(f"malformed path {path!r}")
    return match.group(1), tuple(steps)


def resolve(roots: Mapping[str, Any], path: str) -> Any:
    root, steps = parse_path(path)
    value = roots[root]
    for step in steps:
        value = value[step]
    return value


def replace_at(roots: dict[str, Any], path: str, new: Any) -> dict[str, Any]:
    root, steps = parse_path(path)

    def put(value: Any, steps: tuple[Any, ...]) -> Any:
        if not steps:
            return new
        head, tail = steps[0], steps[1:]
        if isinstance(value, tuple):
            items = list(value)
            items[head] = put(items[head], tail)
            return tuple(items)
        if isinstance(value, list):
            items = list(value)
            items[head] = put(items[head], tail)
            return items
        if isinstance(value, dict):
            items = dict(value)
            items[head] = put(items[head], tail)
            return items
        raise ValueError(f"{path}: cannot index a {type(value).__name__}")

    out = dict(roots)
    out[root] = put(roots[root], steps)
    return out


def describe_tree(value: Any) -> Any:
    """A JSON-safe structural description: tensors by shape/dtype, scalars by value."""
    if torch.is_tensor(value):
        return {"tensor": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [describe_tree(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): describe_tree(v) for k, v in value.items()}
    if hasattr(value, "to_tuple") and hasattr(value, "keys"):   # transformers ModelOutput
        return {str(k): describe_tree(value[k]) for k in value.keys()}
    return {"object": type(value).__name__}


def content_identity(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """A canonical, content-addressed description of an input tree.

    Returns ``(identity, opaque_paths)``. Tensors are described by the content
    hash of their bytes plus shape, dtype, strides and storage offset (the
    layout a kernel may depend on); scalars, strings, ``None``, dtypes and
    devices by value; containers recursively. Anything else cannot be told
    apart by content: its path is listed in ``opaque_paths`` and the identity
    carries only its type, so a caller can refuse to bind it.
    """
    from evograd.verification.common.tensors import tensor_digest

    opaque: list[str] = []

    def walk(v: Any, p: str) -> Any:
        if torch.is_tensor(v):
            strided = v.layout == torch.strided
            return {"tensor": tensor_digest(v), "shape": list(v.shape),
                    "dtype": str(v.dtype).removeprefix("torch."),
                    "stride": list(v.stride()) if strided else None,
                    "storage_offset": v.storage_offset() if strided else None,
                    "layout": str(v.layout).removeprefix("torch.")}
        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, float):
            return {"float": repr(v)}
        if isinstance(v, (torch.dtype, torch.device)):
            return {type(v).__name__: str(v)}
        if isinstance(v, (tuple, list)):
            return {type(v).__name__: [walk(x, f"{p}[{i}]") for i, x in enumerate(v)]}
        if isinstance(v, Mapping):
            keys = sorted(v, key=str)
            if not all(isinstance(k, str) for k in keys):
                opaque.append(p or "<root>")
            return {"dict": {str(k): walk(v[k], f"{p}.{k}" if p else str(k)) for k in keys}}
        opaque.append(p or "<root>")
        return {"opaque": f"{type(v).__module__}.{type(v).__qualname__}"}

    return walk(value, path), opaque
