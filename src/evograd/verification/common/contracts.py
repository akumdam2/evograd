"""Named result contracts, and the one place results are matched by name.

A contract is the list of results a verification must compare. It comes from
the declaration and the trusted reference -- never from what the candidate
happened to return -- so a result the candidate leaves out is a missing result,
not a smaller comparison.

Every result has a stable name, a kind, and a presence rule:

``required``  both sides must produce it;
``optional``  it may be absent from both; if the reference produces it, the
              candidate must too, and a candidate may not invent it;
``absent``    the reference does not produce it (a parameter that takes no part
              in the computation), and the candidate must not either.

Matching is by name. Nothing here zips two sequences, so a short return can
never truncate the comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from evograd.verification.common.reports import Finding, failure, missing

OUTPUT = "output"
INPUT_GRAD = "input_grad"
PARAM_GRAD = "param_grad"
METADATA = "metadata"
KINDS = (OUTPUT, INPUT_GRAD, PARAM_GRAD, METADATA)

#: Which floor family each kind is calibrated against (Gate 2).
FLOOR_FAMILY = {OUTPUT: "output", INPUT_GRAD: "gradient", PARAM_GRAD: "gradient"}

REQUIRED = "required"
OPTIONAL = "optional"
ABSENT = "absent"
PRESENCE = (REQUIRED, OPTIONAL, ABSENT)

#: Name prefixes, one per kind. Shared by every scope so a report reads the same.
PREFIX = {OUTPUT: "out:", INPUT_GRAD: "grad_in:", PARAM_GRAD: "grad:", METADATA: "meta:"}


class ContractError(ValueError):
    """A result does not have the structure its contract declares."""


@dataclass(frozen=True)
class ResultSpec:
    name: str
    kind: str
    presence: str = REQUIRED
    #: Compared for exact equality rather than numerically. Always true for a
    #: non-floating result; set explicitly for metadata.
    exact: bool = False
    #: Other names bound to the same tensor (tied parameters). The result is
    #: the single accumulated value; the aliases are part of the contract.
    aliases: tuple[str, ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a result needs a name")
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.presence not in PRESENCE:
            raise ValueError(f"{self.name}: presence must be one of {PRESENCE}, "
                             f"got {self.presence!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "presence": self.presence,
                "exact": self.exact, "aliases": list(self.aliases),
                "shape": None if self.shape is None else list(self.shape),
                "dtype": self.dtype}


@dataclass(frozen=True)
class ResultContract:
    specs: tuple[ResultSpec, ...]

    def __post_init__(self) -> None:
        names = [s.name for s in self.specs]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"result names must be unique; repeated {dup}")
        aliases = [a for s in self.specs for a in s.aliases]
        clash = sorted(set(aliases) & set(names))
        if clash or len(aliases) != len(set(aliases)):
            raise ValueError(f"aliases must be unique and distinct from result names: {clash}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def get(self, name: str) -> ResultSpec | None:
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"results": [s.to_dict() for s in self.specs]}

    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def compared(self) -> tuple[ResultSpec, ...]:
        """Specs that carry a value to compare (``absent`` ones carry none)."""
        return tuple(s for s in self.specs if s.presence != ABSENT)


# ── matching ─────────────────────────────────────────────────────────────────

_NOTHING = object()


def produced(value: Any) -> bool:
    return value is not _NOTHING and value is not None


@dataclass
class Pairing:
    spec: ResultSpec
    candidate: Any
    reference: Any


def match_results(contract: ResultContract, candidate: Mapping[str, Any],
                  reference: Mapping[str, Any]) -> tuple[list[Pairing], list[Finding]]:
    """Line up both sides by contract name; report everything that does not fit.

    Returns the pairs that must be compared numerically or exactly, and the
    findings for presence violations. A missing candidate result is a
    ``failed`` finding; a reference that lacks a required result, or produces
    one it should not, is ``incomplete`` -- the reference is not valid for this
    contract and nothing can be concluded from it.
    """
    pairs: list[Pairing] = []
    findings: list[Finding] = []
    for spec in contract.specs:
        cand = candidate.get(spec.name, _NOTHING)
        ref = reference.get(spec.name, _NOTHING)
        if spec.exact or spec.kind == METADATA:
            # An exact result's value may legitimately be ``None`` (attention
            # weights not requested, say); only an absent key is missing.
            has_c, has_r = cand is not _NOTHING, ref is not _NOTHING
        else:
            has_c, has_r = produced(cand), produced(ref)
        if spec.presence == ABSENT:
            if has_r:
                findings.append(missing("reference_contract_mismatch",
                                        "the reference produced a result the contract says is absent",
                                        subject=spec.name))
            if has_c:
                findings.append(failure("unexpected_result",
                                        "the candidate produced a result the reference does not "
                                        "(e.g. a gradient for a parameter that takes no part)",
                                        subject=spec.name))
            continue
        if not has_r:
            if spec.presence == REQUIRED:
                findings.append(missing("reference_missing_result",
                                        "the reference did not produce a required result",
                                        subject=spec.name))
            elif has_c:
                findings.append(failure("unexpected_result",
                                        "the candidate produced an optional result the reference lacks",
                                        subject=spec.name))
            continue
        if not has_c:
            findings.append(failure("missing_result", "the candidate did not produce this result",
                                    subject=spec.name, kind=spec.kind))
            continue
        pairs.append(Pairing(spec, cand, ref))
    known = set(contract.names)
    for name in candidate:
        if name not in known and produced(candidate[name]):
            findings.append(failure("unexpected_result",
                                    "the candidate produced a result no contract entry names",
                                    subject=name))
    for name in reference:
        if name not in known and produced(reference[name]):
            findings.append(missing("reference_contract_mismatch",
                                    "the reference produced a result no contract entry names",
                                    subject=name))
    return pairs, findings


# ── structure of one pair ────────────────────────────────────────────────────


def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": dtype_name(tensor.dtype),
            "stride": list(tensor.stride()) if tensor.layout == torch.strided else None,
            "device": str(tensor.device)}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def is_float_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())


@dataclass
class Structure:
    """What the contract says about one pair before any number is compared."""

    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    #: ``numeric`` -- compare numerically; ``exact`` -- compared here already;
    #: ``none`` -- a finding stopped the comparison.
    mode: str = "none"


def check_structure(pair: Pairing) -> Structure:
    """Type, shape, dtype, device, finiteness and exact equality for one result.

    Non-floating results (and those the contract marks ``exact``) are settled
    here by exact equality. A non-finite reference makes the result
    ``incomplete``; a non-finite candidate fails it. Both are checked, so a
    case where both are wrong records both.
    """
    spec, cand, ref = pair.spec, pair.candidate, pair.reference
    out = Structure()
    name = spec.name
    if not torch.is_tensor(ref):
        out.meta["reference"] = {"type": type(ref).__name__}
        out.meta["candidate"] = ({"type": type(cand).__name__} if not torch.is_tensor(cand)
                                 else tensor_meta(cand))
        if torch.is_tensor(cand):
            out.findings.append(failure("type_mismatch", "a tensor where the reference has a value",
                                        subject=name))
            return out
        if not exact_equal(cand, ref):
            out.findings.append(failure("exact_mismatch",
                                        f"value {cand!r} differs from the reference's {ref!r}",
                                        subject=name))
        out.mode = "exact"
        return out
    out.meta["reference"] = tensor_meta(ref)
    if not torch.is_tensor(cand):
        out.meta["candidate"] = {"type": type(cand).__name__}
        out.findings.append(failure("type_mismatch", f"expected a tensor, got {type(cand).__name__}",
                                    subject=name))
        return out
    out.meta["candidate"] = tensor_meta(cand)
    if spec.shape is not None and tuple(ref.shape) != tuple(spec.shape):
        out.findings.append(missing("reference_contract_mismatch",
                                    f"reference shape {tuple(ref.shape)} differs from the declared "
                                    f"{tuple(spec.shape)}", subject=name))
    if spec.dtype is not None and dtype_name(ref.dtype) != spec.dtype:
        out.findings.append(missing("reference_contract_mismatch",
                                    f"reference dtype {dtype_name(ref.dtype)} differs from the "
                                    f"declared {spec.dtype}", subject=name))
    if tuple(cand.shape) != tuple(ref.shape):
        out.findings.append(failure("shape_mismatch",
                                    f"shape {tuple(cand.shape)} != reference {tuple(ref.shape)}",
                                    subject=name))
    if cand.dtype != ref.dtype:
        out.findings.append(failure("dtype_mismatch",
                                    f"dtype {dtype_name(cand.dtype)} != reference "
                                    f"{dtype_name(ref.dtype)}", subject=name))
    if cand.device != ref.device:
        out.findings.append(failure("device_mismatch",
                                    f"device {cand.device} != reference {ref.device}", subject=name))
    exact = spec.exact or not is_float_tensor(ref)
    ref_finite = (torch.isfinite(ref.detach()) if is_float_tensor(ref)
                  else torch.ones(ref.shape, dtype=torch.bool, device=ref.device))
    if not bool(ref_finite.all()):
        out.findings.append(missing("invalid_reference", "the reference contains non-finite values",
                                    subject=name, elements=int((~ref_finite).sum())))
    if is_float_tensor(cand) and not bool(torch.isfinite(cand.detach()).all()):
        # The candidate is at fault for non-finite values where the reference
        # is finite. Where the reference itself is not, there is no evidence
        # either way -- that is already recorded as an invalid reference.
        own = ~torch.isfinite(cand.detach())
        if cand.shape == ref.shape:
            own = own & ref_finite.to(own.device)
        if bool(own.any()):
            out.findings.append(failure("non_finite_candidate",
                                        "the candidate has non-finite values where the reference "
                                        "is finite", subject=name, elements=int(own.sum())))
    if out.findings:
        return out
    if exact:
        if not bitwise_equal(cand, ref):
            out.findings.append(failure("exact_mismatch", "an exact result differs from the reference",
                                        subject=name))
        out.mode = "exact"
        return out
    out.mode = "numeric"
    return out


def exact_equal(a: Any, b: Any) -> bool:
    """Exact equality through containers: tensors bitwise, everything else by
    type and ``==``. Never raises; an uncomparable pair is unequal."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and bitwise_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping):
        return set(a) == set(b) and all(exact_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return False


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bits. NaN equals NaN here; ``torch.equal`` would not."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    a = a.detach().reshape(-1)
    b = b.detach().to(a.device).reshape(-1)
    if a.dtype == torch.bool:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


# ── shared parameters ────────────────────────────────────────────────────────


def parameter_groups(module: torch.nn.Module) -> list[tuple[str, ...]]:
    """Parameter names grouped by object, in registration order.

    ``named_parameters()`` drops the second name of a tied parameter. Tying is
    part of the contract (the tied gradient is one accumulated sum), so every
    name is kept: the first is the result name, the rest are its aliases.
    """
    groups: dict[int, list[str]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    return [tuple(names) for names in groups.values()]


def unique_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(module.named_parameters())


# ── stable identity ──────────────────────────────────────────────────────────


def stable_hash(payload: Any) -> str:
    """SHA-256 of a canonical JSON rendering. Floats are written with repr."""
    text = json.dumps(payload, sort_keys=True, default=repr, allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment_fingerprint(device: str | torch.device | None = None) -> dict[str, Any]:
    """What can move a number without a file changing. Bound into Gate 2 policy.

    Kept here, with no model or vendor import, so a policy is reproducible from
    this package alone.
    """
    import platform

    fingerprint: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    kind = torch.device(device).type if device is not None else "cpu"
    fingerprint["device_type"] = kind
    if kind == "cuda" and torch.cuda.is_available():
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        fingerprint["device_name"] = torch.cuda.get_device_name(index)
        fingerprint["compute_capability"] = f"{props.major}.{props.minor}"
        fingerprint["sdpa_backends"] = {
            name: bool(getattr(torch.backends.cuda, f"{name}_sdp_enabled")())
            for name in ("flash", "mem_efficient", "math", "cudnn")
            if hasattr(torch.backends.cuda, f"{name}_sdp_enabled")
        }
    else:
        fingerprint["device_name"] = platform.processor() or platform.machine()
        fingerprint["cpu_threads"] = torch.get_num_threads()
    return fingerprint


def environment_differences(stored: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(stored) | set(live))
    return [f"{k}: policy {stored.get(k)!r}, now {live.get(k)!r}"
            for k in keys if stored.get(k) != live.get(k)]


def names_of(specs: Iterable[ResultSpec]) -> list[str]:
    return [s.name for s in specs]
