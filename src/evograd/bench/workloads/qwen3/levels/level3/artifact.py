"""The layer-replay artifact: everything needed to re-run one decoder layer alone.

A harvest manifest says *what* the canonical run invoked. It cannot say whether
one of those invocations can be lifted out and executed on its own -- that needs
the actual numbers. This artifact is those numbers: the arguments Transformers
really passed to one ``Qwen3DecoderLayer``, its weights, its output, the upstream
gradient the full-model backward really delivered to it, and the gradients that
backward produced. With those, a separate process can construct one layer and
reproduce the full model's behaviour at that point exactly.

Three properties the format is built for.

**Nothing references the live graph.** Every tensor is detached, cloned and moved
to CPU at the moment of capture. A capture that held a view into the running
model would keep its activations alive and change the memory behaviour of the run
it is describing.

**The structure of the call is preserved, not flattened.** Transformers passes
``position_embeddings`` as a tuple of two tensors and ``attention_mask`` as
``None``; a format that stored only "the tensors" would lose the fact that the
mask was absent, which is precisely what makes SDPA take its causal path.

**Content is hashed logically, not as a file.** ``torch.save`` output is not
byte-reproducible, but the numbers in it are. :func:`content_hash` walks the
payload in a fixed order and hashes tensor bytes and scalar values, so two
captures of the same deterministic workload produce the same hash even though
their files differ.

Two hashes, not one, because there are two distinct failure modes.
:func:`content_hash` covers the numbers alone and catches a corrupted or
truncated file. :func:`artifact_hash` binds those numbers to the schema version
and to the provenance identity, and catches a file whose *label* was edited --
a correct capture relabelled as a different layer, workload or manifest would
pass a content check and be exactly as wrong.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

#: Bumped from /1 when the identity-bound artifact hash was added: a /1 file
#: carries no ``artifact_hash``, so it cannot be checked the way a consumer
#: now requires and is refused rather than half-verified.
SCHEMA_VERSION = "evograd-qwen3-layer-replay/2"

#: Payload keys that hold captured numbers, in the order the hash walks them.
CONTENT_KEYS = (
    "args",
    "kwargs",
    "output",
    "grad_output",
    "grad_input",
    "state_dict",
    "param_grads",
)


class ArtifactError(RuntimeError):
    """The artifact is malformed, or holds something that cannot be replayed."""


# --------------------------------------------------------------------------
# moving captured values between devices without losing their structure
# --------------------------------------------------------------------------

_PASSTHROUGH = (bool, int, float, str, type(None))


def to_cpu(value: Any, *, where: str = "$") -> Any:
    """Detach, clone and move to CPU, preserving containers.

    Anything that is not a tensor, a plain scalar or a container of those is
    refused by name rather than pickled: an opaque object in a captured argument
    would replay as a different call, and a silent success there is worse than a
    failure here.
    """
    if torch.is_tensor(value):
        if not value.is_contiguous():
            raise ArtifactError(
                f"{where}: captured tensor is not contiguous (stride "
                f"{tuple(value.stride())}); layout restoration for non-contiguous "
                "captures is not implemented, and silently making it contiguous "
                "would replay a different call"
            )
        return value.detach().clone().to("cpu")
    if isinstance(value, _PASSTHROUGH):
        return value
    if isinstance(value, tuple):
        return tuple(to_cpu(v, where=f"{where}[{i}]") for i, v in enumerate(value))
    if isinstance(value, list):
        return [to_cpu(v, where=f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: to_cpu(v, where=f"{where}.{k}") for k, v in value.items()}
    raise ArtifactError(
        f"{where}: cannot persist a {type(value).__name__}. The replay would have "
        "to invent a substitute for it, so the capture stops here instead."
    )


def to_device(value: Any, device: str) -> Any:
    """The inverse. Dtypes are carried by the tensors themselves."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------
# metadata and hashing
# --------------------------------------------------------------------------


def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": tensor.device.type,
        "requires_grad": bool(tensor.requires_grad),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
    }


def describe(value: Any) -> Any:
    """A JSON-safe description of a captured value, keeping its structure."""
    if torch.is_tensor(value):
        return {"kind": "tensor", **tensor_meta(value)}
    if isinstance(value, _PASSTHROUGH):
        return {"kind": "none"} if value is None else {"kind": "scalar", "value": value}
    if isinstance(value, (tuple, list)):
        return {"kind": "sequence", "items": [describe(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "mapping", "items": {k: describe(v) for k, v in sorted(value.items())}}
    return {"kind": "opaque", "type": type(value).__name__}


def _feed(hasher, value: Any, path: str) -> None:
    hasher.update(path.encode("utf-8"))
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu").contiguous()
        hasher.update(f"tensor|{tuple(tensor.shape)}|{tensor.dtype}".encode("utf-8"))
        # Reinterpret as bytes rather than going through numpy, which has no
        # bfloat16. This hashes the exact stored bits, so BF16 rounding cannot
        # make two logically identical captures hash differently.
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, _PASSTHROUGH):
        hasher.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    elif isinstance(value, (tuple, list)):
        hasher.update(f"seq|{len(value)}".encode("utf-8"))
        for index, item in enumerate(value):
            _feed(hasher, item, f"{path}[{index}]")
    elif isinstance(value, dict):
        hasher.update(f"map|{len(value)}".encode("utf-8"))
        for key in sorted(value):
            _feed(hasher, value[key], f"{path}.{key}")
    else:  # pragma: no cover - to_cpu refuses these first
        raise ArtifactError(f"{path}: {type(value).__name__} cannot be hashed")


def content_hash_over(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    """The same logical hash, over an arbitrary set of payload sections.

    Derived artifacts (a task extracted from a layer capture) hold different
    sections but need identical hashing semantics, so the walk lives here once.
    """
    hasher = hashlib.sha256()
    hasher.update(str(payload.get("schema_version")).encode("utf-8"))
    for key in keys:
        if key not in payload:
            raise ArtifactError(f"artifact is missing the required section {key!r}")
        _feed(hasher, payload[key], f"${key}")
    return hasher.hexdigest()


def identity_hash_over(
    payload: dict[str, Any], identity_keys: tuple[str, ...]
) -> str:
    """Bind a payload's content hash to its schema and provenance identity."""
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ArtifactError("artifact has no identity section")
    missing = [key for key in identity_keys if key not in identity]
    if missing:
        raise ArtifactError(f"artifact identity is missing {missing}")
    bound = {
        "schema_version": payload.get("schema_version"),
        "identity": {key: identity[key] for key in identity_keys},
        "content_hash": payload.get("content_hash"),
    }
    return hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    """A hash of the captured numbers, independent of how they were written.

    Covers only :data:`CONTENT_KEYS` -- the tensors and the argument structure.
    Identity (workload, manifest, layer) lives in the metadata and is checked on
    its own, so a corrupted artifact and a mislabelled one fail differently.
    """
    hasher = hashlib.sha256()
    hasher.update(SCHEMA_VERSION.encode("utf-8"))
    for key in CONTENT_KEYS:
        if key not in payload:
            raise ArtifactError(f"artifact is missing the required section {key!r}")
        _feed(hasher, payload[key], f"${key}")
    return hasher.hexdigest()


# --------------------------------------------------------------------------
# file IO
# --------------------------------------------------------------------------


#: Identity fields the artifact hash binds. Fixed here rather than taken from
#: whatever keys happen to be present, so adding a field cannot silently drop it
#: out of the hash.
IDENTITY_KEYS = (
    "workload_id",
    "workload_hash",
    "config_hash",
    "manifest_hash",
    "layer_index",
    "module_path",
    "event_ordinal",
    "module_class",
    "provenance_kind",
)


def artifact_hash(payload: dict[str, Any]) -> str:
    """Bind the captured numbers to the schema and the provenance identity.

    An artifact whose ``identity`` was edited -- a layer-14 capture relabelled
    as layer 9, or as belonging to a different manifest -- has intact content and
    a valid content hash. Only a hash that covers the identity catches it.
    """
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ArtifactError("artifact has no identity section")
    missing = [key for key in IDENTITY_KEYS if key not in identity]
    if missing:
        raise ArtifactError(f"artifact identity is missing {missing}")
    bound = {
        "schema_version": payload.get("schema_version"),
        "identity": {key: identity[key] for key in IDENTITY_KEYS},
        "content_hash": payload.get("content_hash") or content_hash(payload),
    }
    return hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class LayerArtifact:
    payload: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return self.payload["identity"]

    @property
    def arch(self) -> dict[str, Any]:
        return self.payload["arch"]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload, path)
        return path

    @classmethod
    def load(cls, path: Path, *, verify: bool = True) -> "LayerArtifact":
        """Load and, by default, verify both hashes.

        ``weights_only=True``: the payload is tensors, plain scalars, and the
        containers holding them, so nothing here needs arbitrary unpickling, and
        an artifact is a file a consumer may not have produced itself.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(
                f"{path}: schema {payload.get('schema_version')!r}, expected "
                f"{SCHEMA_VERSION!r}. Artifacts written before the identity-bound "
                "hash was added must be recaptured, not reinterpreted."
            )
        artifact = cls(payload)
        if verify:
            artifact.verify()
        return artifact

    def verify(self) -> dict[str, str]:
        """Both hashes, always together. There is no way to check one and skip
        the other, because either alone leaves a whole class of wrong file
        passing."""
        return {
            "content_hash": self.verify_content(),
            "artifact_hash": self.verify_artifact(),
        }

    def verify_artifact(self) -> str:
        recomputed = artifact_hash(self.payload)
        stored = self.payload.get("artifact_hash")
        if stored != recomputed:
            raise ArtifactError(
                f"artifact identity hash mismatch: stored {stored}, recomputed "
                f"{recomputed}. The provenance identity or the schema version has "
                "been edited; the numbers may be intact but the label is not."
            )
        return recomputed

    def verify_content(self) -> str:
        """Recompute the content hash and check it against the stored one."""
        recomputed = content_hash(self.payload)
        stored = self.payload.get("content_hash")
        if stored != recomputed:
            raise ArtifactError(
                f"artifact content hash mismatch: stored {stored}, recomputed "
                f"{recomputed}. The file has been modified or truncated."
            )
        return recomputed

    def verify_identity(
        self,
        *,
        workload_id: str | None = None,
        config_hash: str | None = None,
        manifest_hash: str | None = None,
        layer_index: int | None = None,
    ) -> None:
        identity = self.identity
        for field, expected in (
            ("workload_id", workload_id),
            ("config_hash", config_hash),
            ("manifest_hash", manifest_hash),
            ("layer_index", layer_index),
        ):
            if expected is None:
                continue
            actual = identity.get(field)
            if actual != expected:
                raise ArtifactError(
                    f"artifact {field} is {actual!r}, expected {expected!r}; this "
                    "artifact does not describe the requested execution"
                )


def load_canonical(
    path: Path,
    *,
    layer_index: int | None = None,
    snapshot_path: Path | None = None,
) -> LayerArtifact:
    """Load an artifact and require it to be *the* canonical one.

    The single entry point for anything downstream that claims to be derived
    from the canonical execution. It verifies both hashes and all four identity
    fields against the tracked snapshot, and takes no argument that turns any of
    that off -- a consumer cannot accidentally build a Level-2 task on a debug
    capture, or on the right numbers with the wrong label.
    """
    from ...harvest.snapshot import load as load_snapshot

    snapshot = load_snapshot(snapshot_path)
    artifact = LayerArtifact.load(path)  # verifies content and artifact hashes
    artifact.verify_identity(
        workload_id=snapshot["workload_id"],
        config_hash=snapshot["config_hash"],
        manifest_hash=snapshot["manifest_hash"],
        layer_index=(
            snapshot["representative_layer"]["layer_index"]
            if layer_index is None
            else layer_index
        ),
    )
    return artifact
