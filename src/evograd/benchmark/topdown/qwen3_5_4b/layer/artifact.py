"""The layer-replay artifact: everything needed to re-run one Qwen3.5 decoder layer alone.

The format is the dense workloads' (``qwen3_0_6b``/``llama3_2_1b``
``layer.artifact``): the arguments Transformers really passed to one
``Qwen3_5DecoderLayer``, its weights, its output, the upstream gradient the
full-model backward delivered to it, and the gradients that backward produced.
Every tensor is detached, cloned and moved to CPU at capture; the call's
structure (``position_embeddings`` as a tuple, ``attention_mask=None``) is kept;
content is hashed logically and bound to the provenance identity by a second
hash. Two things are Qwen3.5's: the schema string, and the identity, which also
names the layer's kind (``linear_attention`` or ``full_attention``) because one
harvest yields one artifact per kind.

The helpers are copied rather than imported: workload packages do not depend on
each other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "evograd-qwen3_5-layer-replay/1"

#: Payload keys that hold captured numbers, in the order the hash walks them.
CONTENT_KEYS = ("args", "kwargs", "output", "grad_output", "grad_input", "state_dict",
                "param_grads")

#: Identity fields the artifact hash binds.
IDENTITY_KEYS = ("workload_id", "workload_hash", "config_hash", "manifest_hash", "layer_index",
                 "layer_kind", "module_path", "event_ordinal", "module_class",
                 "provenance_kind")


class ArtifactError(RuntimeError):
    """The artifact is malformed, or holds something that cannot be replayed."""


_PASSTHROUGH = (bool, int, float, str, type(None))


def to_cpu(value: Any, *, where: str = "$") -> Any:
    """Detach, clone and move to CPU, preserving containers; refuse anything else."""
    if torch.is_tensor(value):
        if not value.is_contiguous():
            raise ArtifactError(
                f"{where}: captured tensor is not contiguous (stride {tuple(value.stride())}); "
                "silently making it contiguous would replay a different call")
        return value.detach().clone().to("cpu")
    if isinstance(value, _PASSTHROUGH):
        return value
    if isinstance(value, tuple):
        return tuple(to_cpu(v, where=f"{where}[{i}]") for i, v in enumerate(value))
    if isinstance(value, list):
        return [to_cpu(v, where=f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: to_cpu(v, where=f"{where}.{k}") for k, v in value.items()}
    raise ArtifactError(f"{where}: cannot persist a {type(value).__name__}")


def to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


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


def content_hash(payload: dict[str, Any]) -> str:
    """A hash of the captured numbers, independent of how they were written."""
    hasher = hashlib.sha256()
    hasher.update(SCHEMA_VERSION.encode("utf-8"))
    for key in CONTENT_KEYS:
        if key not in payload:
            raise ArtifactError(f"artifact is missing the required section {key!r}")
        _feed(hasher, payload[key], f"${key}")
    return hasher.hexdigest()


def artifact_hash(payload: dict[str, Any]) -> str:
    """Bind the captured numbers to the schema and the provenance identity."""
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
        """Load (tensors and plain containers only) and, by default, verify both hashes."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(f"{path}: schema {payload.get('schema_version')!r}, expected "
                                f"{SCHEMA_VERSION!r}")
        artifact = cls(payload)
        if verify:
            artifact.verify()
        return artifact

    def verify(self) -> dict[str, str]:
        return {"content_hash": self.verify_content(), "artifact_hash": self.verify_artifact()}

    def verify_artifact(self) -> str:
        recomputed = artifact_hash(self.payload)
        stored = self.payload.get("artifact_hash")
        if stored != recomputed:
            raise ArtifactError(f"artifact identity hash mismatch: stored {stored}, recomputed "
                                f"{recomputed}; the label has been edited")
        return recomputed

    def verify_content(self) -> str:
        recomputed = content_hash(self.payload)
        stored = self.payload.get("content_hash")
        if stored != recomputed:
            raise ArtifactError(f"artifact content hash mismatch: stored {stored}, recomputed "
                                f"{recomputed}; the file has been modified or truncated")
        return recomputed

    def verify_identity(self, *, workload_id: str | None = None, config_hash: str | None = None,
                        manifest_hash: str | None = None, layer_index: int | None = None) -> None:
        identity = self.identity
        for field, expected in (("workload_id", workload_id), ("config_hash", config_hash),
                                ("manifest_hash", manifest_hash), ("layer_index", layer_index)):
            if expected is None:
                continue
            if identity.get(field) != expected:
                raise ArtifactError(f"artifact {field} is {identity.get(field)!r}, expected "
                                    f"{expected!r}; it does not describe the requested execution")


def load_canonical(path: Path, *, layer_index: int | None = None,
                   snapshot_path: Path | None = None) -> LayerArtifact:
    """Load an artifact and require it to be one of *the* canonical captures.

    Both hashes, and the workload, config and manifest identity against the
    tracked snapshot; the layer must be one of the snapshot's representative
    layers (or exactly ``layer_index`` when given). No argument turns any of
    that off.
    """
    from ..harvest.snapshot import load as load_snapshot

    snapshot = load_snapshot(snapshot_path)
    artifact = LayerArtifact.load(path)
    artifact.verify_identity(workload_id=snapshot["workload_id"],
                             config_hash=snapshot["config_hash"],
                             manifest_hash=snapshot["manifest_hash"], layer_index=layer_index)
    allowed = {entry["layer_index"] for entry in snapshot["representative_layers"].values()}
    if artifact.identity["layer_index"] not in allowed:
        raise ArtifactError(f"{path} captures layer {artifact.identity['layer_index']}, which is "
                            f"not a representative layer {sorted(allowed)}")
    return artifact


__all__ = ["ArtifactError", "CONTENT_KEYS", "IDENTITY_KEYS", "LayerArtifact", "SCHEMA_VERSION",
           "artifact_hash", "content_hash", "describe", "load_canonical", "tensor_meta",
           "to_cpu", "to_device"]
