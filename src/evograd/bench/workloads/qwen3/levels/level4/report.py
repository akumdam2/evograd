"""The smoke report: everything needed to decide whether a run is trustworthy.

A report is written even when the run fails, because "it failed, here, with this
message, on this GPU, at this Transformers version" is the answer most often
wanted and the hardest to reconstruct afterwards.

Every timing and memory number lives under ``diagnostics`` and carries a note
saying so. This milestone establishes that the execution works; it does not
measure it. A peak-memory figure from a single unwarmed step is a sanity check
on the shape of the allocation, not a benchmark result, and putting it anywhere
else in the schema would invite it to be quoted as one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "evograd-qwen3-smoke/1"

STATUS_OK = "ok"
STATUS_FAILED = "failed"


@dataclass
class SmokeReport:
    """One machine-readable record of one attempted canonical execution."""

    workload: dict[str, Any]
    environment: dict[str, Any]
    status: str = STATUS_OK
    failure: str | None = None
    effective: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "failure": self.failure,
            "workload": self.workload,
            "environment": self.environment,
            "effective": self.effective,
            "result": self.result,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SmokeReport":
        return cls(
            workload=payload["workload"],
            environment=payload["environment"],
            status=payload.get("status", STATUS_OK),
            failure=payload.get("failure"),
            effective=payload.get("effective", {}),
            result=payload.get("result", {}),
            diagnostics=payload.get("diagnostics", {}),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "SmokeReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        """The three lines a human wants before opening the JSON."""
        result = self.result
        lines = [
            f"[{self.status}] {self.workload.get('workload_id')}"
            + ("" if self.workload.get("canonical") else "   (NON-CANONICAL)"),
        ]
        if self.failure:
            lines.append(f"  failure: {self.failure}")
        if result:
            lines.append(
                "  loss={loss} finite={finite}  grads {have}/{trainable}"
                "  all-finite={gfin}".format(
                    loss=result.get("loss"),
                    finite=result.get("loss_is_finite"),
                    have=result.get("params_with_grad"),
                    trainable=result.get("trainable_params"),
                    gfin=result.get("grads_all_finite"),
                )
            )
        peak = self.diagnostics.get("peak_allocated_bytes")
        if peak is not None:
            lines.append(f"  peak allocated {peak / 2**30:.2f} GiB (diagnostic only)")
        return "\n".join(lines)
