"""Candidate representation: GEPA evolves only the EVOLVE-BLOCK body."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

START = "# EVOLVE-BLOCK-START"
END = "# EVOLVE-BLOCK-END"

_FORBIDDEN = (
    "torch.autograd",
    "torch.nn.functional.layer_norm",
    "torch.layer_norm",
    "torch.compile",
    "liger_kernel",
)
_REQUIRED = (
    "def layernorm_forward_with_saved",
    "def layernorm_backward_from_saved",
)


@dataclass(frozen=True)
class EvolveBlockTemplate:
    prefix: str
    seed_body: str
    suffix: str

    @classmethod
    def from_source(cls, source: str) -> "EvolveBlockTemplate":
        if source.count(START) != 1 or source.count(END) != 1:
            raise ValueError("source must contain exactly one EVOLVE-BLOCK marker pair")
        prefix, remainder = source.split(START, 1)
        body, suffix = remainder.split(END, 1)
        return cls(prefix=prefix, seed_body=body.strip(), suffix=suffix)

    @classmethod
    def from_path(cls, path: str | Path) -> "EvolveBlockTemplate":
        return cls.from_source(Path(path).read_text(encoding="utf-8"))

    def validate_body(self, body: str) -> str:
        normalized = body.strip()
        if not normalized:
            raise ValueError("EVOLVE-BLOCK candidate is empty")
        if START in normalized or END in normalized:
            raise ValueError("candidate body must not contain EVOLVE-BLOCK markers")
        missing = [symbol for symbol in _REQUIRED if symbol not in normalized]
        if missing:
            raise ValueError(f"candidate is missing required public APIs: {missing}")
        forbidden = [token for token in _FORBIDDEN if token in normalized]
        if forbidden:
            raise ValueError(f"candidate uses forbidden high-level implementation: {forbidden}")
        return normalized

    def render(self, body: str) -> str:
        normalized = self.validate_body(body)
        return f"{self.prefix}{START}\n{normalized}\n{END}{self.suffix}"

    def assert_scope(self, full_source: str) -> None:
        other = self.from_source(full_source)
        if other.prefix != self.prefix or other.suffix != self.suffix:
            raise ValueError("candidate changed source outside EVOLVE-BLOCK")

    @staticmethod
    def digest(body: str) -> str:
        return hashlib.sha256(body.strip().encode()).hexdigest()
