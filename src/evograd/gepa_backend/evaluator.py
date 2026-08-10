"""GEPA adapter: per-shape scores with a once-per-candidate global hard gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from evograd.gepa_backend.candidate import EvolveBlockTemplate
from evograd.ops import get_op


@dataclass(frozen=True)
class ShapeExample:
    id: str
    dims: dict[str, int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_id(dims: dict[str, int], dtype: str) -> str:
    dimensions = "_".join(f"{name}{int(value)}" for name, value in sorted(dims.items()))
    return f"{dimensions}_{dtype}"


def examples_for_suite(
    op_name: str,
    suite: str,
    *,
    dtypes: tuple[str, ...] = ("bfloat16",),
) -> list[dict[str, Any]]:
    op = get_op(op_name)
    return [
        ShapeExample(
            id=_shape_id(workload.dims, workload.dtype),
            dims=dict(workload.dims),
            dtype=workload.dtype,
        ).to_dict()
        for workload in op.benchmark_workloads(suite=suite, dtypes=dtypes)
    ]


class ShapeBatchEvaluator:
    """Callable evaluator and batch_evaluator for GEPA optimize_anything."""

    def __init__(
        self,
        *,
        seed_path: str | Path,
        cache_dir: str | Path,
        op_name: str = "layernorm",
        baseline: str = "liger",
        warmup: int = 3,
        reps: int = 10,
        timeout: int = 850,
        python: str | Path | None = None,
    ) -> None:
        self.seed_path = Path(seed_path).resolve()
        self.template = EvolveBlockTemplate.from_path(self.seed_path)
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.op_name = op_name
        self.baseline = baseline
        self.warmup = int(warmup)
        self.reps = int(reps)
        self.timeout = int(timeout)
        self.python = str(python or sys.executable)

    @property
    def seed_body(self) -> str:
        return self.template.seed_body

    @staticmethod
    def _body(candidate: str | dict[str, str]) -> str:
        if isinstance(candidate, str):
            return candidate
        if "current_candidate" in candidate:
            return candidate["current_candidate"]
        if "kernel_code" in candidate:
            return candidate["kernel_code"]
        if len(candidate) == 1:
            return next(iter(candidate.values()))
        raise ValueError(f"cannot identify code component in candidate keys {sorted(candidate)}")

    def _candidate_dir(self, body: str) -> Path:
        return self.cache_dir / self.template.digest(body)

    def _shape_cache_path(self, directory: Path, shape_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in shape_id)
        return directory / "shapes" / f"{safe}_w{self.warmup}_r{self.reps}.json"

    @staticmethod
    def _zero(shape: dict[str, Any], status: str, error: str) -> tuple[float, dict]:
        return 0.0, {
            "shape_id": shape["id"],
            "dims": shape["dims"],
            "dtype": shape["dtype"],
            "status": status,
            "error": error,
        }

    def _run_worker(
        self,
        *,
        source: str,
        shapes: list[dict[str, Any]],
        run_gate: bool,
    ) -> dict[str, Any]:
        request = {
            "source": source,
            "shapes": shapes,
            "run_gate": run_gate,
            "op": self.op_name,
            "baseline": self.baseline,
            "warmup": self.warmup,
            "reps": self.reps,
        }
        with tempfile.TemporaryDirectory(prefix="evograd_gepa_request_") as temporary:
            request_path = Path(temporary) / "request.json"
            result_path = Path(temporary) / "result.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.setdefault(
                "EVOGRAD_BASELINE_TIMING_CACHE_PATH",
                str(self.cache_dir / "baseline_timing_cache.json"),
            )
            try:
                completed = subprocess.run(
                    [
                        self.python,
                        "-m",
                        "evograd.gepa_backend.worker",
                        str(request_path),
                        str(result_path),
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return {
                    "gate_ok": False,
                    "shapes": {
                        shape["id"]: {
                            "score": 0.0,
                            "info": {
                                "shape_id": shape["id"],
                                "status": "timeout",
                                "error": f"candidate evaluation exceeded {self.timeout}s",
                            },
                        }
                        for shape in shapes
                    },
                }
            if completed.returncode != 0 or not result_path.is_file():
                detail = (completed.stderr or completed.stdout)[-8000:]
                return {
                    "gate_ok": False,
                    "shapes": {
                        shape["id"]: {
                            "score": 0.0,
                            "info": {
                                "shape_id": shape["id"],
                                "status": "worker_failed",
                                "error": detail,
                            },
                        }
                        for shape in shapes
                    },
                }
            return json.loads(result_path.read_text(encoding="utf-8"))

    def _evaluate_group(
        self,
        candidate: str | dict[str, str],
        shapes: list[dict[str, Any]],
    ) -> dict[str, tuple[float, dict]]:
        try:
            body = self._body(candidate)
            source = self.template.render(body)
        except Exception as exc:
            return {
                shape["id"]: self._zero(shape, "candidate_contract_error", str(exc))
                for shape in shapes
            }

        directory = self._candidate_dir(body)
        shape_dir = directory / "shapes"
        shape_dir.mkdir(parents=True, exist_ok=True)
        (directory / "block.py").write_text(body.strip() + "\n", encoding="utf-8")
        (directory / "program.py").write_text(source, encoding="utf-8")
        gate_path = directory / "gate.json"

        cached: dict[str, tuple[float, dict]] = {}
        missing = []
        for shape in shapes:
            path = self._shape_cache_path(directory, shape["id"])
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                cached[shape["id"]] = (float(record["score"]), record["info"])
            else:
                missing.append(shape)
        if not missing:
            return cached

        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None
        if gate is not None and not gate.get("ok", False):
            for shape in missing:
                cached[shape["id"]] = self._zero(
                    shape, gate.get("status", "gate_failed"), gate.get("error", "gate failed")
                )
            return cached

        result = self._run_worker(source=source, shapes=missing, run_gate=gate is None)
        if gate is None:
            sample_info = next(iter(result.get("shapes", {}).values()), {}).get("info", {})
            gate = {
                "ok": bool(result.get("gate_ok")),
                "status": sample_info.get("status", "ok"),
                "error": sample_info.get("error", ""),
            }
            gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

        for shape in missing:
            record = result["shapes"].get(shape["id"])
            if record is None:
                score, info = self._zero(shape, "missing_result", "worker omitted shape")
            else:
                score, info = float(record["score"]), record["info"]
            payload = {"score": score, "info": info}
            self._shape_cache_path(directory, shape["id"]).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            cached[shape["id"]] = (score, info)
        return cached

    def evaluate(
        self,
        candidate: str | dict[str, str],
        example: dict[str, Any],
    ) -> tuple[float, dict]:
        return self._evaluate_group(candidate, [example])[example["id"]]

    def evaluate_batch(
        self,
        pairs: Iterable[tuple[str | dict[str, str], dict[str, Any]]],
        opt_states=None,
    ) -> list[tuple[float, dict]]:
        del opt_states
        pair_list = list(pairs)
        grouped: dict[str, dict[str, Any]] = {}
        for index, (candidate, example) in enumerate(pair_list):
            body = self._body(candidate)
            digest = self.template.digest(body)
            group = grouped.setdefault(digest, {"candidate": candidate, "items": []})
            group["items"].append((index, example))

        outputs: list[tuple[float, dict] | None] = [None] * len(pair_list)
        for group in grouped.values():
            unique = {example["id"]: example for _, example in group["items"]}
            evaluated = self._evaluate_group(group["candidate"], list(unique.values()))
            for index, example in group["items"]:
                outputs[index] = evaluated[example["id"]]
        return [output if output is not None else (0.0, {"status": "missing"}) for output in outputs]
