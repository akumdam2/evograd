"""The one evaluator every agent's candidates go through.

Two depths, both in a killable child process:

* **full** -- ``evograd.evolve.evaluator_entry``, the file ``evograd evolve``
  hands OpenEvolve, under the environment ``evograd.evolve.run.run_evolve``
  sets: scoped declaration, forward override, scoring policy, performance
  baseline, baseline timing cache. Static rules, import, correctness on every
  declared output and gradient, a run at the timed shape, then timing and
  ``combined_score``. This is the in-loop score and feedback.
* **correctness** -- the same static rules, import and correctness/coverage
  checks, without timing. For agents whose loop gates on a pass/fail test
  (KernelAgent's test file, GEAK's oracle) and does its own timing.

Command line, for harnesses that run in another process or language::

    python -m evograd.adbench_tools.evaluate --task TASK_DIR [--run-dir RUN] \\
        [--correctness-only] [--origin NAME] [--json OUT] CANDIDATE.py

prints one JSON report and exits 0 only when the candidate is correct.
``CANDIDATE.py`` may be a bare agent module or an already packaged artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evograd.adbench_tools.artifact import END, HEADER_MARK, PackagingError, package, unpackage
from evograd.adbench_tools.gpu_lock import run_evaluation
from evograd.adbench_tools.ledger import RunLedger
from evograd.adbench_tools.task import DECLARED_RULE, TASK_FILE, AgentTask, load

FAILED = -1e9
DEFAULT_TIMEOUT = 900


@dataclass
class EvalRecord:
    """One evaluation's outcome: OpenEvolve-shaped metrics and artifacts."""

    kind: str
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)
    program: str | None = None
    seconds: float = 0.0

    @property
    def correct(self) -> bool:
        return float(self.metrics.get("correct", 0.0)) == 1.0

    @property
    def combined_score(self) -> float:
        return float(self.metrics.get("combined_score", FAILED))

    def agent_score(self) -> float:
        """A non-negative score for optimizers that need one: 0 when incorrect."""
        return max(self.combined_score, 0.0) if self.correct else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "correct": self.correct, "metrics": self.metrics,
                "artifacts": self.artifacts, "program": self.program,
                "seconds": round(self.seconds, 2)}

    def feedback(self, limit: int = 12000) -> str:
        """The evaluator's verdict as text, in the shape OpenEvolve shows its LLM."""
        keep = ("combined_score", "correct", "partial_correctness", "forward_correct",
                "speedup", "full_step_speedup", "backward_speedup", "weighted_speedup",
                "forward_ms", "backward_from_saved_ms", "raw_forward_backward_full_step_ms",
                "baseline_latency_ms", "baseline_full_step_ms", "baseline_raw_full_step_ms",
                "saved_bytes", "input_bytes")
        metrics = {k: v for k, v in self.metrics.items()
                   if k in keep or k.endswith("_correct")}
        text = "metrics:\n" + json.dumps(metrics, indent=2, sort_keys=True)
        for name, value in self.artifacts.items():
            if name == "benchmark" and isinstance(value, dict):
                value = {k: value.get(k) for k in ("aggregate", "cases", "primitives") if k in value}
            rendered = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
            text += f"\n\n{name}:\n{rendered}"
        return text if len(text) <= limit else text[: limit - 40] + "\n... [feedback truncated]"


def evaluator_env(task: AgentTask, *, cache_dir: str | Path | None = None,
                  warmup: int | None = None, reps: int | None = None,
                  contract: str | Path | None = None) -> dict[str, str]:
    """The environment ``run_evolve`` gives the evaluator, for this task."""
    from evograd.pipelines.shared.runner import evograd_env

    op = task.op()
    env = evograd_env()
    env["EVOGRAD_OP"] = op.name
    env["EVOGRAD_SCORING"] = task.spec["scoring"]
    env["EVOGRAD_FORWARD_OVERRIDE"] = op.forward
    env["EVOGRAD_DECLARATION"] = task.declaration
    env["EVOGRAD_ALLOWED_PRIMITIVES"] = ",".join(task.allowed_primitives)
    cache = Path(cache_dir) if cache_dir else task.root / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("EVOGRAD_BASELINE_TIMING_CACHE_PATH",
                   str(cache / "baseline_timing_cache.json"))
    if task.spec["performance_baseline"] != "pytorch_autograd":
        env["EVOGRAD_PERFORMANCE_BASELINE"] = task.spec["performance_baseline"]
    if warmup is not None:
        env["EVOGRAD_BENCHMARK_WARMUP"] = str(int(warmup))
    if reps is not None:
        env["EVOGRAD_BENCHMARK_REPS"] = str(int(reps))
    from evograd.pipelines.unified.contract import CONTRACT_ENV

    # Only evograd's own unified-seed lineage carries an evaluator-owned contract.
    if contract is not None:
        env[CONTRACT_ENV] = str(Path(contract).resolve())
    else:
        env.pop(CONTRACT_ENV, None)
    for name in ("EVOGRAD_PROGRAM_ARCHIVE_DIR", "EVOGRAD_EVAL_CHILD", "EVOGRAD_RESULT_JSON"):
        env.pop(name, None)
    from evograd.verification.payloads.gate1 import SEARCH_CORRECTNESS_ENV, SEARCH_RULE

    # The pack alone decides how correctness is judged, never the caller's environment.
    if search_rule(task) == SEARCH_RULE:
        env[SEARCH_CORRECTNESS_ENV] = str(task.root / TASK_FILE)
    else:
        env.pop(SEARCH_CORRECTNESS_ENV, None)
    return env


def search_rule(task: AgentTask) -> str:
    """The pack's search-feedback correctness rule: Gate 1 on calibration payloads for a
    model case, never the declaration's tolerances; those for an operator pack."""
    from evograd.verification.payloads.gate1 import SEARCH_RULE

    rule = (task.spec.get("search_correctness") or {}).get("rule", DECLARED_RULE)
    if "model" in task.spec and rule != SEARCH_RULE:
        raise ValueError(f"{task.root}: a model case is judged by {SEARCH_RULE!r}, not "
                         f"{rule!r}; rebuild the pack")
    return rule


def _run_child(module: str, program: Path, env: dict[str, str], timeout: int,
               extra: list[str] | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="adbench_tools_eval_") as scratch:
        result_path = Path(scratch) / "result.json"
        env = {**env, "EVOGRAD_RESULT_JSON": str(result_path)}
        try:
            completed = run_evaluation(
                [sys.executable, "-m", module, *(extra or []), str(program)],
                env=env, timeout=timeout, label=f"{module} {program.name}",
            )
        except subprocess.TimeoutExpired as exc:
            tail = exc.stderr if isinstance(exc.stderr, str) else ""
            return {"metrics": {"combined_score": FAILED, "correct": 0.0},
                    "artifacts": {"failure": {"error_type": "Timeout",
                                              "error_message": f"evaluation exceeded {timeout}s",
                                              "stderr_tail": tail[-4000:]}}}
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            payload = {"metrics": {"combined_score": FAILED, "correct": 0.0},
                       "artifacts": {"failure": {
                           "error_type": "EvaluatorCrashed",
                           "error_message": f"exit {completed.returncode}",
                           "stderr_tail": completed.stderr[-4000:],
                           "stdout_tail": completed.stdout[-2000:]}}}
    artifacts = {}
    for name, value in (payload.get("artifacts") or {}).items():
        try:
            artifacts[name] = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            artifacts[name] = value
    return {"metrics": payload.get("metrics") or {}, "artifacts": artifacts}


def prepare_program(task: AgentTask, source: str, directory: Path, *,
                    origin: str, repackage: bool = False) -> Path:
    """Package ``source`` and write it.

    A bare agent module is packaged. An evograd artifact (one with its own
    deployment entry, e.g. evograd's unified seed lineage) is used as it is.
    An adbench-tools artifact is used as it is in the loop, and with
    ``repackage=True`` rebuilt from its block body, so nothing outside the
    block survives from the agent's side.
    """
    from evograd.pipelines.shared.artifact import DEPLOYMENT_ENTRY_ATTR

    if HEADER_MARK in source:
        text = package(task, unpackage(source), origin=origin) if repackage else source
    elif f"\n{DEPLOYMENT_ENTRY_ATTR} = " in source:
        text = source
    else:
        text = package(task, source, origin=origin)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = directory / f"{stamp}_{digest}.py"
    path.write_text(text, encoding="utf-8")
    return path


def fixed_code_intact(task: AgentTask, source: str) -> bool | None:
    """Whether an adbench-tools artifact's code after the block is the harness's.

    ``None`` for anything that is not an adbench-tools artifact.
    """
    if HEADER_MARK not in source or source.count(END) != 1:
        return None
    try:
        fresh = package(task, unpackage(source), origin="check")
    except PackagingError:
        return False
    return source.split(END, 1)[1] == fresh.split(END, 1)[1]


def evaluate_source(
    task: AgentTask,
    source: str,
    *,
    run_dir: str | Path | None = None,
    origin: str = "unknown",
    correctness_only: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    warmup: int | None = None,
    reps: int | None = None,
    contract: str | Path | None = None,
) -> EvalRecord:
    """Package and evaluate one candidate module; log it to ``run_dir``."""
    kind = "correctness" if correctness_only else "full"
    started = time.time()
    ledger = RunLedger(run_dir) if run_dir and (Path(run_dir) / "run.json").is_file() else None
    programs = Path(run_dir) / "programs" if run_dir else task.root / ".cache" / "programs"
    try:
        program = prepare_program(task, source, programs, origin=origin)
    except PackagingError as exc:
        record = EvalRecord(kind, {"combined_score": FAILED, "correct": 0.0},
                            {"failure": {"error_type": "PackagingError",
                                         "error_message": str(exc)}})
    else:
        # The pack may be shared and read-only: a run keeps its own timing cache.
        env = evaluator_env(task, cache_dir=Path(run_dir) / ".cache" if run_dir else None,
                            warmup=warmup, reps=reps, contract=contract)
        if correctness_only:
            payload = _run_child("evograd.adbench_tools._check", program, env, timeout,
                                 ["--declaration", task.declaration, "--op", task.op_name])
        else:
            payload = _run_child("evograd.evolve.evaluator_entry", program, env, timeout)
        record = EvalRecord(kind, payload["metrics"], payload["artifacts"], str(program))
    record.seconds = time.time() - started
    if ledger is not None:
        failure = record.artifacts.get("failure") if isinstance(record.artifacts, dict) else None
        ledger.log_evaluation({
            "kind": kind, "origin": origin, "correct": record.correct,
            "combined_score": record.combined_score, "program": record.program,
            "seconds": round(record.seconds, 2),
            "failure": (failure or {}).get("error_type") if isinstance(failure, dict) else None,
        })
    return record


def evaluate_file(task: AgentTask, path: str | Path, **kwargs) -> EvalRecord:
    return evaluate_source(task, Path(path).read_text(encoding="utf-8"), **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=Path, required=True, help="task pack directory")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="agent run directory; the evaluation is logged to its ledger")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--origin", default="cli")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)

    task = load(args.task)
    record = evaluate_file(task, args.candidate, run_dir=args.run_dir, origin=args.origin,
                           correctness_only=args.correctness_only, timeout=args.timeout)
    report = record.to_dict()
    report["feedback"] = record.feedback()
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if record.correct else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
