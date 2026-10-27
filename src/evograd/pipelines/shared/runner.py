"""Shared subprocess plumbing for the seed pipelines.

Replaces the old repo's per-pipeline copies of graph extraction, candidate
verification, and code-fence stripping. Subprocesses import ``evograd`` via
an explicit PYTHONPATH (derived from the installed package location), so
pipelines work identically from a source checkout or a pip install — no
``cwd=repo_root`` assumptions.

Verification is always declaration-driven: candidates are checked by
``evograd.opdecl.verify_cli`` against the op's autograd oracle. This replaces
the old ``--evaluator path/to/evaluator_autograd_pair.py`` per-bench wiring.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import evograd


def evograd_env() -> dict[str, str]:
    """Environment for subprocesses that must be able to ``import evograd``."""
    env = dict(os.environ)
    package_root = str(Path(evograd.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    if package_root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = package_root + (os.pathsep + existing if existing else "")
    return env


def _run(
    cmd: list[str],
    log_dir: Path | None,
    log_stem: str,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    env = evograd_env()
    if extra_env:
        env.update(extra_env)
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        completed = subprocess.CompletedProcess(
            cmd,
            -9,
            stdout=stdout,
            stderr=(
                f"[eval_timeout] candidate killed after {timeout}s; likely a "
                "hanging/deadlocked kernel. Last progress:\n"
                + stderr[-2000:]
            ),
        )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{log_stem}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{log_stem}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return completed


def extract_graph(
    *,
    python: str,
    forward: str,
    example_input: str,
    out_path: Path,
    dynamic_shapes: bool = True,
    log_dir: Path | None = None,
    scalars: dict | None = None,
    math_sdpa: bool = False,
) -> Path:
    """Extract the AtenIR backward graph for a forward reference (GPU node).

    ``scalars`` and ``math_sdpa`` are passed to the extractor as ``--scalars`` /
    ``--math-sdpa``; left at their defaults the command is unchanged.
    """
    cmd = [
        python,
        "-m",
        "evograd.atenir.extract",
        "--fn",
        forward,
        "--example-input",
        example_input,
        "--out",
        str(out_path),
    ]
    if dynamic_shapes:
        cmd.append("--dynamic")
    if scalars:
        cmd.extend(["--scalars", json.dumps(scalars)])
    if math_sdpa:
        cmd.append("--math-sdpa")
    completed = _run(cmd, log_dir, "extract")
    if completed.returncode != 0:
        raise RuntimeError(f"AtenIR extraction failed:\n{completed.stderr}")
    return out_path


def verify_candidate(
    *,
    python: str,
    op_name: str,
    program_path: Path,
    log_dir: Path | None = None,
    dtypes: tuple[str, ...] | None = None,
    forward: str | None = None,
    declaration: str | None = None,
    timeout: int = 120,
    device: str = "cuda",
    seed_acceptance: bool = False,
    benchmark_suite: str | None = None,
) -> dict:
    """Verify a candidate against the op's autograd oracle in a subprocess."""
    # Validated before the command is built, so a misuse cannot be half-applied.
    if seed_acceptance and dtypes:
        raise ValueError("seed acceptance checks every declared correctness dtype")
    if benchmark_suite is not None and not seed_acceptance:
        raise ValueError("benchmark_suite requires seed_acceptance")
    cmd = [
        python,
        "-m",
        "evograd.evaluation.common.seed" if seed_acceptance else "evograd.opdecl.verify_cli",
        "--op",
        op_name,
        "--device",
        device,
    ]
    if forward:
        cmd.extend(["--forward", forward])
    if declaration:
        cmd.extend(["--declaration", declaration])
    for dtype in dtypes or ():
        cmd.extend(["--dtype", dtype])
    if benchmark_suite is not None:
        cmd.extend(["--benchmark-suite", benchmark_suite])
    cmd.append(str(program_path))
    completed = _run(
        cmd,
        log_dir,
        "verify",
        {"EVOGRAD_DECLARATION": declaration} if declaration else None,
        timeout,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {
            "metrics": {"correct": 0.0},
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    if completed.stderr:
        report["stderr"] = completed.stderr
    if completed.returncode != 0:
        # A valid-looking JSON payload followed by a crash/timeout is not an
        # accepted seed. Keep the diagnostic for the next repair prompt.
        report["metrics"] = {**report.get("metrics", {}), "correct": 0.0}
        report["returncode"] = completed.returncode
    return report


def report_passed(report: dict) -> bool:
    return float(report.get("metrics", {}).get("correct", 0.0)) == 1.0


def strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"
