"""Drive an OpenEvolve run for one operator + seed + scoring policy.

Wraps the ``openevolve-run`` CLI from the pip ``openevolve`` package:

* renders the config template with the op's contract,
* points OpenEvolve at the shared ``evaluator_entry.py`` with ``EVOGRAD_OP`` /
  ``EVOGRAD_SCORING`` in the environment,
* copies the final best program out afterwards (re-implements the old fork's
  ``--save-best-to`` patch, which upstream openevolve does not have).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from evograd.opdecl.activity import OpDecl
from evograd.pipelines.shared.runner import evograd_env

_TEMPLATE = Path(__file__).with_name("config_template.yaml")
_EVALUATOR_ENTRY = Path(__file__).with_name("evaluator_entry.py")


def render_config(
    op: OpDecl,
    *,
    iterations: int = 10,
    primary_model: str = "gpt-4o-mini",
    secondary_model: str = "gpt-4o",
    api_base: str = "https://api.openai.com/v1",
) -> str:
    text = _TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__MAX_ITERATIONS__": str(iterations),
        "__PRIMARY_MODEL__": primary_model,
        "__SECONDARY_MODEL__": secondary_model,
        "__API_BASE__": api_base,
        "__FORWARD_FN__": op.forward_fn_name,
        "__FORWARD_ARGS__": op.forward_parameters(),
        "__BACKWARD_FN__": op.backward_fn_name,
        "__BACKWARD_ARGS__": op.backward_parameters(),
        "__BACKWARD_RETURNS__": op.backward_returns(),
        "__FORWARD_SEMANTICS__": op.forward_semantics.replace("\n", " "),
        "__BACKWARD_SEMANTICS__": op.backward_semantics.replace("\n", " "),
        "__EXTRA_CONSTRAINTS__": (op.extra_constraints or "none").replace("\n", " "),
        "__ACTIVITY_CONTRACT__": ", ".join(
            f"{arg.name}: {type(arg).__name__}" for arg in op.args
        ),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _openevolve_cmd() -> list[str]:
    exe = shutil.which("openevolve-run")
    if exe:
        return [exe]
    return [sys.executable, "-m", "openevolve.cli"]


def run_evolve(
    op: OpDecl,
    *,
    seed_path: Path,
    output_dir: Path,
    scoring: str = "speed_memory",
    iterations: int = 10,
    config_path: Path | None = None,
    save_best_to: Path | None = None,
    primary_model: str = "gpt-4o-mini",
    secondary_model: str = "gpt-4o",
    api_base: str = "https://api.openai.com/v1",
    benchmark_suite: str | None = None,
    benchmark_dtypes: tuple[str, ...] | None = None,
    performance_baseline: str = "pytorch_autograd",
    extra_env: dict[str, str] | None = None,
) -> int:
    from evograd.evolve.scoring import get_policy

    get_policy(scoring)
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not seed_path.is_file():
        raise FileNotFoundError(f"seed program not found: {seed_path}")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"OpenEvolve config not found: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_path is None:
        config_path = output_dir / "openevolve_config.yaml"
        config_path.write_text(
            render_config(
                op,
                iterations=iterations,
                primary_model=primary_model,
                secondary_model=secondary_model,
                api_base=api_base,
            ),
            encoding="utf-8",
        )

    env = evograd_env()
    env["EVOGRAD_OP"] = op.name
    env["EVOGRAD_SCORING"] = scoring
    if benchmark_suite:
        # Validate before paying the startup cost of an OpenEvolve subprocess.
        op.benchmark_workloads(suite=benchmark_suite, dtypes=benchmark_dtypes)
        env["EVOGRAD_BENCHMARK_SUITE"] = benchmark_suite
    elif benchmark_dtypes:
        op.benchmark_workloads(dtypes=benchmark_dtypes)
    if benchmark_dtypes:
        env["EVOGRAD_BENCHMARK_DTYPES"] = ",".join(benchmark_dtypes)
    if performance_baseline != "pytorch_autograd":
        if performance_baseline not in op.performance_baselines:
            available = ["pytorch_autograd", *sorted(op.performance_baselines)]
            raise KeyError(
                f"{op.name}: unknown performance baseline {performance_baseline!r}; "
                f"available: {available}"
            )
        env["EVOGRAD_PERFORMANCE_BASELINE"] = performance_baseline
    if extra_env:
        env.update(extra_env)

    cmd = _openevolve_cmd() + [
        str(seed_path),
        str(_EVALUATOR_ENTRY),
        "--config",
        str(config_path),
        "--iterations",
        str(iterations),
        "--output",
        str(output_dir),
    ]
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode

    # Re-implementation of the old fork's --save-best-to convenience.
    best = _find_best_program(output_dir)
    if best is None:
        print(f"ERROR: OpenEvolve succeeded but no best program was found under {output_dir}")
        return 1
    target = save_best_to or (output_dir / "evolved_best_program.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() != target.resolve():
        shutil.copyfile(best, target)
    print(f"Copied final best program to: {target}")
    return 0


def _find_best_program(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("**/best/best_program*.py"), key=os.path.getmtime)
    return candidates[-1] if candidates else None
