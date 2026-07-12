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
from evograd.opdecl.compat import to_operator_spec
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
    spec = to_operator_spec(op)
    text = _TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__MAX_ITERATIONS__": str(iterations),
        "__PRIMARY_MODEL__": primary_model,
        "__SECONDARY_MODEL__": secondary_model,
        "__API_BASE__": api_base,
        "__FORWARD_FN__": spec.forward_fn_name,
        "__FORWARD_ARGS__": spec.forward_args,
        "__BACKWARD_FN__": spec.backward_fn_name,
        "__BACKWARD_ARGS__": spec.backward_args,
        "__BACKWARD_RETURNS__": spec.backward_returns,
        "__FORWARD_SEMANTICS__": spec.forward_semantics.replace("\n", " "),
        "__BACKWARD_SEMANTICS__": spec.backward_semantics.replace("\n", " "),
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
    extra_env: dict[str, str] | None = None,
) -> int:
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
        print(f"Warning: no best program found under {output_dir}")
        return 0
    target = save_best_to or (output_dir / "evolved_best_program.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, target)
    print(f"Copied final best program to: {target}")
    return 0


def _find_best_program(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("**/best/best_program*.py"), key=os.path.getmtime)
    return candidates[-1] if candidates else None
