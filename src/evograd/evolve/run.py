"""Drive an OpenEvolve run for one operator + seed + scoring policy.

* renders the config template with the op's contract,
* calls OpenEvolve's supported ``run_evolution`` library API,
* points it at the shared evaluator with declaration selection in the environment,
* writes the returned best code directly to the requested destination.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from evograd.opdecl.activity import OpDecl
from evograd.evolve.scoring import (
    DEFAULT_FEATURE_DIMENSIONS,
    validate_feature_dimensions,
)
from evograd.pipelines.shared.runner import evograd_env

_TEMPLATE = Path(__file__).with_name("config_template.yaml")
_MAP_SHAPE_TEMPLATE = Path(__file__).with_name("config_map_shape_template.yaml")
_EVALUATOR_ENTRY = Path(__file__).with_name("evaluator_entry.py")


def render_config(
    op: OpDecl,
    *,
    iterations: int = 10,
    primary_model: str = "gpt-5.6-sol",
    secondary_model: str = "gpt-5.6-sol",
    api_base: str = "https://api.openai.com/v1",
    template: Path | None = None,
    feature_dimensions: tuple[str, ...] = DEFAULT_FEATURE_DIMENSIONS,
    feature_bins: int = 10,
    num_islands: int = 2,
    archive_size: int = 20,
) -> str:
    feature_dimensions = validate_feature_dimensions(tuple(feature_dimensions))
    if feature_bins < 1:
        raise ValueError(f"feature_bins must be >= 1, got {feature_bins}")
    if num_islands < 1:
        raise ValueError(f"num_islands must be >= 1, got {num_islands}")
    if archive_size < 1:
        raise ValueError(f"archive_size must be >= 1, got {archive_size}")
    # OpenEvolve clamps integer feature_bins up to ceil(archive_size^(1/dims));
    # an archive larger than the grid silently undoes a coarse-bin request.
    min_cells = feature_bins ** len(feature_dimensions)
    if archive_size > min_cells:
        raise ValueError(
            f"archive_size {archive_size} exceeds the {min_cells}-cell grid "
            f"({feature_bins} bins ^ {len(feature_dimensions)} dims); OpenEvolve "
            "would raise the bin count and remove cell contention"
        )
    text = (template or _TEMPLATE).read_text(encoding="utf-8")
    replacements = {
        "__MAX_ITERATIONS__": str(iterations),
        "__PRIMARY_MODEL__": primary_model,
        "__SECONDARY_MODEL__": secondary_model,
        "__API_BASE__": api_base,
        "__FEATURE_DIMENSIONS__": json.dumps(list(feature_dimensions)),
        "__FEATURE_BINS__": str(feature_bins),
        "__NUM_ISLANDS__": str(num_islands),
        "__ARCHIVE_SIZE__": str(archive_size),
        "__FORWARD_FN__": op.forward_fn_name,
        "__FORWARD_ARGS__": op.forward_parameters(),
        "__FORWARD_RETURNS__": op.forward_returns(),
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


def render_map_shape_config(
    op: OpDecl,
    *,
    iterations: int = 30,
    primary_model: str = "gpt-5.6-sol",
    secondary_model: str = "gpt-5.6-sol",
    api_base: str = "https://api.openai.com/v1",
) -> str:
    """Config with small/large regime speedups as MAP-Elites feature axes."""
    return render_config(
        op,
        iterations=iterations,
        primary_model=primary_model,
        secondary_model=secondary_model,
        api_base=api_base,
        template=_MAP_SHAPE_TEMPLATE,
    )


@contextmanager
def _temporary_environ(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_evolve(
    op: OpDecl,
    *,
    seed_path: Path,
    output_dir: Path,
    scoring: str = "speed_memory",
    iterations: int = 10,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    save_best_to: Path | None = None,
    primary_model: str = "gpt-5.6-sol",
    secondary_model: str = "gpt-5.6-sol",
    api_base: str = "https://api.openai.com/v1",
    benchmark_suite: str | None = None,
    benchmark_dtypes: tuple[str, ...] | None = None,
    dtype: str | None = None,
    performance_baseline: str = "auto",
    save_programs: bool = False,
    feature_dimensions: tuple[str, ...] | None = None,
    feature_bins: int = 10,
    num_islands: int = 2,
    archive_size: int = 20,
    extra_env: dict[str, str] | None = None,
    ncu: bool = False,
    ncu_model: str | None = None,
    ncu_api_key: str | None = None,
    ncu_timeout: int = 120,
    ncu_optimizer_timeout: int = 360,
    ncu_skip_at_roofline_pct: float = 95.0,
) -> int:
    from evograd.opdecl.baselines import resolve_performance_baseline
    from evograd.evolve.scoring import get_policy

    get_policy(scoring)
    feature_dimensions = (
        tuple(feature_dimensions) if feature_dimensions else DEFAULT_FEATURE_DIMENSIONS
    )
    validate_feature_dimensions(feature_dimensions)
    performance_baseline = resolve_performance_baseline(op, performance_baseline)
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not seed_path.is_file():
        raise FileNotFoundError(f"seed program not found: {seed_path}")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"OpenEvolve config not found: {config_path}")
    if checkpoint_path is not None and not checkpoint_path.is_dir():
        raise FileNotFoundError(f"OpenEvolve checkpoint not found: {checkpoint_path}")
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
                feature_dimensions=feature_dimensions,
                feature_bins=feature_bins,
                num_islands=num_islands,
                archive_size=archive_size,
            ),
            encoding="utf-8",
        )

    env = evograd_env()
    env["EVOGRAD_OP"] = op.name
    env["EVOGRAD_SCORING"] = scoring
    env["EVOGRAD_FORWARD_OVERRIDE"] = op.forward
    if op.declaration:
        env["EVOGRAD_DECLARATION"] = op.declaration
    env.setdefault(
        "EVOGRAD_BASELINE_TIMING_CACHE_PATH",
        str(output_dir / ".baseline_timing_cache.json"),
    )
    if benchmark_suite:
        # Validate before paying the startup cost of an OpenEvolve subprocess.
        op.benchmark_workloads(suite=benchmark_suite, dtypes=benchmark_dtypes)
        env["EVOGRAD_BENCHMARK_SUITE"] = benchmark_suite
    elif benchmark_dtypes:
        op.benchmark_workloads(dtypes=benchmark_dtypes)
    if benchmark_dtypes:
        env["EVOGRAD_BENCHMARK_DTYPES"] = ",".join(benchmark_dtypes)
    if dtype:
        # Dtype-specialist run (Pipeline D seeds): gate correctness on this
        # dtype alone, and measure on it too unless the caller asked otherwise.
        if not any(w.dtype == dtype for w in op.correctness):
            raise ValueError(
                f"{op.name}: no declared correctness workload with dtype {dtype!r}; "
                f"available: {sorted({w.dtype for w in op.correctness})}"
            )
        env["EVOGRAD_CORRECTNESS_DTYPES"] = dtype
        env.setdefault("EVOGRAD_BENCHMARK_DTYPES", dtype)
    if performance_baseline != "pytorch_autograd":
        env["EVOGRAD_PERFORMANCE_BASELINE"] = performance_baseline
    programs_dir = output_dir / "programs"
    if save_programs:
        # The evaluator sees every candidate OpenEvolve generates; run_evolution
        # only hands back the best one. Archiving there is the only place that
        # sees the whole population.
        env["EVOGRAD_PROGRAM_ARCHIVE_DIR"] = str(programs_dir)
    if extra_env:
        env.update(extra_env)

    from openevolve import run_evolution

    print(
        f"+ OpenEvolve.run_evolution(seed={seed_path}, iterations={iterations}, "
        f"output={output_dir})"
    )
    with _temporary_environ(env):
        result = run_evolution(
            initial_program=seed_path,
            evaluator=_EVALUATOR_ENTRY,
            config=config_path,
            iterations=iterations,
            output_dir=str(output_dir),
            cleanup=False,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        )

    if not result.best_code:
        print("ERROR: OpenEvolve completed without returning a best program")
        return 1
    target = save_best_to or (output_dir / "evolved_best_program.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.best_code, encoding="utf-8")
    if ncu:
        from evograd.ncu.refine import refine_candidate

        record = refine_candidate(
            op,
            target,
            output_dir=output_dir / "ncu_passes" / "final",
            baseline=performance_baseline,
            model=ncu_model or primary_model,
            api_base=api_base,
            api_key=ncu_api_key,
            ncu_timeout=ncu_timeout,
            optimizer_timeout=ncu_optimizer_timeout,
            skip_at_roofline_pct=ncu_skip_at_roofline_pct,
        )
        print(f"NCU refinement: {record['outcome']}")
    print(f"Wrote final best program to: {target}")
    if save_programs:
        archived = len(list(programs_dir.glob("*.py"))) if programs_dir.is_dir() else 0
        print(f"Archived {archived} evaluated program(s) to: {programs_dir}")
    return 0
