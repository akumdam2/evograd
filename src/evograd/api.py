"""High-level file-in, deployed-autograd-pair-out interface."""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Callable

from evograd.dispatch import dispatch
from evograd.opdecl.activity import OpDecl, example_input_spec
from evograd.ops import get_op

_SNAPSHOT_HEADER = "import math\n\nimport torch\nimport torch.nn.functional as F\n"
_SNAPSHOT_GLOBALS = {"math", "torch", "F"}


@dataclass(frozen=True)
class EvogradResult:
    """Artifacts produced by :func:`evograd`."""

    op: str
    output_dir: Path
    seed: Path
    program: Path
    report: Path
    metrics: Path
    baseline: str


@contextmanager
def _environment(values: dict[str, str]):
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


def _materialize_callable(fn: Callable, output_dir: Path) -> str:
    if getattr(fn, "__name__", "<lambda>") == "<lambda>":
        raise ValueError("forward callable must be a named def, not a lambda")
    if fn.__closure__:
        raise ValueError(
            f"forward callable {fn.__name__} captures a closure; make it "
            "self-contained or pass a Python file"
        )
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"cannot read source for {fn.__name__}; put it in a Python file"
        ) from exc
    unsupported = sorted(
        name
        for name in fn.__code__.co_names
        if name in fn.__globals__ and name not in _SNAPSHOT_GLOBALS
    )
    if unsupported:
        raise ValueError(
            f"forward callable {fn.__name__} uses globals the snapshot cannot carry: "
            f"{', '.join(unsupported)}; only math, torch, and F are provided"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / "user_forward.py").resolve()
    path.write_text(
        f'"""Snapshot of callable {fn.__name__} passed to evograd()."""\n'
        f"{_SNAPSHOT_HEADER}\n{source}",
        encoding="utf-8",
    )
    return f"{path}:{fn.__name__}"


def _file_forward_spec(value: str | Path) -> str:
    raw = str(value)
    path_text, separator, function_name = raw.partition(":")
    path = Path(path_text)
    if not path.is_file():
        # Preserve importable ``package.module:function`` references.
        if "/" not in path_text and not path_text.endswith(".py"):
            return raw
        raise FileNotFoundError(f"forward file not found: {path}")
    path = path.resolve()
    if separator:
        return f"{path}:{function_name}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    if len(functions) != 1:
        raise ValueError(
            f"{path} has {len(functions)} public top-level functions "
            f"({', '.join(functions) or 'none'}); pass 'path.py:function'"
        )
    return f"{path}:{functions[0]}"


def _resolve_forward_input(
    forward: str | Path | Callable | None, op: OpDecl | None, output_dir: Path
) -> str:
    if forward is None:
        if op is None:
            raise ValueError("a forward is required for an undeclared operator")
        return op.forward
    if isinstance(forward, (str, Path)):
        return _file_forward_spec(forward)
    if not callable(forward):
        raise TypeError(
            f"forward must be a path, callable, or None; got {type(forward).__name__}"
        )
    return _materialize_callable(forward, output_dir)


def _seed(
    op: OpDecl,
    *,
    pipeline: str,
    output_dir: Path,
    api_base: str,
    model: str,
    api_key: str | None,
    max_attempts: int,
) -> Path:
    if pipeline == "a":
        from evograd.pipelines.a_atenir_llm.synthesize import (
            AutogradPairConfig,
            synthesize_autograd_pair,
        )

        rc = synthesize_autograd_pair(
            AutogradPairConfig(
                op=op,
                forward=op.forward,
                example_input=example_input_spec(op),
                output_dir=output_dir,
                api_base=api_base,
                model=model,
                api_key=api_key,
                max_attempts=max_attempts,
                max_tokens=16000,
                temperature=0.2,
                timeout=240,
                python=os.sys.executable,
            )
        )
        path = output_dir / "best" / "initial_program_autograd_pair.py"
    elif pipeline == "b":
        from evograd.pipelines.b_dispatch.synthesize import (
            HandwrittenDispatchConfig,
            synthesize_handwritten_dispatch,
        )

        dtypes = tuple(dict.fromkeys(case.dtype for case in op.correctness))
        rc = synthesize_handwritten_dispatch(
            HandwrittenDispatchConfig(
                op=op,
                forward=op.forward,
                example_input=example_input_spec(op),
                output_dir=output_dir,
                dtypes=dtypes,
                atol=2e-5,
                rtol=2e-5,
                fp16_atol=8e-2,
                fp16_rtol=8e-2,
                python=os.sys.executable,
            )
        )
        path = output_dir / "initial_program_autograd_pair.py"
    elif pipeline == "c":
        from evograd.pipelines.c_forward_only.synthesize import (
            ForwardOnlyConfig,
            synthesize_forward_only,
        )

        rc = synthesize_forward_only(
            ForwardOnlyConfig(
                op=op,
                forward=op.forward,
                output_dir=output_dir,
                api_base=api_base,
                model=model,
                api_key=api_key,
                max_attempts=max_attempts,
                max_tokens=16000,
                temperature=0.2,
                timeout=240,
                python=os.sys.executable,
            )
        )
        path = output_dir / "best" / "initial_program_autograd_pair.py"
    else:
        raise ValueError("pipeline must be one of: a, b, c")
    if rc != 0 or not path.is_file():
        raise RuntimeError(f"Pipeline {pipeline.upper()} failed for {op.name}")
    return path


def _evolve_group(
    op_name: str,
    forward: str,
    seed: str,
    group_dir: str,
    group: str,
    iterations: int,
    model: str,
    api_base: str,
    baseline: str,
    gpu: int | None,
    ncu: bool,
    declaration: str | None,
) -> tuple[str, str]:
    from evograd.evolve.run import run_evolve

    if declaration:
        from evograd.ops import load_op

        op = load_op(declaration)
    else:
        op = get_op(op_name)
    op = replace(op, forward=forward)
    target = Path(group_dir) / "evolved_best_program.py"
    extra_env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
    suite = group if group != "full" or "full" in op.benchmark_suites else None
    scoring = (
        "speed_memory_min"
        if group == "full"
        else "speed_memory_min_weighted_geomean"
    )
    rc = run_evolve(
        op,
        seed_path=Path(seed),
        output_dir=Path(group_dir),
        scoring=scoring,
        iterations=iterations,
        save_best_to=target,
        primary_model=model,
        secondary_model=model,
        api_base=api_base,
        benchmark_suite=suite,
        performance_baseline=baseline,
        extra_env=extra_env,
        ncu=ncu,
        ncu_model=model,
    )
    if rc != 0:
        raise RuntimeError(f"OpenEvolve failed for {op_name}/{group}")
    return group, str(target)


def evograd(
    forward: str | Path | Callable | None = None,
    *,
    op: str,
    output_dir: str | Path | None = None,
    pipeline: str = "a",
    api_key: str | None = None,
    baseline: str = "auto",
    iterations: int = 10,
    gpus: int = 1,
    model: str = "gpt-5.5",
    api_base: str = "https://api.openai.com/v1",
    max_attempts: int = 5,
    force: bool = False,
    ncu: bool = False,
) -> EvogradResult:
    """Generate, evolve, dispatch, and report for one declared operator.

    A supplied forward replaces a known declaration's reference implementation
    while keeping its argument/activity/workload contract. For an unknown
    operator name, Evograd first synthesizes and validates a declaration beside
    the run artifacts, then uses that declaration for every pipeline stage.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if gpus not in (1, 3):
        raise ValueError("gpus must be 1 or 3")
    root = (
        Path(output_dir)
        if output_dir is not None
        else Path.cwd() / "evograd_output" / op
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        base_op = get_op(op)
    except KeyError:
        base_op = None
    forward_spec = _resolve_forward_input(forward, base_op, root)
    if base_op is None:
        from evograd.scaffold import synthesize_declaration
        from evograd.ops import load_op

        scaffold = synthesize_declaration(
            op,
            forward_spec,
            output_dir=root / "scaffold",
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_attempts=max_attempts,
        )
        declared = load_op(f"{scaffold.declaration}:op")
    else:
        declared = replace(base_op, forward=forward_spec)
    declared.validate()

    seed_dir = root / "seed"
    expected_seed = (
        seed_dir / "initial_program_autograd_pair.py"
        if pipeline == "b"
        else seed_dir / "best" / "initial_program_autograd_pair.py"
    )
    env = {"OPENAI_API_KEY": api_key} if api_key else {}
    if declared.declaration:
        env["EVOGRAD_DECLARATION"] = declared.declaration
    with _environment(env):
        seed_path = (
            expected_seed
            if expected_seed.is_file() and not force
            else _seed(
                declared,
                pipeline=pipeline,
                output_dir=seed_dir,
                api_base=api_base,
                model=model,
                api_key=api_key,
                max_attempts=max_attempts,
            )
        )

        groups = ["full"]
        if (
            declared.regime_feature is not None
            and declared.benchmark_suites.get("small")
            and declared.benchmark_suites.get("large")
        ):
            groups.extend(("small", "large"))

        programs: dict[str, Path] = {}
        pending = []
        for index, group in enumerate(groups):
            group_dir = root / "evolve" / group
            target = group_dir / "evolved_best_program.py"
            if target.is_file() and not force:
                programs[group] = target
            else:
                pending.append((index, group, group_dir))

        if pending and gpus == 3 and len(groups) == 3:
            with ProcessPoolExecutor(
                max_workers=3, mp_context=get_context("spawn")
            ) as pool:
                futures = {
                    pool.submit(
                        _evolve_group,
                        declared.name,
                        declared.forward,
                        str(seed_path),
                        str(group_dir),
                        group,
                        iterations,
                        model,
                        api_base,
                        baseline,
                        index,
                        ncu,
                        declared.declaration,
                    ): group
                    for index, group, group_dir in pending
                }
                for future in as_completed(futures):
                    group, path = future.result()
                    programs[group] = Path(path)
        else:
            for _index, group, group_dir in pending:
                tag, path = _evolve_group(
                    declared.name,
                    declared.forward,
                    str(seed_path),
                    str(group_dir),
                    group,
                    iterations,
                    model,
                    api_base,
                    baseline,
                    None,
                    ncu,
                    declared.declaration,
                )
                programs[tag] = Path(path)

    deploy_dir = root / "deploy"
    metrics_path = deploy_dir / f"{op}_dispatch_report.json"
    if force or not metrics_path.is_file():
        report = dispatch(
            declared,
            programs,
            output_dir=deploy_dir,
            baseline=baseline,
        )
    else:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        report.setdefault("report_path", str(metrics_path))
        reports = sorted(deploy_dir.glob(f"RESULTS_{op}_vs_*.md"))
        report.setdefault("markdown_path", str(reports[0]) if reports else "")

    return EvogradResult(
        op=op,
        output_dir=root,
        seed=seed_path,
        program=Path(report["final_program"]),
        report=Path(report["markdown_path"]),
        metrics=metrics_path,
        baseline=str(report["baseline"]),
    )


__all__ = ["EvogradResult", "evograd"]
