"""Pipeline D: Inductor-generated autograd-pair seed (LLM-free).

Per dtype:

1. Trace the declared forward with AOTAutograd, partition the joint graph with
   ``min_cut_rematerialization_partition``, and lower both halves with Inductor.
2. Merge the two generated modules into one evolvable seed
   (``initial_program_autograd_pair.py``) holding one forward and one backward
   kernel set.
3. Verify it against ``torch.autograd.grad`` on that dtype's workloads.
4. Write the raw captures and the partitioner's save-set alongside, as grounding
   context and as a reference point for the evolved result.

Seeds are dtype specialists because Inductor specializes on dtype; shapes are
generic. With one dtype the seed is written directly into ``--output-dir``, and
with several into a subdirectory per dtype plus a ``seeds.json`` index.

Unlike Pipeline B, the save-set is not a policy of this pipeline -- it is
whatever the partitioner chose. Pipeline B's inputs-only contract is the same
decision taken at ``activation_memory_budget = 0``.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

from evograd.opdecl.activity import OpDecl, Workload
from evograd.pipelines.shared.runner import report_passed, verify_candidate

_DTYPE_ALIASES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


@dataclass(frozen=True)
class InductorSeedConfig:
    op: OpDecl
    forward: str
    output_dir: Path
    dtypes: tuple[str, ...]
    python: str
    device: str = "cuda"
    dynamic_shapes: bool = True
    autotune: bool = True
    eval_timeout: int = 120
    skip_verify: bool = False


def _dims_are_distinct(workload: Workload) -> bool:
    values = list(workload.dims.values())
    return len(set(values)) == len(values) and all(v > 1 for v in values)


def _with_distinct_dims(workload: Workload) -> Workload:
    """Perturb duplicate dims apart, minimally.

    Symbolic tracing unifies dimensions that happen to be equal at trace time,
    so a capture on M == N produces kernels that assert M == N forever. Sizes
    are nudged by +1 rather than to round numbers: the goal is only to break the
    coincidence, and staying near the declared size keeps the divisibility
    characteristics Inductor specializes on roughly intact.
    """
    seen: set[int] = set()
    dims = {}
    for name, value in workload.dims.items():
        candidate = max(int(value), 2)
        while candidate in seen:
            candidate += 1
        seen.add(candidate)
        dims[name] = candidate
    return replace(workload, dims=dims)


def select_capture_workloads(
    op: OpDecl, dtypes: tuple[str, ...]
) -> dict[str, Workload]:
    """Pick one workload per dtype to trace on.

    Dynamic-shape capture makes the kernels size-generic, and the cut is decided
    by graph structure and the ban heuristics far more than by tensor sizes, so
    any reasonable case per dtype will do -- with one constraint: dimensions that
    are equal at trace time get unified into a single symbol, which bakes that
    equality into the kernel. Prefer a declared workload whose dims are pairwise
    distinct; perturb one if none is.
    """
    cases = op.correctness or op.benchmark
    if not cases:
        raise ValueError(f"{op.name}: declaration has no workloads to capture on")

    selected: dict[str, Workload] = {}
    for dtype in dtypes:
        matching = [c for c in cases if c.dtype == dtype]
        if not matching:
            raise ValueError(
                f"{op.name}: no declared workload with dtype {dtype!r}; "
                f"available: {sorted({c.dtype for c in cases})}"
            )
        distinct = [c for c in matching if _dims_are_distinct(c)]
        if distinct:
            selected[dtype] = distinct[len(distinct) // 2]
        else:
            selected[dtype] = _with_distinct_dims(matching[len(matching) // 2])
    return selected


def _emit_one(
    config: InductorSeedConfig, dtype: str, workload: Workload, seed_dir: Path
) -> tuple[int, dict]:
    """Capture, assemble, and verify one dtype specialist."""
    from evograd.pipelines.d_inductor.capture import capture_inductor_pair
    from evograd.pipelines.d_inductor.seed_codegen import generate_inductor_seed

    op = config.op
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"dtype": dtype, "dims": dict(workload.dims), "dir": str(seed_dir)}

    print(f"[D] {dtype}: capture on {dict(workload.dims)} ({config.device})")
    try:
        captured = capture_inductor_pair(
            op, workload, device=config.device, dynamic=config.dynamic_shapes
        )
    except Exception as exc:
        print(f"  ERROR: Inductor capture failed: {exc}")
        (seed_dir / "capture_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 1, {**summary, "passed": False, "error": "capture"}

    raw_dir = seed_dir / "inductor_raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "forward.py").write_text(captured.forward_source, encoding="utf-8")
    (raw_dir / "backward.py").write_text(captured.backward_source, encoding="utf-8")

    save_set = {
        "op": op.name,
        "dtype": dtype,
        "device": config.device,
        "dynamic_shapes": config.dynamic_shapes,
        "capture_workload": {"dims": dict(workload.dims), "dtype": dtype},
        "saved_bytes_at_capture": captured.saved_bytes_at_capture,
        "saved": [
            {
                "name": s.name,
                "producer": s.producer,
                "shape": list(s.shape),
                "dtype": s.dtype,
                "is_forward_input": s.is_input,
            }
            for s in captured.saved
        ],
    }
    (seed_dir / "partitioner_save_set.json").write_text(
        json.dumps(save_set, indent=2), encoding="utf-8"
    )
    print(
        f"  save-set: {len(captured.saved)} entries, "
        f"{captured.saved_bytes_at_capture:,} bytes at capture shape"
    )
    for entry in captured.saved:
        print(f"    {entry.describe()}")

    try:
        seed = generate_inductor_seed(op, dtype, captured, autotune=config.autotune)
    except Exception as exc:
        print(f"  ERROR: seed assembly failed: {exc}")
        (seed_dir / "generation_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 1, {**summary, "passed": False, "error": "assembly"}

    pair_path = seed_dir / "initial_program_autograd_pair.py"
    pair_path.write_text(seed, encoding="utf-8")
    summary.update(
        seed=str(pair_path),
        chars=len(seed),
        saved_bytes_at_capture=captured.saved_bytes_at_capture,
    )
    print(f"  → {pair_path} ({len(seed):,} chars)")

    if config.skip_verify:
        print("  verification skipped (--skip-verify)")
        return 0, {**summary, "passed": None}

    result = verify_candidate(
        python=config.python,
        op_name=op.name,
        program_path=pair_path,
        log_dir=seed_dir,
        dtypes=(dtype,),
        forward=config.forward,
        declaration=op.declaration,
        timeout=config.eval_timeout,
        device=config.device,
    )
    passed = report_passed(result)
    cases = result.get("verify", {}).get("cases", [])
    passed_cases = sum(int(case.get("ok", False)) for case in cases)
    result.update(
        {
            "passed": passed,
            "passed_cases": passed_cases,
            "failed_cases": len(cases) - passed_cases,
            "total_cases": len(cases),
            "dtypes": [dtype],
            "save_set": save_set,
        }
    )
    (seed_dir / "verification_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"  {'PASS' if passed else 'FAIL'} ({passed_cases}/{len(cases)} cases)")
    return (
        0 if passed else 1,
        {**summary, "passed": passed, "cases": f"{passed_cases}/{len(cases)}"},
    )


def synthesize_inductor_seed(config: InductorSeedConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    op = config.op
    dtypes = tuple(dict.fromkeys(_DTYPE_ALIASES.get(d, d) for d in config.dtypes))

    try:
        workloads = select_capture_workloads(op, dtypes)
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        return 1

    single = len(dtypes) == 1
    print(f"[D] {op.name}: {config.forward}")
    print(f"    {len(dtypes)} dtype specialist(s): {', '.join(dtypes)}")
    if not config.autotune:
        print("    launch configs pinned (--no-autotune)")

    failures = 0
    summaries = []
    for dtype in dtypes:
        seed_dir = config.output_dir if single else config.output_dir / dtype
        rc, summary = _emit_one(config, dtype, workloads[dtype], seed_dir)
        failures += rc
        summaries.append(summary)

    if not single:
        index = config.output_dir / "seeds.json"
        index.write_text(
            json.dumps({"op": op.name, "seeds": summaries}, indent=2), encoding="utf-8"
        )
        print(f"\n[D] index: {index}")
        print("    each specialist evolves separately:")
        for summary in summaries:
            if summary.get("seed"):
                print(
                    f"      evograd evolve --op {op.name} --seed {summary['seed']} "
                    f"--dtype {summary['dtype']} --output-dir <dir>"
                )

    print(f"\n[D] {len(dtypes) - failures}/{len(dtypes)} dtype seeds ok")
    return 0 if failures == 0 else 1
