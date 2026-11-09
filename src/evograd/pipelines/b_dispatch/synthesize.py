"""Pipeline B: LLM-free hand-written dispatch autograd-pair seed.

1. Extract the AtenIR backward graph (symbolic shapes by default), with the
   declaration's scalars and SDPA through the math backend.
2. Emit a dispatch-free Triton program (``dispatch_program.py``) from the
   hand-written primitive kernel registry — no LLM.
3. Wrap it into an autograd-pair artifact (``candidate.py``) using the
   declaration-native wrapper codegen, and check the static rules.
4. Seed acceptance (Gate 1 on the task pack's search payloads); only a passing
   candidate is published as ``initial_program_autograd_pair.py``.
5. Write ``lowering_context.md`` (grounding context reusable by LLM pipelines).
"""

from __future__ import annotations

import importlib
import inspect
import json
import re as _re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evograd.opdecl.activity import OpDecl
from evograd.pipelines.shared.runner import (
    acceptance_cases,
    extract_graph,
    report_passed,
    verify_candidate,
)

@dataclass(frozen=True)
class HandwrittenDispatchConfig:
    op: OpDecl
    forward: str        # callable spec "pkg.mod:fn"
    example_input: str  # spec string "[(8,64) f32, (64) f32, (64) f32]"
    output_dir: Path
    python: str
    eval_timeout: int = 120
    # Extract the AtenIR graph with symbolic shapes so the generated program is
    # correct at any input shape (no traced dims baked into expand/view/scalars).
    dynamic_shapes: bool = True
    # Trusted Level-1 primitives the run grants, recorded in ARTIFACT_CONTRACT.
    allowed_primitives: tuple[str, ...] = ()
    # Publish the statically checked pair without seed acceptance (no-CUDA machines).
    skip_verify: bool = False
    #: The model case's task pack; Gate 1 on its search payloads accepts the seed.
    pack: Path | None = None


def declared_scalars(op: OpDecl) -> dict[str, Any]:
    """The scalar arguments (e.g. ``eps``) the declaration's case inputs carry.

    The graph is traced with these, as the unified pipeline bakes them: the
    effective value is what ``make_inputs`` supplies, which may differ from both
    the declared default and the forward's own default.
    """
    scalars = op.scalar_inactive_args()
    if not scalars:
        return {}
    from evograd.opdecl.inputs import make_case_inputs

    values = make_case_inputs(op, op.correctness[0], device="cpu")
    return {a.name: values.get(a.name, a.default) for a in scalars}


# ── context generation ────────────────────────────────────────────────────────

_TRITON_SUBMODULES = ("elementwise", "reduction", "gemm", "scatter_gather")


def _get_triton_source(kernel) -> tuple[str, str]:
    """Return (dedup_key, source_text) for a kernel from make_registry.

    Direct Triton module functions  → full function source via inspect.getsource.
    Factory-returned closures       → factory source + referenced helpers (2 levels deep),
                                      surfacing the actual @triton.jit kernel bodies.
    Dispatch.py closures            → closure source (thin wrapper, still informative).
    """
    module_name = getattr(kernel, "__module__", "") or ""
    qualname = getattr(kernel, "__qualname__", "") or ""

    if any(m in module_name for m in _TRITON_SUBMODULES):
        try:
            mod = sys.modules.get(module_name) or importlib.import_module(module_name)
        except ImportError:
            mod = None

        if "<locals>" in qualname and mod is not None:
            factory_name = qualname.split(".<locals>")[0]
            dedup_key = f"{module_name}.{factory_name}"
            factory_fn = getattr(mod, factory_name, None)
            if factory_fn is None:
                return dedup_key, ""
            try:
                factory_src = inspect.getsource(factory_fn)
            except (OSError, TypeError):
                return dedup_key, ""

            parts = [factory_src]
            seen_helpers: set[str] = set()

            def _safe_getsource(fn) -> str | None:
                # Triton JITFunction wraps the original Python fn in .fn or
                # __wrapped__; unwrap first so we get the kernel body.
                for candidate in (fn,
                                   getattr(fn, "__wrapped__", None),
                                   getattr(fn, "fn", None)):
                    if candidate is None:
                        continue
                    try:
                        return inspect.getsource(candidate)
                    except (OSError, TypeError):
                        pass
                return None

            def _collect(text: str, depth: int) -> None:
                if depth <= 0:
                    return
                for name in set(_re.findall(r"\b(_[a-zA-Z0-9_]+)\b", text)):
                    if name in seen_helpers:
                        continue
                    fn = getattr(mod, name, None)
                    if fn is not None and callable(fn):
                        seen_helpers.add(name)
                        fn_src = _safe_getsource(fn)
                        if fn_src:
                            parts.append(fn_src)
                            _collect(fn_src, depth - 1)

            _collect(factory_src, depth=2)
            return dedup_key, "\n\n".join(parts)

        dedup_key = f"{module_name}.{qualname}"
        try:
            return dedup_key, inspect.getsource(kernel)
        except (OSError, TypeError):
            return dedup_key, ""

    dedup_key = f"dispatch.{qualname}"
    try:
        return dedup_key, inspect.getsource(kernel)
    except (OSError, TypeError):
        return dedup_key, ""


def generate_dispatch_context(graph_path: Path) -> str:
    """Build a lowering_context.md from the dispatch registry for each node.

    Safe to call without CUDA: make_kernel creates closures but never executes
    Triton kernels. Consumable by LLM pipelines as grounding context.
    """
    from evograd.atenir.primitive_triton.dispatch import make_registry, _kernel_label
    from evograd.pipelines.shared.graph_summary import summarize_graph

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    registry = make_registry(graph)
    call_nodes = [n for n in graph["nodes"] if n["op"] == "call_function"]

    lines = [
        "# Hand-Written Dispatch Context",
        "",
        "These kernels come from `evograd/atenir/primitive_triton/dispatch.py`, a verified",
        "hand-written dispatch table that passes all autograd correctness tests.",
        "Triton kernels are in the `elementwise`, `reduction`, `gemm`, and",
        "`scatter_gather` modules; everything else falls back to PyTorch native.",
        "",
        "## AtenIR Graph",
        "",
        summarize_graph(graph),
        "## Per-Node Kernel Mapping",
        "",
    ]

    seen_sources: dict[str, str] = {}

    for node in call_nodes:
        name = node["name"]
        target = node["target"]
        kernel = registry.get(name)
        label = _kernel_label(kernel) if kernel else "unregistered"

        lines.append(f"### `{name}` — `{target}`")
        lines.append(f"dispatch: `{label}`")
        lines.append("")

        for entry in node.get("args_ordered") or []:
            if entry["kind"] == "node":
                meta = next(
                    (n for n in (node.get("input_nodes") or []) if n["name"] == entry["name"]),
                    {},
                )
                lines.append(
                    f"- tensor `{entry['name']}`: "
                    f"shape={meta.get('shape')} dtype={meta.get('dtype')}"
                )
            elif entry["kind"] == "sym_node":
                lines.append(f"- symbolic size `{entry['name']}` (runtime int)")
            elif entry["kind"] == "shape_list":
                parts = [
                    f"`{it['name']}`" if it["kind"] in ("node", "sym_node") else repr(it["value"])
                    for it in entry["items"]
                ]
                lines.append(f"- shape list [{', '.join(parts)}]")
            else:
                lines.append(f"- scalar value={entry['value']!r}")
        lines.append(
            f"- output: shape={node.get('output_shape')} dtype={node.get('output_dtype')}"
        )

        if kernel is not None:
            dedup_key, src = _get_triton_source(kernel)
            if src:
                if dedup_key not in seen_sources:
                    seen_sources[dedup_key] = name
                    lines += ["```python", src.strip(), "```"]
                else:
                    lines.append(
                        f"_(same Triton implementation as `{seen_sources[dedup_key]}`"
                        f" — see source above)_"
                    )

        lines.append("")

    return "\n".join(lines) + "\n"


# ── orchestrator ──────────────────────────────────────────────────────────────


def synthesize_handwritten_dispatch(config: HandwrittenDispatchConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Published only once accepted; stale results must not read as this run's.
    published = config.output_dir / "initial_program_autograd_pair.py"
    report_path = config.output_dir / "verification_report.json"
    for stale in (published, report_path):
        stale.unlink(missing_ok=True)
    scalars = declared_scalars(config.op)

    print(f"[B] Extract: {config.forward} {scalars or ''}")
    graph_path = extract_graph(
        python=config.python,
        forward=config.forward,
        example_input=config.example_input,
        out_path=config.output_dir / "atenir_graph.json",
        dynamic_shapes=config.dynamic_shapes,
        log_dir=config.output_dir,
        scalars=scalars,
        # Attention decomposes into bmm/softmax/mask, which B implements, rather
        # than a device-specific fused node it could only call through torch.ops.
        math_sdpa=True,
    )
    print(f"  → {graph_path}")

    print("[B] Generate dispatch context (lowering_context.md)")
    try:
        context = generate_dispatch_context(graph_path)
        context_path = config.output_dir / "lowering_context.md"
        context_path.write_text(context, encoding="utf-8")
        print(f"  → {context_path} ({len(context)} chars)")
    except Exception as exc:
        print(f"  Warning: context generation failed: {exc}")

    print("[B] Generate dispatch program (dispatch_program.py — the optimized seed)")
    try:
        from evograd.pipelines.b_dispatch.program_codegen import (
            generate_autograd_pair_program,
            generate_dispatch_program,
        )

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        program = generate_dispatch_program(graph)
        program_path = config.output_dir / "dispatch_program.py"
        program_path.write_text(program, encoding="utf-8")
        print(f"  → {program_path} ({len(program)} chars)")
        from evograd.opdecl.importing import load_candidate
        from evograd.pipelines.shared.artifact import validate_artifact

        pair_program = generate_autograd_pair_program(
            graph, forward=config.forward, op=config.op, scalars=scalars,
            allowed_primitives=config.allowed_primitives,
        )
        pair_path = config.output_dir / "candidate.py"
        pair_path.write_text(pair_program, encoding="utf-8")
        print(f"  → {pair_path} ({len(pair_program)} chars)")
        # The artifact contract and static rules, under the run's grant.
        validate_artifact(config.op, load_candidate(pair_path), source=pair_program,
                          allowed_primitives=config.allowed_primitives)
        print("  artifact contract and static rules: OK")
    except Exception as exc:
        print(f"  ERROR: dispatch program generation failed: {exc}")
        (config.output_dir / "generation_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return 1

    if config.skip_verify:
        report_path.write_text(json.dumps({"verification": "skipped"}), encoding="utf-8")
        pair_path.replace(published)
        print(f"[B] Verification skipped (--skip-verify); unverified seed: {published}")
        return 0
    print("[B] Seed acceptance: Gate 1 on the task pack's search payloads")
    result = verify_candidate(
        python=config.python,
        pack=config.pack,
        program_path=pair_path,
        log_dir=config.output_dir,
        timeout=config.eval_timeout,
    )
    passed = report_passed(result)
    passed_cases, total_cases = acceptance_cases(result)
    result.update(
        {
            "passed": passed,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "total_cases": total_cases,
        }
    )
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[B] {'PASS' if passed else 'FAIL'}  ({passed_cases}/{total_cases} cases)")
    print(f"    report: {report_path}")
    if passed:
        pair_path.replace(published)
        print(f"    accepted seed: {published}")
    return 0 if passed else 1
