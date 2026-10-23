"""The task pack: what every agent is given for one operator configuration.

``build_task`` writes a directory that is self-contained and portable (every
path inside it is relative to it):

    task.json        machine-readable task: op, configuration, ABI, rules
    declaration.py   evograd's scoped declaration -- the unified frontend's own
    reference.py     the PyTorch forward reference with its dependencies
    TASK.md          the shared problem statement (see ``statement``)
    skeleton.py      the deliverable's signatures, bodies unimplemented

The configuration is selected by ``evograd.pipelines.unified.synthesize``'s
``select_workload`` and scoped by its ``scoped_op``; ``declaration.py`` is
written by the same function that writes the frontend's. So the correctness
cases, tolerances and timed case are exactly the ones ``evograd seed unified``
followed by ``evograd evolve`` is measured on.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evograd.adbench_tools import TASK_SCHEMA

TASK_FILE = "task.json"
DECLARATION_FILE = "declaration.py"
REFERENCE_FILE = "reference.py"
STATEMENT_FILE = "TASK.md"
SKELETON_FILE = "skeleton.py"

#: Library primitives a task may grant (``evograd.pipelines.shared.primitives``).
LIBRARY_POLICIES = ("auto", "none", "vendor_gemm")
#: Which declared configuration a task targets when no dims are given.
CASE_SELECTIONS = ("benchmark", "correctness")

#: Calls in a forward reference that the unified frontend leaves to a vendor
#: library (Inductor's extern mm/bmm/addmm/convolution). ``auto`` grants the
#: trusted GEMM primitive when the reference contains one of them, so an agent
#: writing from scratch has the library access evograd's seed preserves.
_CONTRACTIONS = frozenset({
    "linear", "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "conv1d", "conv2d",
    "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "tensordot",
    "scaled_dot_product_attention",
})


@dataclass(frozen=True)
class AgentTask:
    """One loaded task pack. ``root`` is the pack directory."""

    root: Path
    spec: dict[str, Any] = field(repr=False)

    @property
    def task_id(self) -> str:
        return self.spec["task_id"]

    @property
    def op_name(self) -> str:
        return self.spec["op"]

    @property
    def declaration(self) -> str:
        """``path.py:op`` of the scoped declaration, resolved against ``root``."""
        return f"{(self.root / DECLARATION_FILE).resolve()}:op"

    @property
    def allowed_primitives(self) -> tuple[str, ...]:
        return tuple(self.spec["allowed_primitives"])

    @property
    def workload(self) -> dict[str, Any]:
        return self.spec["workload"]

    @property
    def statement(self) -> str:
        return (self.root / STATEMENT_FILE).read_text(encoding="utf-8")

    @property
    def reference_source(self) -> str:
        return (self.root / REFERENCE_FILE).read_text(encoding="utf-8")

    def op(self):
        """The scoped ``OpDecl`` -- the declaration every evaluation uses."""
        from evograd.benchmark import load_task

        op = load_task(self.declaration)
        if op.name != self.op_name:
            raise ValueError(f"{self.root}: declaration is {op.name!r}, task is {self.op_name!r}")
        return op


def load(root: str | Path) -> AgentTask:
    root = Path(root).resolve()
    spec = json.loads((root / TASK_FILE).read_text(encoding="utf-8"))
    if spec.get("schema") != TASK_SCHEMA:
        raise ValueError(f"{root}: not an adbench-tools task pack ({spec.get('schema')!r})")
    return AgentTask(root=root, spec=spec)


# ── the forward reference, as a standalone module ────────────────────────────


def _imports_for(name: str, value: Any) -> str | None:
    """An import line binding ``name`` to ``value``, when one exists."""
    if isinstance(value, types.ModuleType):
        module = value.__name__
        if name == module:
            return f"import {module}"
        if "." not in module and name != module:
            return f"import {module} as {name}"
        parent, _, leaf = module.rpartition(".")
        if leaf == name:
            return f"from {parent} import {leaf}"
        return f"import {module} as {name}"
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname and "." not in qualname and not module.startswith("evograd"):
        alias = "" if qualname == name else f" as {name}"
        return f"from {module} import {qualname}{alias}"
    return None


def _global_names(node: ast.AST) -> list[str]:
    """Free names a definition reads, in first-use order.

    Conservative about scope: a name bound anywhere inside the definition
    (argument, assignment, nested def, import, handler) counts as local.
    """
    bound = set()
    seen: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and child is not node:
            bound.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in bound and child.id not in seen:
                seen.append(child.id)
    return seen


def reference_closure(forward_spec: str) -> tuple[str, str]:
    """Source of the forward reference and everything it reads from evograd.

    Returns ``(source, entry_name)``. Library names (``torch``, ``F``,
    ``math``...) become imports; functions and literal constants defined in an
    evograd module are copied, recursively, in dependency order. This is the
    information the frontend reaches by tracing the callable, written out so a
    text-only agent can read the same computation.
    """
    from evograd.opdecl.importing import resolve_callable

    entry = resolve_callable(forward_spec)
    imports: dict[str, str] = {}
    definitions: dict[str, str] = {}
    unresolved: list[str] = []
    order: list[str] = []

    def visit(name: str, value: Any, namespace: dict[str, Any]) -> None:
        if name in definitions or name in imports:
            return
        line = _imports_for(name, value)
        if line is not None and not (
            callable(value) and getattr(value, "__module__", "").startswith("evograd")
        ):
            imports[name] = line
            return
        if inspect.isfunction(value) or inspect.isclass(value):
            source = textwrap.dedent(inspect.getsource(value))
            tree = ast.parse(source)
            if value.__name__ != name:
                source += f"\n{name} = {value.__name__}\n"
            definitions[name] = source
            inner = getattr(value, "__globals__", None) or vars(sys.modules[value.__module__])
            for free in _global_names(tree.body[0]):
                if free in inner:
                    visit(free, inner[free], inner)
                elif not hasattr(builtins, free):
                    unresolved.append(free)
            order.append(name)
            return
        if isinstance(value, (int, float, str, bool, tuple, frozenset, type(None))):
            definitions[name] = f"{name} = {value!r}\n"
            order.append(name)
            return
        unresolved.append(name)

    name = entry.__name__
    visit(name, entry, entry.__globals__)
    if unresolved:
        raise ValueError(
            f"{forward_spec}: cannot write a standalone reference; unresolved names "
            f"{sorted(set(unresolved))}"
        )
    body = "\n\n".join(definitions[n].rstrip() + "\n" for n in order)
    header = (
        '"""The PyTorch forward reference this task is defined by.\n\n'
        f"Source: {forward_spec}\n"
        "The oracle differentiates this function with torch.autograd; candidates\n"
        "must not call it.\n"
        '"""\n'
    )
    import_block = "\n".join(sorted(set(imports.values())))
    source = f"{header}\n{import_block}\n\n\n{body}\nreference_forward = {name}\n"
    compile(source, f"<reference {forward_spec}>", "exec")
    return source, name


def _uses_contraction(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            leaf = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if leaf in _CONTRACTIONS:
                return True
    return False


def resolve_library(policy: str, reference_source: str) -> tuple[tuple[str, ...], str]:
    if policy not in LIBRARY_POLICIES:
        raise ValueError(f"library policy must be one of {LIBRARY_POLICIES}, got {policy!r}")
    if policy == "none":
        return (), "requested: no library primitives"
    if policy == "vendor_gemm":
        return ("vendor_gemm",), "requested: vendor_gemm"
    if _uses_contraction(reference_source):
        return ("vendor_gemm",), (
            "auto: the reference contains a contraction, which the unified frontend "
            "keeps in a vendor library call; vendor_gemm gives agents the same access"
        )
    return (), "auto: the reference contains no contraction"


# ── the ABI, as data ─────────────────────────────────────────────────────────


def describe_abi(op, workload) -> dict[str, Any]:
    from evograd.opdecl.activity import Active, bind_shape
    from evograd.opdecl.inputs import arg_dtype_name

    def tensor_entry(arg):
        shape = getattr(arg, "shape", None)
        if shape is None:
            return {"name": arg.name, "kind": "scalar", "default": arg.default}
        return {
            "name": arg.name,
            "kind": "active" if isinstance(arg, Active) else "inactive_tensor",
            "shape": shape,
            "size": list(bind_shape(shape, workload.dims)),
            "dtype": arg_dtype_name(arg, workload),
            **({"grad": arg.grad_name} if isinstance(arg, Active) else {}),
        }

    return {
        "forward_fn": op.forward_fn_name,
        "backward_fn": op.backward_fn_name,
        "forward_signature": f"def {op.forward_fn_name}({op.forward_parameters()})",
        "forward_returns": op.forward_returns(),
        "backward_signature": f"def {op.backward_fn_name}({op.backward_parameters()})",
        "backward_returns": op.backward_returns(),
        "arguments": [tensor_entry(arg) for arg in op.args],
        "outputs": [tensor_entry(out) for out in op.outputs],
        "upstream_grads": list(op.upstream_grad_names),
        "grad_order": list(op.grad_names()),
        "multi_output": op.is_multi_output,
    }


def _tolerances(op, workload) -> dict[str, list[float]]:
    names = [*op.output_names, *op.grad_names()]
    return {name: list(op.tolerance_for(workload, name)) for name in names}


def _evograd_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                     "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", None
    return {"commit": commit or None, "dirty": dirty}


def _task_id(op_name: str, workload) -> str:
    dims = "_".join(f"{k}{v}" for k, v in sorted(workload.dims.items()))
    short = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}[workload.dtype]
    return f"{op_name}.{short}.{dims}"


def build_task(
    op_name: str,
    output_dir: str | Path,
    *,
    declaration: str | None = None,
    dtype: str | None = None,
    dims: dict[str, int] | None = None,
    library: str = "auto",
    scoring: str = "speed_memory",
    performance_baseline: str = "auto",
    case: str = "benchmark",
) -> AgentTask:
    """Write a task pack for one operator configuration and return it loaded.

    ``case="benchmark"`` (default) targets the operator's first declared timed
    case of ``dtype``; ``case="correctness"`` reproduces ``evograd seed
    unified``'s default, the first correctness case. Explicit ``dims`` win.
    """
    from dataclasses import replace

    from evograd.adbench_tools.artifact import (
        PRIMITIVES_MODULE,
        render_primitives_module,
        render_skeleton,
    )
    from evograd.adbench_tools.statement import render_statement, system_message
    from evograd.benchmark import get_task, load_task
    from evograd.evolve.scoring import get_policy
    from evograd.opdecl.baselines import resolve_performance_baseline
    from evograd.pipelines.unified.synthesize import (
        _write_declaration,
        scoped_op,
        select_workload,
    )

    get_policy(scoring)
    dims_given = dims is not None
    if case not in CASE_SELECTIONS:
        raise ValueError(f"case must be one of {CASE_SELECTIONS}, got {case!r}")
    base = load_task(declaration) if declaration else get_task(op_name)
    if base.name != op_name:
        raise ValueError(f"declaration name {base.name!r} does not match {op_name!r}")
    if declaration:
        path, _, attribute = declaration.partition(":")
        base = replace(base, declaration=f"{Path(path).resolve()}:{attribute}")
    selection = "explicit dims" if dims_given else case
    if dims is None and case == "benchmark":
        timed = [w for w in base.benchmark_workloads() if dtype is None or w.dtype == dtype]
        if timed:
            dims, dtype = dict(timed[0].dims), timed[0].dtype
        else:
            # Nothing timed is declared (an operator kept for correctness
            # coverage); the frontend's own default is all there is.
            selection = "correctness (no declared benchmark case" + (
                f" of dtype {dtype})" if dtype else ")")
    # With explicit dims this is exactly `evograd seed unified --dims ... --dtype ...`.
    workload = select_workload(base, dtype=dtype, dims=dims)

    root = Path(output_dir).resolve()
    if (root / TASK_FILE).exists():
        raise FileExistsError(f"a task pack already exists at {root}; use a new directory")
    root.mkdir(parents=True, exist_ok=True)

    # The frontend's own scoped declaration, written by the frontend's writer.
    _write_declaration(root / DECLARATION_FILE, base, workload)
    reference, entry = reference_closure(base.forward)
    (root / REFERENCE_FILE).write_text(reference, encoding="utf-8")
    allowed, library_reason = resolve_library(library, reference)

    scoped = scoped_op(base, workload)
    baseline = resolve_performance_baseline(scoped, performance_baseline)
    spec: dict[str, Any] = {
        "schema": TASK_SCHEMA,
        "task_id": _task_id(op_name, workload),
        "op": op_name,
        "base_declaration": base.declaration,
        "forward": base.forward,
        "reference_entry": entry,
        "level": base.level,
        "family": base.family,
        "workload": {"dims": dict(workload.dims), "dtype": workload.dtype},
        "case_selection": selection,
        "abi": describe_abi(scoped, workload),
        "semantics": {
            "forward": scoped.forward_semantics,
            "backward": scoped.backward_semantics,
            "extra_constraints": scoped.extra_constraints,
        },
        "tolerances": _tolerances(scoped, workload),
        "allowed_primitives": list(allowed),
        "library_policy": {"requested": library, "decision": library_reason},
        "scoring": scoring,
        "performance_baseline": baseline,
        "evograd": _evograd_revision(),
    }
    (root / TASK_FILE).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    task = AgentTask(root=root, spec=spec)
    primitives = render_primitives_module(task)
    if primitives is not None:
        (root / f"{PRIMITIVES_MODULE}.py").write_text(primitives, encoding="utf-8")
    statement = render_statement(task)
    (root / STATEMENT_FILE).write_text(statement, encoding="utf-8")
    (root / SKELETON_FILE).write_text(render_skeleton(task), encoding="utf-8")
    spec["statement_sha256"] = hashlib.sha256(statement.encode()).hexdigest()
    spec["system_message_sha256"] = hashlib.sha256(system_message(task).encode()).hexdigest()
    (root / TASK_FILE).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return load(root)
