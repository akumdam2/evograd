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
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evograd.adbench_tools import TASK_SCHEMA
from evograd.opdecl.importing import reference_closure

TASK_FILE = "task.json"
DECLARATION_FILE = "declaration.py"
REFERENCE_FILE = "reference.py"
STATEMENT_FILE = "TASK.md"
SKELETON_FILE = "skeleton.py"

#: Library policies: ``auto`` follows the unified frontend (what Inductor hands
#: to a vendor library is granted), ``none`` grants nothing, or an explicit
#: comma-separated list of ``evograd.pipelines.shared.primitives`` names.
LIBRARY_POLICIES = ("auto", "none", "vendor_gemm", "vendor_sdpa", "vendor_gemm,vendor_sdpa")
#: Which declared configuration a task targets when no dims are given.
CASE_SELECTIONS = ("benchmark", "correctness")

#: Calls in a forward reference that the unified frontend leaves to a vendor
#: library (Inductor's extern mm/bmm/addmm/convolution). ``auto`` grants the
#: trusted GEMM primitive when the reference contains one of them, so an agent
#: writing from scratch has the library access evograd's seed preserves.
_CONTRACTIONS = frozenset({
    "linear", "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "conv1d", "conv2d",
    "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "tensordot",
})
#: Fused attention, which Inductor likewise leaves to a library (cuDNN) and the
#: frontend preserves; ``auto`` grants ``vendor_sdpa`` when the reference calls it.
_ATTENTION = frozenset({"scaled_dot_product_attention"})


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


def _called_leaves(source: str) -> set[str]:
    """Every function name the reference calls, plus ``@`` as ``matmul``."""
    leaves: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            leaves.add("matmul")
        if isinstance(node, ast.Call):
            func = node.func
            leaf = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if leaf:
                leaves.add(leaf)
    return leaves


def _uses_contraction(source: str) -> bool:
    return bool(_called_leaves(source) & _CONTRACTIONS)


def resolve_library(policy: str, reference_source: str) -> tuple[tuple[str, ...], str]:
    if policy not in LIBRARY_POLICIES:
        raise ValueError(f"library policy must be one of {LIBRARY_POLICIES}, got {policy!r}")
    if policy == "none":
        return (), "requested: no library primitives"
    if policy != "auto":
        names = tuple(sorted(policy.split(",")))
        return names, f"requested: {', '.join(names)}"
    leaves = _called_leaves(reference_source)
    granted, reasons = [], []
    if leaves & _CONTRACTIONS:
        granted.append("vendor_gemm")
        reasons.append("a contraction (vendor_gemm)")
    if leaves & _ATTENTION:
        granted.append("vendor_sdpa")
        reasons.append("fused attention (vendor_sdpa)")
    if not granted:
        return (), "auto: the reference contains no contraction or fused attention"
    return tuple(granted), (
        f"auto: the reference contains {' and '.join(reasons)}, which the unified frontend "
        "keeps in a vendor library call; the primitive gives agents the same access"
    )


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


def _gate1_tolerances(op, workload) -> dict[str, Any]:
    """The tolerance Gate 1 accepts each output and gradient under, read from Gate 1.

    Names and dtypes come from Gate 1's result contract (whose dtype the
    reference result must have), the rule and per-result ``(atol, rtol)`` from
    its tolerance policy. A model-reference task gets the fixed dtype defaults
    and the compile envelope; the envelope's value needs a GPU calibration on
    the model's payloads, so only its rule and ``gamma`` (the operator Gate 2
    calibration's, which ``integrations.gates.judge_case`` uses) are recorded.
    """
    import torch

    from evograd.opdecl.inputs import resolve_dtype
    from evograd.verification.gate1.runner import declared_name, result_contract
    from evograd.verification.gate1.tolerances import policy_identity, result_tolerances
    from evograd.verification.gate2.calibration import SAFETY_FACTOR_GAMMA

    results = {}
    for spec in result_contract(op, workload).specs:
        name = declared_name(spec.name)
        reference = torch.empty(0, dtype=resolve_dtype(spec.dtype), device="meta")
        atol, rtol = result_tolerances(op, workload, name, reference)
        results[name] = {"dtype": spec.dtype, "atol": atol, "rtol": rtol}
    policy = policy_identity(op)
    return {"policy": policy, "results": results,
            **({"envelope_gamma": SAFETY_FACTOR_GAMMA} if "envelope" in policy else {})}


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


# ── a model's observed case ──────────────────────────────────────────────────


def observed_case(model_name: str, case_id: str):
    """``(model, case)`` of one observed benchmark case, as the gates see it."""
    from evograd.verification.integrations import load as load_model
    from evograd.verification.integrations.base import observed_cases

    model = load_model(model_name)
    cases = {c.case_id: c for c in observed_cases(model)}
    if case_id not in cases:
        raise ValueError(f"{model_name} has no observed case {case_id!r}; known: {sorted(cases)}")
    return model, cases[case_id]


def _write_model_declaration(path: Path, model_name: str, task: str, workload) -> None:
    """The model's reference declaration, scoped to the observed case.

    The base is ``Model.reference_op(task)``: the task re-declared with the
    model's own computation as forward, under its model-reference contract,
    which is what every gate of ``evograd evaluate`` judges against. The case's
    workload is written whole, provenance included, so the inputs the pack
    generates (and the scalar defaults ``scoped_op`` reads off them) are the
    observed case's: its recorded strides, the model's eps.
    """
    source = f'''"""The observed configuration of {model_name}'s case {task}, under its model reference."""
from pathlib import Path
from evograd.opdecl.activity import Provenance, Workload
from evograd.pipelines.unified.synthesize import scoped_op
from evograd.verification.integrations import load

_base = load({model_name!r}).reference_op({task!r})
_case = {workload!r}
op = scoped_op(_base, _case, declaration=str(Path(__file__).resolve()) + ":op")
'''
    path.write_text(source, encoding="utf-8")


def build_task(
    op_name: str | None,
    output_dir: str | Path,
    *,
    declaration: str | None = None,
    dtype: str | None = None,
    dims: dict[str, int] | None = None,
    library: str = "auto",
    scoring: str = "speed_memory",
    performance_baseline: str = "auto",
    case: str = "benchmark",
    model: str | None = None,
    case_id: str | None = None,
) -> AgentTask:
    """Write a task pack for one operator configuration and return it loaded.

    ``case="benchmark"`` (default) targets the operator's first declared timed
    case of ``dtype``; ``case="correctness"`` reproduces ``evograd seed
    unified``'s default, the first correctness case. Explicit ``dims`` win.

    With ``model`` and ``case_id`` the task is that model's observed case
    (``evograd evaluate``'s unit): the configuration the model trains at and
    the model's own reference computation, so what the agent optimizes is what
    the gates judge. ``op_name``, ``declaration``, ``dtype``, ``dims`` and
    ``case`` are then taken from the case.
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
    observed = None
    if (model is None) != (case_id is None):
        raise ValueError("model and case_id go together")
    if model is not None:
        if declaration or dtype or dims or case != "benchmark":
            raise ValueError("a model's observed case fixes the declaration and configuration")
        model_obj, observed = observed_case(model, case_id)
        base, workload = model_obj.reference_op(observed.task), observed.workload
        op_name = base.name
        selection = f"observed case {case_id} of {model}"
    else:
        if op_name is None:
            raise ValueError("op_name is required without a model")
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

    if observed is not None:
        # The model reference, scoped to the case; the pack's own declaration
        # is then the base every agent (and evograd's seed) starts from.
        _write_model_declaration(root / DECLARATION_FILE, model, observed.task, workload)
        base = replace(base, declaration=f"{(root / DECLARATION_FILE).resolve()}:op")
    else:
        # The frontend's own scoped declaration, written by the frontend's writer.
        _write_declaration(root / DECLARATION_FILE, base, workload)
    reference, entry = reference_closure(base.forward)
    (root / REFERENCE_FILE).write_text(reference, encoding="utf-8")
    allowed, library_reason = resolve_library(library, reference)

    scoped = scoped_op(base, workload)
    baseline = resolve_performance_baseline(scoped, performance_baseline)
    spec: dict[str, Any] = {
        "schema": TASK_SCHEMA,
        "task_id": (f"{model}.{case_id.replace('/', '.')}" if observed is not None
                    else _task_id(op_name, workload)),
        "op": op_name,
        "base_declaration": base.declaration,
        "forward": base.forward,
        "reference_entry": entry,
        "level": observed.level if observed is not None else base.level,
        "family": base.family,
        "workload": {"dims": dict(workload.dims), "dtype": workload.dtype},
        "case_selection": selection,
        **({"model": model, "case_id": case_id, "frequency": observed.frequency,
            "reference_contract": base.reference_contract} if observed is not None else {}),
        "abi": describe_abi(scoped, workload),
        "semantics": {
            "forward": scoped.forward_semantics,
            "backward": scoped.backward_semantics,
            "extra_constraints": scoped.extra_constraints,
        },
        "gate1_tolerances": _gate1_tolerances(scoped, workload),
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
