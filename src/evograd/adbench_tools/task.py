"""The task pack: what every agent is given for one model's observed case.

``build_task`` writes a directory that is self-contained and portable (every
path inside it is relative to it):

    task.json        machine-readable task: op, configuration, ABI, rules
    declaration.py   the model reference scoped to the case (``scoped_op``)
    reference.py     the PyTorch forward reference with its dependencies
    TASK.md          the shared problem statement (see ``statement``)
    skeleton.py      the deliverable's signatures, bodies unimplemented

The case is one of the model's observed L1/L2 sites, as ``evograd evaluate``
judges it: the configuration the model trains at and the model's own reference
computation. ``task.json``'s ``search_correctness`` is the rule every agent's
candidates are judged by during search (Gate 1 on the case's calibration
payloads); ``evograd seed --pack`` and ``evograd evolve --pack`` use the same
pack.
"""

from __future__ import annotations

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

#: Torch functions that the unified frontend leaves to a vendor library
#: (Inductor's extern mm/bmm/addmm/convolution). ``auto`` grants the trusted GEMM
#: primitive when one call of the reference reaches one of them, so an agent
#: writing from scratch has the library access evograd's seed preserves.
_CONTRACTIONS = frozenset({
    "linear", "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "_grouped_mm",
    "grouped_mm", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "conv_transpose3d", "tensordot",
})
#: Fused attention, which Inductor likewise leaves to a library (cuDNN) and the
#: frontend preserves; ``auto`` grants ``vendor_sdpa`` when the reference reaches it.
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


def observed_calls(op, workload, *, device: str) -> list[str]:
    """The torch functions one call of ``op``'s reference reaches at ``workload``,
    however deeply it delegates (shared helpers, Hugging Face modules,
    ``transformers.integrations``).

    Recorded at the function level (a ``TorchFunctionMode``), where ``@`` is
    ``matmul`` and fused attention is ``scaled_dot_product_attention`` on every
    device; the ATen level would see attention on the meta device as its math
    decomposition. Run on the meta device, and on ``device`` when the reference
    needs values.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import resolve_forward

    class Reached(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.names: set[str] = set()

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.names.add(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    forward = resolve_forward(op)
    for where in dict.fromkeys(("meta", device)):
        reached = Reached()
        try:
            values = make_case_inputs(op, workload, device=where)
            with reached:
                forward(*(values.get(a.name, getattr(a, "default", None)) for a in op.args))
            return sorted(reached.names)
        except Exception:
            if where == device:
                raise


def resolve_library(policy: str, calls) -> tuple[tuple[str, ...], str]:
    """The primitives ``policy`` grants. ``auto`` reads ``calls``, the torch
    functions one call of the reference reaches (:func:`observed_calls`)."""
    if policy not in LIBRARY_POLICIES:
        raise ValueError(f"library policy must be one of {LIBRARY_POLICIES}, got {policy!r}")
    if policy == "none":
        return (), "requested: no library primitives"
    if policy != "auto":
        names = tuple(sorted(policy.split(",")))
        return names, f"requested: {', '.join(names)}"
    reached = set(calls)
    granted, reasons = [], []
    if reached & _CONTRACTIONS:
        granted.append("vendor_gemm")
        reasons.append("a contraction (vendor_gemm)")
    if reached & _ATTENTION:
        granted.append("vendor_sdpa")
        reasons.append("fused attention (vendor_sdpa)")
    if not granted:
        return (), "auto: one call of the reference reaches no contraction or fused attention"
    return tuple(granted), (
        f"auto: one call of the reference reaches {' and '.join(reasons)}, which the unified "
        "frontend keeps in a vendor library call; the primitive gives agents the same access"
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
    its tolerance policy: the fixed dtype defaults and the compile envelope. The
    envelope's value needs a GPU calibration on the model's payloads, so only its
    rule and Gate 1's factor (``gate1.tolerances.ENVELOPE_GAMMA``) are recorded.
    """
    import torch

    from evograd.opdecl.inputs import resolve_dtype
    from evograd.verification.gate1.runner import declared_name, result_contract
    from evograd.verification.gate1.tolerances import (
        ENVELOPE_GAMMA,
        policy_identity,
        result_tolerances,
    )

    results = {}
    for spec in result_contract(op, workload).specs:
        name = declared_name(spec.name)
        reference = torch.empty(0, dtype=resolve_dtype(spec.dtype), device="meta")
        atol, rtol = result_tolerances(op, workload, name, reference)
        results[name] = {"dtype": spec.dtype, "atol": atol, "rtol": rtol}
    return {"policy": policy_identity(op), "results": results, "envelope_gamma": ENVELOPE_GAMMA}


def _search_correctness(model_obj, observed, corpus: str | Path | None, *,
                        device: str, state: Path | None) -> dict[str, Any]:
    """What the evaluator judges a model case's candidates by, frozen in the pack:
    Gate 1 on the first, middle and last call of each calibration order, with the
    compile envelope operator Gate 2's calibration gives the case (a GPU run, as
    ``integrations.gates.judge_case`` makes it).
    """
    from evograd.verification.payloads.gate1 import SEARCH_RULE

    if corpus is None:
        raise ValueError(f"{SEARCH_RULE!r} needs the model's payload corpus (its manifest.json)")
    from evograd.verification.integrations.base import configure
    from evograd.verification.integrations.gates import gate1_envelope, operator_calibration
    from evograd.verification.payloads.corpus import Corpus
    from evograd.verification.payloads.gate1 import SEARCH_SELECTION, search_payload_ids

    manifest = Path(corpus).resolve()
    opened = Corpus.open(manifest)
    payload_ids = search_payload_ids(opened, observed.case_id)
    # Calibrated as `evograd evaluate` calibrates (deterministic algorithms, no
    # TF32): the envelope is then Gate 1's own, and a stored calibration binds.
    configure()
    scope, calibration = operator_calibration(model_obj, observed, opened, device=device,
                                              state=state)
    return {"rule": SEARCH_RULE, "case_id": observed.case_id, "selection": SEARCH_SELECTION,
            "corpus": {"manifest": str(manifest), "corpus_id": opened.corpus_id,
                       "sha256": opened.sha256},
            "payload_ids": payload_ids,
            "envelope": gate1_envelope(scope.op, calibration.policy),
            "calibrated": calibration.policy is not None}


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
    output_dir: str | Path,
    *,
    model: str,
    case_id: str,
    corpus: str | Path | None = None,
    library: str = "auto",
    scoring: str = "speed_memory",
    performance_baseline: str = "auto",
    device: str = "cuda",
    state: str | Path | None = None,
) -> AgentTask:
    """Write the task pack of ``model``'s observed case ``case_id`` and return it loaded.

    The case is ``evograd evaluate``'s unit: the configuration the model trains
    at and the model's own reference computation, so what the agent optimizes is
    what the gates judge. The evaluator judges every agent's candidates by
    :data:`~evograd.verification.payloads.gate1.SEARCH_RULE` during search,
    which needs the model's payload ``corpus`` and a GPU (the compile envelope's
    calibration).
    """
    from dataclasses import replace

    from evograd.adbench_tools.artifact import (
        PRIMITIVES_MODULE,
        render_primitives_module,
        render_skeleton,
    )
    from evograd.adbench_tools.statement import render_statement, system_message
    from evograd.evolve.scoring import get_policy
    from evograd.opdecl.baselines import resolve_performance_baseline
    from evograd.pipelines.unified.synthesize import scoped_op

    get_policy(scoring)
    model_obj, observed = observed_case(model, case_id)
    base, workload = model_obj.reference_op(observed.task), observed.workload
    op_name = base.name

    root = Path(output_dir).resolve()
    if (root / TASK_FILE).exists():
        raise FileExistsError(f"a task pack already exists at {root}; use a new directory")
    # Before the pack exists: a trace or calibration that fails leaves no half-written pack.
    calls = observed_calls(base, workload, device=device)
    allowed, library_reason = resolve_library(library, calls)
    search = _search_correctness(model_obj, observed, corpus, device=device,
                                 state=Path(state) if state else None)
    root.mkdir(parents=True, exist_ok=True)

    # The model reference, scoped to the case; the pack's own declaration is then
    # the base every agent (and evograd's seed) starts from.
    _write_model_declaration(root / DECLARATION_FILE, model, observed.task, workload)
    base = replace(base, declaration=f"{(root / DECLARATION_FILE).resolve()}:op")
    reference, entry = reference_closure(base.forward)
    (root / REFERENCE_FILE).write_text(reference, encoding="utf-8")

    scoped = scoped_op(base, workload)
    baseline = resolve_performance_baseline(scoped, performance_baseline)
    spec: dict[str, Any] = {
        "schema": TASK_SCHEMA,
        "task_id": f"{model}.{case_id.replace('/', '.')}",
        "op": op_name,
        "base_declaration": base.declaration,
        "forward": base.forward,
        "reference_entry": entry,
        "level": observed.level,
        "family": base.family,
        "workload": {"dims": dict(workload.dims), "dtype": workload.dtype},
        "case_selection": f"observed case {case_id} of {model}",
        "model": model, "case_id": case_id, "frequency": observed.frequency,
        "reference_contract": base.reference_contract,
        "abi": describe_abi(scoped, workload),
        "semantics": {
            "forward": scoped.forward_semantics,
            "backward": scoped.backward_semantics,
            "extra_constraints": scoped.extra_constraints,
        },
        "gate1_tolerances": _gate1_tolerances(scoped, workload),
        "search_correctness": search,
        "allowed_primitives": list(allowed),
        "library_policy": {"requested": library, "decision": library_reason,
                           "observed_calls": sorted(set(calls) & (_CONTRACTIONS | _ATTENTION))},
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
