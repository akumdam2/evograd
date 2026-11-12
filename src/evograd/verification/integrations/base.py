"""What every model integration shares: cases, the capture loop, replay, determinism.

A model contributes a :class:`Model`: how to load its pinned pretrained
workload on real data, how to record its training calls (a live recorder and
the mapping of recorded calls to task ABIs), and its model reference. Cases,
payload records, splits, coverage, the per-order capture with its fidelity
check and the replay against the model reference are the same for every
model and live here; the gates are :mod:`.gates`.

Cases are a model's observed benchmark cases (its ``<model>_observed`` suite).
A recorded call belongs to the case of its task whose declared shapes and
dtypes its arguments bind; calls per training step come from the harvest
snapshot. A capture records the payloads of the cases its selection picks
(:func:`recorded_cases`: by default the model's benchmark rows in the search
set), and its corpus identity states that selection. Nothing here edits a task
or a snapshot.

The harvest that writes that snapshot, for every model but MegaFold, runs the
capture's own first step (:func:`real_run`: the setup :func:`capture` shares,
:func:`prepare_run`, and the first TRAIN batch of order :data:`HARVEST_ORDER`)
and records its :func:`run_identity` as the snapshot's ``run``. A snapshot is
written only from such a harvest (:func:`require_real_run`), and
:func:`capture` refuses a snapshot whose ``run`` is missing or not its own
(:func:`require_harvested_run`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

ORDERS = (101, 202, 303, 404)
CALIBRATION_ORDERS = (101, 202)
ACCEPTANCE_ORDERS = (303, 404)
#: The data order whose first TRAIN batch the harvest runs: the capture's first.
HARVEST_ORDER = ORDERS[0]
#: The fields of a :func:`run_identity` a capture compares with its snapshot's ``run``.
RUN_KEYS = ("model", "workload", "data", "order", "batch", "parameter_state")
DETERMINISM = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "use_deterministic_algorithms": True,
               "cudnn.deterministic": True, "cudnn.benchmark": False,
               "cuda.matmul.allow_tf32": False, "float32_matmul_precision": "highest",
               "torch_num_threads": 8}
#: The capture identity's loss text, so it stays as recorded: it omits the auxiliary
#: term a model adds to ``workload.loss`` (OLMoE's router loss), which the gates'
#: ``common.execution.WORKLOAD_LM_LOSS`` names.
LOSS_DEFINITION = ("the HF causal-LM loss: model(input_ids, labels=labels).loss, float32 logits, "
                   "labels shifted once with -100 padding, mean over valid tokens; backward of "
                   "that scalar")


#: Gate 3's protocol horizon (AdamW updates per run), 1000 until 2026-10-27. Over 1000
#: updates every perturbation ends in the same chaotic spread: a 5% fault at one site
#: moves the loss only 1.2-1.9x as far as compiling that site, and a correct candidate's
#: single-step peak can reach 2.3x compile's (Qwen3-0.6B, Llama-3.2-1B, ConvNeXt V2,
#: DLRM, DiT, Qwen2-VL). Over the first 100 updates of the same runs (a run's schedule
#: depends only on the update index) every frontend seed needs at most 1.4 and the fault
#: at least 5.3 on Qwen3-0.6B, Llama-3.2-1B, DLRM and DiT; ConvNeXt V2 and Qwen2-VL set
#: their own horizons.
GATE3_STEPS = 100


@dataclass(frozen=True)
class Model:
    """What one model contributes to the shared capture, replay and gates."""

    name: str
    suite: str
    #: Pinned weights and real data, recorded in the corpus identity.
    workload_config: Mapping[str, Any]
    #: ``() -> workload`` at ``workload_config`` (a tier-3 ``TrainingWorkload``).
    load_workload: Callable[[], Any]
    #: ``(workload, orders) -> ({order: (input_ids, labels)}, identity)``: the
    #: first TRAIN batch of each data order.
    train_batches: Callable[[Any, Sequence[int]], tuple[dict, dict]]
    #: ``model -> recorder`` with ``install()`` / ``remove()``.
    live_capture: Callable[[Any], Any]
    #: ``(recorder, model, cases) -> invocations`` in each case's task ABI.
    invocations: Callable[[Any, Any, Sequence["Case"]], Iterable[dict[str, Any]]]
    #: ``task -> OpDecl`` re-declared with the model's own computation as forward.
    reference_op: Callable[[str], Any]
    contract: str
    #: The Transformers source the reference follows (``source_identity()``).
    source_identity: Callable[[], dict]
    #: Modules whose source is hashed into the corpus identity.
    modeling_modules: tuple[str, ...] = field(default_factory=tuple)
    #: Gate 3 horizon (AdamW updates per training run).
    gate3_steps: int = GATE3_STEPS
    #: Gate 3 optimizer settings beyond the protocol's constant-rate AdamW
    #: (``gate3.adamw`` keywords: ``lr``, ``warmup_steps``, ``clip_grad_norm``).
    #: Empty is the protocol default; whatever is set enters every run identity.
    gate3_optimizer: Mapping[str, Any] = field(default_factory=dict)
    #: Training batches per Gate 3 update (gradient accumulation), for a model
    #: whose batch is too small to carry a stable update but must keep its shape.
    gate3_accumulation: int = 1
    #: Gate 3 safety factor where this model needs its own; ``None`` is the
    #: protocol's (``gates.GAMMA``).
    gate3_gamma: float | None = None
    #: Model-scope Gate 2 safety factor where this model needs its own; ``None`` is
    #: the protocol's (``gates.GATE2_MODEL_GAMMA``).
    gate2_model_gamma: float | None = None
    #: How a training step is run, recorded and judged
    #: (:mod:`.objectives`); ``None`` is the HF causal-LM objective.
    objective: Any = None
    #: Whether Level-2 candidates also get model-scope Gate 2 and Gate 3. A
    #: model without a well-posed whole-model comparison yet is verified by
    #: Gate 1 and operator Gate 2 only.
    model_gates: bool = True
    #: Whether Gate 3 belongs to the model gates. MegaFold: no (its continued
    #: training is ~2.3 h per 1000 updates; user decision 2026-10-27), so its
    #: Level-2 candidates and their combination are accepted on model Gate 2.
    gate3_applies: bool = True
    #: ``case -> calls per step the corpus records`` for a model whose capture
    #: records a declared subset of each case's calls (MegaFold: a full step is
    #: 275 GiB of payloads); ``None`` records every call.
    recorded_calls: Callable[["Case"], int] | None = None
    #: Whether a capture requires the tracked snapshot to record this run
    #: (:func:`require_harvested_run`). MegaFold: no, a deliberate exemption
    #: (its harvest is left as it is). That harvest runs the capture's own
    #: workload, model builder, recorder and settings at step seed 0 and records
    #: the run as ``config`` and ``data``, not as a :func:`run_identity`; its
    #: recorder checks the snapshot's schema instead.
    requires_harvested_run: bool = True


def objective_of(model: Model):
    """The model's objective, the causal-LM one by default."""
    from .objectives import NextTokenLM

    return model.objective if model.objective is not None else NextTokenLM()


#: The CUDA allocator evaluation runs with unless the caller set one: whole-model
#: gate runs fragment memory (OLMoE's combination calibration ran out with 13.7 GiB
#: free in fragments, 2026-11-11). Baselines and candidates of one evaluation share
#: it, and its environment records it.
CUDA_ALLOC_CONF = "expandable_segments:True"


def default_allocator() -> None:
    """:data:`CUDA_ALLOC_CONF`, unless ``PYTORCH_CUDA_ALLOC_CONF`` is set; call it
    before the process allocates on the GPU (child processes inherit it)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)


def configure() -> dict[str, Any]:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISM["CUBLAS_WORKSPACE_CONFIG"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before the process starts")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(DETERMINISM["torch_num_threads"])
    # Here as well as at model construction: a job-local model snapshot is
    # loaded without constructing, and compile or capture must still see it.
    from evograd.benchmark.topdown.common.model import traced_causal_mask_as_eager
    traced_causal_mask_as_eager()
    return determinism_record()


def determinism_record() -> dict[str, Any]:
    import torch

    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
            "sdp_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdp_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "sdp_math": torch.backends.cuda.math_sdp_enabled(),
            "torch_num_threads": torch.get_num_threads()}


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── cases ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One observed benchmark case and how often a training step calls it."""

    case_id: str
    task: str
    level: int
    workload: Any
    frequency: int
    #: The module paths the harvest observed this configuration at (when it
    #: records them): what tells apart configurations with identical task inputs.
    paths: tuple[str, ...] = ()
    #: Calls per step the corpus records, when the model records a subset.
    recorded: int | None = None

    @property
    def calls_recorded(self) -> int:
        """Calls per step the corpus holds: every one, or the model's declared subset."""
        return self.frequency if self.recorded is None else self.recorded


def observed_cases(model: Model) -> tuple[Case, ...]:
    """Every observed L1 and L2 case of the model, calls per step from the snapshot.

    A Level-2 entry whose task the code no longer registers (a task removed or
    merged since the harvest) is no case; the next harvest drops it. A task added
    since is no case until that harvest records it, so a capture refuses its
    calls (:func:`case_for`) and a candidate for it is refused.
    """
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown import load_snapshot

    snapshot = load_snapshot(model.name)
    cases = []
    for task, entry in snapshot["level1"].items():
        workloads = get_task(task).benchmark_suites[model.suite]
        configs = entry["configurations"]
        if len(workloads) != len(configs) or any(dict(w.dims) != c["dims"]
                                                 for w, c in zip(workloads, configs)):
            raise ValueError(f"{task}: suite {model.suite} does not follow the snapshot")
        cases += [Case(f"{task}/{w.provenance.component}", task, 1, w, int(c["frequency"]),
                       tuple(c.get("module_paths") or ()))
                  for w, c in zip(workloads, configs)]
    for task, entry in snapshot["tasks"].items():
        try:
            op = get_task(task)
        except KeyError:
            continue
        if "configurations" in entry:
            # A harvest that observes each L2 boundary per configuration (vision,
            # generated models): one case per configuration, like Level 1.
            workloads = op.benchmark_suites[model.suite]
            configs = entry["configurations"]
            if len(workloads) != len(configs) or any(dict(w.dims) != c["dims"]
                                                     for w, c in zip(workloads, configs)):
                raise ValueError(f"{task}: suite {model.suite} does not follow the snapshot")
            cases += [Case(f"{task}/{w.provenance.component}", task, 2, w, int(c["frequency"]),
                           tuple(c.get("module_paths") or ()))
                      for w, c in zip(workloads, configs)]
            continue
        (workload,) = op.benchmark_suites[model.suite]
        # A Level-2 entry's ``frequency`` counts its anchor op; the boundary's
        # own calls per step are the fusion sites or the observed invocations.
        frequency = ((entry.get("fusion_sites") or {}).get("total")
                     or (entry.get("observed") or {}).get("whole_model_invocations")
                     or entry["frequency"])
        cases.append(Case(f"{task}/{workload.provenance.component}", task, 2, workload,
                          int(frequency)))
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{model.name}: case ids repeat {sorted({i for i in ids if ids.count(i) > 1})}; "
                         "each configuration's provenance component must name its role")
    if model.recorded_calls is not None:
        cases = [dataclasses.replace(c, recorded=model.recorded_calls(c)) for c in cases]
    return tuple(cases)


def case_component(case: Case) -> str:
    """The role half of ``task/component``."""
    return case.case_id.split("/", 1)[1]


def case_for(cases: Sequence[Case], task: str, args: dict[str, Any], *,
             component: str | None = None, path: str | None = None) -> Case:
    """The one case of ``task`` whose declared shapes and dtypes ``args`` bind.

    When several cases bind the same shapes, the module ``path`` the harvest
    observed (ConvNeXt's norms, whose task inputs coincide), ``component`` (the
    recorded call's role, e.g. OLMoE's q and k rope), then the argument layouts
    (recorded strides, else contiguous), then the static parameters the
    configuration records (``Provenance.free``) decide.
    """
    import torch

    from evograd.benchmark import get_task
    from evograd.opdecl.activity import bind_shape
    from evograd.opdecl.inputs import arg_dtype_name, resolve_dtype

    op = get_task(task)
    shaped = [a for a in op.args if getattr(a, "shape", None) not in (None, "unspecified")
              and torch.is_tensor(args.get(a.name))]

    def binds(case: Case) -> bool:
        for arg in shaped:
            value = args[arg.name]
            try:
                shape = tuple(bind_shape(arg.shape, case.workload.dims))
            except (KeyError, ValueError):
                return False
            if shape != tuple(value.shape) or \
                    value.dtype != resolve_dtype(arg_dtype_name(arg, case.workload)):
                return False
        return True

    def laid_out(case: Case) -> bool:
        # A case records the strides of every argument not laid out contiguously
        # (``Provenance.strides``); every other shaped argument is contiguous.
        recorded = dict(getattr(getattr(case.workload, "provenance", None), "strides", None) or ())
        for arg in shaped:
            value = args[arg.name]
            expected = tuple(recorded.get(arg.name) or
                             torch.empty(tuple(value.shape), device="meta").stride())
            if tuple(value.stride()) != expected:
                return False
        return True

    hits = [c for c in cases if c.task == task and binds(c)]
    if len(hits) > 1 and path:
        # The harvest observed each configuration at known modules.
        module = path.split("#", 1)[0]
        hits = [c for c in hits if module in c.paths] or hits
    if len(hits) > 1 and component is not None:
        # The role picks among cases that only the harvest tells apart (how the
        # model's tensor became the task's view); a role no tied case carries
        # (a module merged into another's configuration) leaves them all.
        hits = [c for c in hits if case_component(c) == component] or hits
    def static_match(case: Case) -> bool:
        # Static parameters the configuration records (``Provenance.free``, e.g.
        # a norm's eps) must equal the call's scalar arguments of that name.
        free = dict(getattr(getattr(case.workload, "provenance", None), "free", None) or {})
        return all(free[name] == value for name, value in args.items()
                   if name in free and not torch.is_tensor(value))

    if len(hits) > 1:
        # Same shapes and dtypes, different layouts (ConvNeXt's channels-first
        # stem norm against its blocks' norms): the layout decides.
        hits = [c for c in hits if laid_out(c)]
    if len(hits) > 1:
        # Same layout too, different static parameters (a block norm's eps
        # 1e-6 against the final norm's 1e-12).
        hits = [c for c in hits if static_match(c)]
    if len(hits) != 1:
        shapes = {a.name: tuple(args[a.name].shape) for a in shaped}
        raise KeyError(f"{task}: {len(hits)} observed cases match {shapes}"
                       + (f" and component {component!r}" if component is not None else ""))
    return hits[0]


def invocation(cases: Sequence[Case], task: str, level: int, call: dict, args: dict,
               cotangents: dict, live_outputs: dict, notes: dict | None = None) -> dict[str, Any]:
    """One recorded call as an invocation of its case.

    ``call["component"]``, when a recorder sets it, resolves cases whose shapes
    coincide (and stays in the payload's call record).
    """
    case = case_for(cases, task, args, component=call.get("component"), path=call.get("path"))
    return {"case_id": case.case_id, "task": task, "level": level,
            "call": call, "args": args, "cotangents": cotangents,
            "live_outputs": live_outputs, "notes": notes or {}}


def layer_index(path: str) -> int | None:
    """``model.layers.<i>...`` (or ``...#call``) -> ``i``."""
    parts = path.split("#", 1)[0].split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return None


def shared_row(table):
    """A rotary table ``[B, T, D]`` with identical rows (no padding): its first row ``[1, T, D]``."""
    import torch

    first = table[:1]
    if table.shape[0] > 1 and not torch.equal(table, first.expand_as(table)):
        raise ValueError("cos/sin rows differ across the batch; one shared table does not apply")
    return first


# ── one training step ────────────────────────────────────────────────────────


def build_model(workload):
    """A fresh native model at the pinned initial state (no site replaced)."""
    from evograd.evaluation.tier3.patch import KernelSet

    model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
    return model.train()


def parameter_state(model) -> dict[str, Any]:
    from evograd.verification.common.tensors import tensor_digest

    digests = {name: tensor_digest(p) for name, p in model.named_parameters()}
    embed = model.get_input_embeddings().weight
    return {"parameters": len(digests),
            "tied_lm_head": model.get_output_embeddings().weight is embed,
            "state_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}


def run_step(model, ids, labels):
    output = model(input_ids=ids, labels=labels, use_cache=False)
    output.loss.backward()
    return output


def step_record(model, output) -> dict[str, Any]:
    """Loss, logits and every parameter gradient, by content hash."""
    from evograd.verification.common.tensors import tensor_digest

    grads = {name: (tensor_digest(p.grad) if p.grad is not None else None)
             for name, p in model.named_parameters()}
    return {"loss": float(output.loss.detach()), "loss_digest": tensor_digest(output.loss.detach()),
            "logits_digest": tensor_digest(output.logits.detach()), "grads": grads}


# ── the harvested run ────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """``value`` as JSON reads it back, so a recorded copy compares equal."""
    return json.loads(json.dumps(value, default=str))


def workload_pins(model: Model) -> dict[str, Any]:
    """``model.workload_config`` as a run identity names it.

    A file the config points at (``init_checkpoint``: DiT's warm start) is
    named by the SHA-256 of its content (``init_checkpoint_sha256``), not by
    its path, which depends on the environment (``EVOGRAD_BENCHMARK_DATA_ROOT``,
    ``EVOGRAD_<MODEL>_WARM_START``). The file holds its recipe, so the digest
    pins that too; a warm start regenerated since the harvest has another one.
    """
    pins = dict(model.workload_config)
    path = pins.pop("init_checkpoint", None)
    if path is not None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{model.name}: init_checkpoint {path} is not readable here; "
                                    "the run identity names it by its content")
        pins["init_checkpoint_sha256"] = sha_file(path)
    return _plain(pins)


def run_identity(model: Model, state: Mapping[str, Any], batch_identity: Mapping[str, Any], *,
                 order: int = HARVEST_ORDER) -> dict[str, Any]:
    """The run a harvest observes and a capture's order-``order`` step repeats.

    ``state`` is the objective's ``parameter_state`` of a freshly built model;
    ``batch_identity`` is ``model.train_batches``' identity, of order ``order``
    alone (a harvest) or of every order (a capture). The record has the
    :data:`RUN_KEYS`: ``model`` (its name), ``workload`` (:func:`workload_pins`:
    the weights and data pins), ``data`` (the data identity without its
    per-order batches), ``order``, ``batch`` (that order's first batch, with
    its content digest) and ``parameter_state`` (the initial parameters, with
    their ``state_sha256``), as plain JSON.
    """
    return _plain({
        "model": model.name, "workload": workload_pins(model),
        "data": {key: value for key, value in batch_identity.items() if key != "batches"},
        "order": order, "batch": dict(batch_identity["batches"][str(order)]),
        "parameter_state": dict(state)})


def prepare_run(model: Model, orders: Sequence[int], *,
                device: str) -> tuple[dict[str, Any], Any, dict, dict[str, Any]]:
    """What a capture and a harvest both run on: ``(settings, workload, batches, batch_identity)``.

    Fails at once when ``device`` is CUDA and none is visible (before the
    checkpoint and the data are read); then :func:`configure`, the pinned
    workload (``model.load_workload``) and the first TRAIN batch of each data
    order (``model.train_batches``).
    """
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{model.name}: no CUDA device is visible; run on a GPU node")
    settings = configure()
    workload = model.load_workload()
    batches, batch_identity = model.train_batches(workload, tuple(orders))
    return settings, workload, batches, batch_identity


def real_run(model: Model, *, device: str = "cuda") -> tuple[Any, Any, Any, dict[str, Any]]:
    """The capture's first step, for a harvest to observe: ``(workload, net, batch, identity)``.

    Every harvest but MegaFold's builds its run here: :func:`prepare_run`
    for order :data:`HARVEST_ORDER`, a fresh native model at its initial
    state (:func:`build_model`, as every capture builds it) and that order's
    first TRAIN batch on ``device``. ``identity`` is :func:`run_identity` plus
    the settings it runs under (``execution``, recorded, not compared); the
    harvest records it as its manifest's and snapshot's ``run``.
    """
    from .objectives import to_device

    settings, workload, batches, batch_identity = prepare_run(model, (HARVEST_ORDER,), device=device)
    net = build_model(workload)
    identity = {**run_identity(model, objective_of(model).parameter_state(net), batch_identity),
                "execution": settings}
    return workload, net, to_device(batches[HARVEST_ORDER], device), identity


def _rerun(model: Model) -> str:
    from evograd.benchmark.topdown import TOPDOWN_WORKLOADS, harvest_commands

    if model.name not in TOPDOWN_WORKLOADS:
        return "regenerate it from a harvest of this run"
    return "regenerate it on a GPU node:\n" + harvest_commands(model.name)


def require_real_run(model: Model, manifest: Mapping[str, Any]) -> None:
    """Refuse to write a tracked snapshot from a harvest of anything but :func:`real_run`.

    Every snapshot writer calls this (``harvest.snapshot --write`` and
    ``harvest.harvest --write-snapshot``); it needs no GPU. The manifest's
    ``run`` must hold every :data:`RUN_KEYS` field, be of order
    :data:`HARVEST_ORDER` and of this model, and carry the workload pins as
    they are now (:func:`workload_pins`: a checkpoint or warm start changed
    since the harvest is refused), and the manifest must describe the
    verification workload (its ``workload_id``). A ``--synthetic`` or reduced
    harvest records no run and is refused.
    """
    run = manifest.get("run")
    if not run:
        problem = "it records no run (a --synthetic or reduced harvest)"
    elif any(key not in run for key in RUN_KEYS):
        problem = f"its run lacks {[key for key in RUN_KEYS if key not in run]}"
    elif run["order"] != HARVEST_ORDER:
        problem = f"its run is of order {run['order']}, not {HARVEST_ORDER}"
    elif run["model"] != model.name:
        problem = f"its run is of {run['model']!r}"
    elif _plain(run["workload"]) != workload_pins(model):
        problem = (f"its run's workload pins {run['workload']} are not the current "
                   f"{workload_pins(model)}")
    elif manifest.get("workload_id") != model.load_workload().spec.workload_id:
        problem = f"it describes {manifest.get('workload_id')!r}, not the verification workload"
    else:
        return
    raise ValueError(f"{model.name}: only a harvest of the verification run writes the tracked "
                     f"snapshot, and {problem}; {_rerun(model)}")


def require_harvested_run(model: Model, identity: Mapping[str, Any]) -> None:
    """Refuse to capture against a snapshot that was not harvested on this run.

    The capture binds every recorded call to a snapshot case and counts it
    against that case's calls per step; those describe this capture only if
    the harvest ran the same pinned weights, data and first batch of order
    :data:`HARVEST_ORDER` from the same initial parameters. ``identity`` is the
    capture's own :func:`run_identity`; every :data:`RUN_KEYS` field must equal
    the snapshot's ``run``. A snapshot without one (harvested before the
    harvest ran the real step, or from a random-weight synthetic-token debug
    harvest) is refused as well. Models with ``requires_harvested_run`` False
    (MegaFold) are not checked.
    """
    from evograd.benchmark.topdown import load_snapshot

    if not model.requires_harvested_run:
        return
    recorded = load_snapshot(model.name).get("run")
    if not recorded:
        raise RuntimeError(f"{model.name}: the tracked snapshot records no harvested run, so its "
                           f"cases may not describe this capture; {_rerun(model)}")
    differ = [key for key in RUN_KEYS if recorded.get(key) != identity.get(key)]
    if differ:
        raise RuntimeError(f"{model.name}: the tracked snapshot was harvested on another run "
                           f"({', '.join(differ)} differ); {_rerun(model)}")


# ── payloads ─────────────────────────────────────────────────────────────────


def record_step(writer, invocations: Iterable[dict[str, Any]], batch: dict[str, Any], *,
                case_ids: Collection[str] | None = None) -> int:
    """Write every invocation of one recorded step (of ``case_ids`` only, when
    given) as payloads; returns how many were written.

    ``index_in_step`` counts every invocation, written or not, so a call's
    record is the same whichever cases a capture records.
    """
    import torch

    from evograd.verification.common.tensors import tensor_digest
    from evograd.verification.payloads.corpus import cotangent_stats

    writer.new_step()
    count = 0
    for index, inv in enumerate(invocations):
        if case_ids is not None and inv["case_id"] not in case_ids:
            continue
        pid = f"o{batch['order']}/{inv['case_id']}/{inv['call']['path']}"
        args = {name: writer.put(value) if torch.is_tensor(value) else {"scalar": value}
                for name, value in inv["args"].items()}
        cots = {name: writer.put(value) for name, value in inv["cotangents"].items()}
        stats = {name: cotangent_stats(value) for name, value in inv["cotangents"].items()}
        for name in inv["notes"].get("absent_cotangents", ()):
            stats[name]["origin"] = "absent_in_model_zero_supplied"
        storages: dict[str, list[str]] = {}
        for name, rec in {**args, **cots}.items():
            if "storage" in rec:
                storages.setdefault(rec["storage"], []).append(name)
        writer.add_payload({
            "payload_id": pid, "case_id": inv["case_id"], "task": inv["task"], "level": inv["level"],
            "batch": dict(batch), "call": {**inv["call"], "index_in_step": index},
            "args": args, "cotangents": cots, "cotangent_stats": stats,
            "aliases": [sorted(names) for names in storages.values() if len(names) > 1],
            # A recorder may hand the summary over instead of the output (MegaFold's
            # outputs are a third of a step's recorded bytes).
            "live_outputs": {name: (dict(v) if isinstance(v, Mapping) else
                                    {"digest": tensor_digest(v), "shape": list(v.shape),
                                     "stride": list(v.stride()), "dtype": str(v.dtype)})
                             for name, v in inv["live_outputs"].items()},
            "notes": inv["notes"]})
        count += 1
    # The writer holds every storage of the step until its next ``new_step``; once
    # the step is written they can go, rather than outlive the next step's capture.
    writer.new_step()
    return count


def splits(payloads: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Calibration (orders 101/202), acceptance (303/404) and a development subset.

    Development subset: per case, role and calibration order, the first,
    middle and last recorded layer; frozen with the corpus.
    """
    calibration = [p["payload_id"] for p in payloads if p["batch"]["order"] in CALIBRATION_ORDERS]
    acceptance = [p["payload_id"] for p in payloads if p["batch"]["order"] in ACCEPTANCE_ORDERS]
    groups: dict[tuple, list[dict]] = {}
    for p in payloads:
        if p["batch"]["order"] in CALIBRATION_ORDERS:
            groups.setdefault((p["case_id"], p["call"]["role"], p["batch"]["order"]), []).append(p)
    development = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda p: (p["call"]["layer"] if p["call"]["layer"] is not None
                                                     else -1, p["call"]["path"]))
        picks = {members[0]["payload_id"], members[len(members) // 2]["payload_id"],
                 members[-1]["payload_id"]}
        development += [m["payload_id"] for m in members if m["payload_id"] in picks]
    return {"calibration": calibration, "acceptance": acceptance, "development": development}


def coverage(cases: Sequence[Case], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Every case received exactly the calls per step it records, in every recorded order."""
    expected = {c.case_id: c.calls_recorded for c in cases}
    seen: dict[str, dict[str, int]] = {}
    for p in payloads:
        per_case = seen.setdefault(p["case_id"], {})
        order = str(p["batch"]["order"])
        per_case[order] = per_case.get(order, 0) + 1
    orders = sorted({str(p["batch"]["order"]) for p in payloads})
    problems = [f"{case}: order {o}: {seen.get(case, {}).get(o, 0)} of {n} invocations"
                for case, n in expected.items() for o in orders if seen.get(case, {}).get(o, 0) != n]
    unmapped = sorted(set(seen) - set(expected))
    return {"expected_per_order": expected, "observed": seen, "orders": orders,
            "problems": problems, "unmapped_cases": unmapped,
            "complete": not problems and not unmapped}


# ── capture driver ───────────────────────────────────────────────────────────


#: What a capture records (``--cases``). ``benchmark``, the default (user decision
#: 2026-11-11): the model's benchmark rows, the only cases an evaluation verifies
#: or times; every other observed case stays in the snapshot without payloads.
#: ``all``: every observed case.
CASE_SELECTIONS = ("benchmark", "all")
BENCHMARK_CASES = ("the model's benchmark rows in the search set "
                   "(evograd.benchmark.search_set.evaluated): its searched cases plus its "
                   "identical_config_of cases")


def recorded_cases(model: Model, cases: Sequence[Case],
                   selection: str) -> tuple[tuple[Case, ...], dict[str, Any]]:
    """``(the cases a capture records, its case_selection record)``, among ``cases``.

    ``benchmark`` needs the model's search-set entry, generated from the tracked
    snapshot (the same ``snapshot_hash``), whose rows are all observed cases.
    The record names the rule, the search set's content and the observed cases
    left out, and enters the corpus identity.
    """
    if selection == "all":
        return tuple(cases), {"rule": "all", "definition": "every observed case"}
    if selection != "benchmark":
        raise ValueError(f"case selection {selection!r} is not one of {CASE_SELECTIONS}")
    from evograd.benchmark import search_set
    from evograd.benchmark.topdown import load_snapshot

    entry = search_set.entry(model.name)
    if entry is None:
        raise KeyError(f"{model.name} has no entry in the search set {search_set.PATH}, so it has "
                       "no benchmark rows; capture it with --cases all")
    tracked = load_snapshot(model.name).get("snapshot_hash")
    if entry["snapshot_hash"] != tracked:
        raise ValueError(f"{model.name}: the search set was generated from snapshot_hash "
                         f"{entry['snapshot_hash']}, not the tracked {tracked}; regenerate it: "
                         "scripts/adbench_selection/write_search_set.py")
    rows = search_set.evaluated(model.name)
    if not rows:
        raise ValueError(f"{model.name}: the search set has no benchmark row, so a benchmark "
                         "capture would record nothing; capture it with --cases all")
    observed = {c.case_id for c in cases}
    if set(rows) - observed:
        raise ValueError(f"{model.name}: benchmark rows {sorted(set(rows) - observed)} are not "
                         "observed cases; regenerate the search set")
    return tuple(c for c in cases if c.case_id in rows), {
        "rule": "benchmark", "definition": BENCHMARK_CASES,
        "search_set": {"schema": search_set.SCHEMA, "sha256": sha_file(search_set.PATH),
                       "snapshot_hash": entry["snapshot_hash"]},
        "not_recorded": sorted(observed - set(rows))}


def _capture_identity(model: Model, cases, selection, batch_identity, state,
                      settings) -> dict[str, Any]:
    import importlib

    import torch
    import transformers

    from evograd.benchmark.topdown import load_snapshot
    from evograd.opdecl.oracle import reference_identity

    return {"name": f"{model.name}-real-training-payloads", "suite": model.suite,
            "snapshot_hash": load_snapshot(model.name).get("snapshot_hash"),
            "case_selection": selection,
            "cases": {c.case_id: {"task": c.task, "level": c.level, "dims": dict(c.workload.dims),
                                  "dtype": c.workload.dtype, "frequency": c.frequency,
                                  "recorded": c.calls_recorded} for c in cases},
            "references": {t: reference_identity(model.reference_op(t))
                           for t in sorted({c.task for c in cases})},
            "modeling_sha256": {name: sha_file(importlib.import_module(name).__file__)
                                for name in model.modeling_modules},
            "transformers": transformers.__version__, "torch": str(torch.__version__),
            "workload": dict(model.workload_config), "checkpoint": model.workload_config.get("weights"),
            "parameter_state": state,
            "parameter_state_note": "every capture rebuilds the model at this initial pretrained state; "
                                    "the corpus samples that state, not later optimizer states",
            "data": batch_identity, "loss": objective_of(model).loss_definition,
            "execution": settings,
            "capture_implementation": {
                "module": model.live_capture.__module__,
                "sha256": sha_file(importlib.import_module(model.live_capture.__module__).__file__)}}


def capture(model: Model, out: str | Path, *, orders=ORDERS, selection: str = "benchmark",
            log=print, device: str = "cuda") -> Path:
    """Capture the corpus: one fresh initial model per data order, fidelity per order.

    Records the payloads of the cases ``selection`` picks (:func:`recorded_cases`;
    by default the model's benchmark rows). Every recorded call still binds
    exactly one observed case, and every recorded case must receive its calls
    per step in every order. Refused before anything is written unless the
    tracked snapshot was harvested on this run (:func:`require_harvested_run`;
    ``orders`` must hold :data:`HARVEST_ORDER`). The setup is the harvest's
    (:func:`prepare_run`).
    """
    import gc

    import torch

    from evograd.verification.payloads.corpus import CorpusWriter, sha256_text

    from .objectives import step_record as record, to_device

    if HARVEST_ORDER not in orders:
        raise ValueError(f"orders {tuple(orders)} omit {HARVEST_ORDER}, the order the snapshot's "
                         "harvest ran; the capture checks its run against that order")
    observed = observed_cases(model)
    cases, chosen = recorded_cases(model, observed, selection)
    log(f"recording {len(cases)} of {len(observed)} observed cases ({selection})")
    settings, workload, batches, batch_identity = prepare_run(model, orders, device=device)
    objective = objective_of(model)
    if hasattr(objective, "bind"):
        objective = objective.bind(workload)
    primary = objective.primary
    writer = None
    fidelity: dict[str, Any] = {}
    state = None
    for order in orders:
        t0 = time.time()
        batch_now = to_device(batches[order], device)
        net = build_model(workload)
        state_now = objective.parameter_state(net)
        if state is None:
            # Before anything is written: the snapshot must describe this run
            # (MegaFold is exempt, and its warm start is not hashed again here).
            if model.requires_harvested_run:
                require_harvested_run(model, run_identity(model, state_now, batch_identity))
            state, writer = state_now, CorpusWriter(out)
        if state_now != state:
            raise RuntimeError("the initial parameter state differs between captures")
        plain = record(net, *objective.capture_step(net, batch_now), primary)
        del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        net = build_model(workload)
        recorder = model.live_capture(net).install()
        try:
            loss, output = objective.capture_step(net, batch_now)
        finally:
            recorder.remove()
        instrumented = record(net, loss, output, primary)
        key = f"{primary}_digest"
        differing = sorted(n for n in plain["grads"] if plain["grads"][n] != instrumented["grads"][n])
        fidelity[str(order)] = {
            "loss_uninstrumented": plain["loss"], "loss_instrumented": instrumented["loss"],
            "loss_bitwise": plain["loss_digest"] == instrumented["loss_digest"],
            f"{primary}_bitwise": plain[key] == instrumented[key],
            "grads_compared": len(plain["grads"]), "grads_differing": differing,
            "identical": (plain["loss_digest"] == instrumented["loss_digest"]
                          and plain[key] == instrumented[key] and not differing)}
        if not fidelity[str(order)]["identical"]:
            raise RuntimeError(f"order {order}: instrumentation changed the step: {fidelity[str(order)]}")
        del loss, output
        batch = {"order": order, "sha256": batch_identity["batches"][str(order)][objective.batch_key]}
        count = record_step(writer, model.invocations(recorder, net, observed), batch,
                            case_ids={c.case_id for c in cases})
        log(f"order {order}: {count} payloads, {time.time() - t0:.1f}s, "
            f"{writer.bytes_written / 2**30:.1f} GiB written")
        del recorder, net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payloads = writer.payloads
    covered = coverage(cases, payloads)
    if not covered["complete"]:
        raise RuntimeError(f"incomplete capture: {covered['problems'][:5]} {covered['unmapped_cases']}")
    frozen = splits(payloads)
    identity = _capture_identity(model, cases, chosen, batch_identity, state, settings)
    return writer.finish(identity=identity, splits=frozen, coverage=covered,
                         extra={"capture_fidelity": fidelity,
                                "split_rules": {
                                    "calibration": f"every payload of orders {list(CALIBRATION_ORDERS)}",
                                    "acceptance": f"every payload of orders {list(ACCEPTANCE_ORDERS)}",
                                    "development": "per case, role and calibration order: first, middle "
                                                   "and last recorded layer (frozen with the corpus)"},
                                "splits_sha256": {k: sha256_text(json.dumps(sorted(v)))
                                                  for k, v in frozen.items()}})


# ── replay fidelity ──────────────────────────────────────────────────────────


def replay_fidelity(model: Model, corpus_path: str | Path, *, device: str = "cuda",
                    log=print) -> dict[str, Any]:
    """Every payload loaded back: layout as recorded, and whether the model
    reference reproduces the live output bit for bit (counted per task)."""
    import torch

    from evograd.opdecl.inputs import as_output_tuple
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.verification.common.tensors import tensor_digest
    from evograd.verification.payloads.corpus import Corpus

    configure()
    corpus = Corpus.open(corpus_path)
    rows, by_task = [], {}
    started = time.time()
    for n, record in enumerate(corpus.payloads()):
        op = model.reference_op(record["task"])
        values = corpus.load(record, device=device)
        layout_ok = all(list(values[name].stride()) == entry["stride"] and
                        list(values[name].shape) == entry["shape"]
                        for name, entry in {**record["args"], **record["cotangents"]}.items()
                        if "storage" in entry)
        with torch.no_grad():
            out = resolve_runtime_forward(op)(*[values.get(a.name, getattr(a, "default", None))
                                                for a in op.args])
        outs = dict(zip(op.output_names, as_output_tuple(op, out)))
        equal = {name: tensor_digest(outs[name]) == live["digest"]
                 for name, live in record["live_outputs"].items()}
        rows.append({"payload_id": record["payload_id"], "layout_ok": layout_ok, "outputs_bitwise": equal})
        task = by_task.setdefault(record["task"], {"payloads": 0, "layout_failures": 0,
                                                   "output_mismatches": 0})
        task["payloads"] += 1
        task["layout_failures"] += not layout_ok
        task["output_mismatches"] += not all(equal.values())
        if n % 200 == 0:
            log(f"replay {n}: {time.time() - started:.0f}s")
        del values, out, outs
    return {"corpus_id": corpus.corpus_id, "payloads": len(rows), "by_task": by_task,
            "layouts_ok": all(r["layout_ok"] for r in rows),
            "outputs_bitwise": all(all(r["outputs_bitwise"].values()) for r in rows),
            "seconds": time.time() - started, "rows": rows}
