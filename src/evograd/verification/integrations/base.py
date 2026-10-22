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
snapshot. Nothing here edits a task or a snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ORDERS = (101, 202, 303, 404)
CALIBRATION_ORDERS = (101, 202)
ACCEPTANCE_ORDERS = (303, 404)
DETERMINISM = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "use_deterministic_algorithms": True,
               "cudnn.deterministic": True, "cudnn.benchmark": False,
               "cuda.matmul.allow_tf32": False, "float32_matmul_precision": "highest",
               "torch_num_threads": 8}
LOSS_DEFINITION = ("the HF causal-LM loss: model(input_ids, labels=labels).loss, float32 logits, "
                   "labels shifted once with -100 padding, mean over valid tokens; backward of "
                   "that scalar")


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
    gate3_steps: int = 1000
    #: Gate 3 optimizer settings beyond the protocol's constant-rate AdamW
    #: (``gate3.adamw`` keywords: ``lr``, ``warmup_steps``, ``clip_grad_norm``).
    #: Empty is the protocol default; whatever is set enters every run identity.
    gate3_optimizer: Mapping[str, Any] = field(default_factory=dict)
    #: Training batches per Gate 3 update (gradient accumulation), for a model
    #: whose batch is too small to carry a stable update but must keep its shape.
    gate3_accumulation: int = 1
    #: How a training step is run, recorded and judged
    #: (:mod:`.objectives`); ``None`` is the HF causal-LM objective.
    objective: Any = None
    #: Whether Level-2 candidates also get model-scope Gate 2 and Gate 3. A
    #: model without a well-posed whole-model comparison yet (OLMoE: routed
    #: experts) is verified by Gate 1 and operator Gate 2 only.
    model_gates: bool = True


def objective_of(model: Model):
    """The model's objective, the causal-LM one by default."""
    from .objectives import NextTokenLM

    return model.objective if model.objective is not None else NextTokenLM()


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


def observed_cases(model: Model) -> tuple[Case, ...]:
    """Every observed L1 and L2 case of the model, calls per step from the snapshot."""
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
        if "configurations" in entry:
            # A harvest that observes each L2 boundary per configuration (vision,
            # generated models): one case per configuration, like Level 1.
            workloads = get_task(task).benchmark_suites[model.suite]
            configs = entry["configurations"]
            if len(workloads) != len(configs) or any(dict(w.dims) != c["dims"]
                                                     for w, c in zip(workloads, configs)):
                raise ValueError(f"{task}: suite {model.suite} does not follow the snapshot")
            cases += [Case(f"{task}/{w.provenance.component}", task, 2, w, int(c["frequency"]),
                           tuple(c.get("module_paths") or ()))
                      for w, c in zip(workloads, configs)]
            continue
        (workload,) = get_task(task).benchmark_suites[model.suite]
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


# ── payloads ─────────────────────────────────────────────────────────────────


def record_step(writer, invocations: Iterable[dict[str, Any]], batch: dict[str, Any]) -> int:
    """Write every invocation of one recorded step as payloads; returns how many."""
    import torch

    from evograd.verification.common.tensors import tensor_digest
    from evograd.verification.payloads.corpus import cotangent_stats

    writer.new_step()
    count = 0
    for inv in invocations:
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
            "batch": dict(batch), "call": {**inv["call"], "index_in_step": count},
            "args": args, "cotangents": cots, "cotangent_stats": stats,
            "aliases": [sorted(names) for names in storages.values() if len(names) > 1],
            "live_outputs": {name: {"digest": tensor_digest(v), "shape": list(v.shape),
                                    "stride": list(v.stride()), "dtype": str(v.dtype)}
                             for name, v in inv["live_outputs"].items()},
            "notes": inv["notes"]})
        count += 1
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
    """Every case received exactly its calls per step, in every recorded order."""
    expected = {c.case_id: c.frequency for c in cases}
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


def _capture_identity(model: Model, cases, batch_identity, state, settings) -> dict[str, Any]:
    import importlib

    import torch
    import transformers

    from evograd.benchmark.topdown import load_snapshot
    from evograd.opdecl.oracle import reference_identity

    return {"name": f"{model.name}-real-training-payloads", "suite": model.suite,
            "snapshot_hash": load_snapshot(model.name).get("snapshot_hash"),
            "cases": {c.case_id: {"task": c.task, "level": c.level, "dims": dict(c.workload.dims),
                                  "dtype": c.workload.dtype, "frequency": c.frequency} for c in cases},
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


def capture(model: Model, out: str | Path, *, orders=ORDERS, log=print, device: str = "cuda") -> Path:
    """Capture the corpus: one fresh initial model per data order, fidelity per order."""
    import gc

    import torch

    from evograd.verification.payloads.corpus import CorpusWriter, sha256_text

    from .objectives import step_record as record, to_device

    settings = configure()
    cases = observed_cases(model)
    workload = model.load_workload()
    objective = objective_of(model)
    if hasattr(objective, "bind"):
        objective = objective.bind(workload)
    primary = objective.primary
    batches, batch_identity = model.train_batches(workload, orders)
    writer = CorpusWriter(out)
    fidelity: dict[str, Any] = {}
    state = None
    for order in orders:
        t0 = time.time()
        batch_now = to_device(batches[order], device)
        net = build_model(workload)
        state_now = objective.parameter_state(net)
        state = state or state_now
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
        count = record_step(writer, model.invocations(recorder, net, cases), batch)
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
    identity = _capture_identity(model, cases, batch_identity, state, settings)
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
