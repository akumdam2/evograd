"""Qwen3.5-4B GPU baseline sessions: L3 feasibility, harvest, capture, smokes, baselines.

    python scripts/qwen3_5_4b/driver.py --run-dir results/experiments/qwen3_5_4b_cases/<run_id>

Every step runs in its own child process (fresh CUDA context, fresh imports --
the tracked snapshot a step writes is read by the next one's task registry) and
ends in ``<step>.DONE`` in the run directory; a step whose marker exists is
skipped, so the same command resumes on a fresh allocation. Nothing here sets a
tolerance or chooses a bar: numbers are recorded for the hand-off.

Session 1 (feasibility)
  l3_b1        canonical B1 x T2048: build, 3 full training steps (fwd + bwd +
               AdamW, eager; step 1 is warmup), peak memory and per-step time.
               Out of memory here STOPS the driver (the canonical batch is B1).
  l3_b2        one B2 x T2048 attempt, recorded only (out of memory expected and
               counted as done).
  l3_smoke     the shared Level-3 smoke report (one step, loss + backward).
  harvest      the observed whole-model step -> harvest_manifest.json and the tracked
               harvest/snapshot.json; declared dims against it; the observed suite.
  ops_smoke    every bound L1 primitive at its observed shapes and every L2 task
               at its canonical case: oracle vs runtime_forward, max errors; the
               Liger cross entropy on the 248,320-wide rows.
Session 2 (capture)
  capture      layers 14 and 15 from one canonical step -> layer14.pt, layer15.pt
  replay       standalone replay of each (bitwise against the full model), and
               every L2 site's invocation taken from the artifacts.
Session 3 (baselines, recorded only)
  t2           each L2 task: eager, torch.compile and its library line (Liger;
               FLA in its own sidecar child) on the observed case.
  l3_baselines the tier-3 model scope: eager, structural identity,
               --compile-site gated_delta_rule (20-step loss trajectory), then
               whole-model compile alone under a 20-minute limit.
  table        BASELINES.md from everything above.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = "qwen3_5_4b"
OBSERVED = "qwen3_5_4b_observed"

#: Exit code a child uses for "B1 does not fit": the driver stops on it.
RC_B1_OOM = 3

SESSION1 = ("l3_b1", "l3_b2", "l3_smoke", "harvest", "ops_smoke", "ops_smoke_l2",
            "l3_identity")
SESSION2 = ("capture", "replay")
SESSION3 = ("t2", "l3_baselines", "table")
STEPS = SESSION1 + SESSION2 + SESSION3

#: Seconds a step's child may run before it is killed and recorded as such.
TIMEOUTS = {"l3_b1": 1200, "l3_b2": 900, "l3_smoke": 900, "harvest": 1200,
            "ops_smoke": 1800, "ops_smoke_l2": 1800,
            "l3_identity": 1500, "capture": 1200, "replay": 1500,
            "t2": 5400, "l3_baselines": 7200, "table": 300}


def log(message: str) -> None:
    print(f"[qwen3_5 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _torch():
    import torch

    return torch


def sync() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_reset() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak() -> dict:
    torch = _torch()
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def free_cuda() -> None:
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── session 1 ────────────────────────────────────────────────────────────────


def _l3_steps(batch: int, steps: int = 3) -> dict:
    """Build the canonical model at this batch and run full training steps."""
    torch = _torch()
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.runner import OPTIMIZER_DEFAULTS
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

    workload = Qwen3_5Workload(batch_size=batch)
    out: dict = {"batch": batch, "seq": workload.spec.seq_len,
                 "workload_id": workload.spec.workload_id,
                 "canonical": workload.spec.is_canonical, "steps": []}
    peak_reset()
    tick = time.perf_counter()
    model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
    sync()
    out["build_s"] = time.perf_counter() - tick
    out["after_build_allocated_bytes"] = int(torch.cuda.memory_allocated())
    out["parameters"] = sum(p.numel() for p in model.parameters())
    out["runtime_report"] = workload.runtime_report(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, **OPTIMIZER_DEFAULTS)
    data = workload.batch_for(seed=0)
    try:
        for index in range(steps):
            sync()
            started = time.perf_counter()
            loss = workload.loss(model, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync()
            out["steps"].append({"step": index + 1, "ms": (time.perf_counter() - started) * 1e3,
                                 "loss": float(loss.detach()), "warmup": index == 0,
                                 **peak()})
            log(f"  B{batch} step {index + 1}: {out['steps'][-1]['ms']:.0f} ms, loss "
                f"{out['steps'][-1]['loss']:.4f}, peak "
                f"{out['steps'][-1]['peak_allocated_gib']:.1f} GiB")
        out["oom"] = False
    except torch.cuda.OutOfMemoryError as exc:
        out["oom"] = True
        out["error"] = f"OutOfMemoryError: {str(exc)[:1500]}"
        out["at_failure"] = peak()
        log(f"  B{batch}: out of memory at step {len(out['steps']) + 1}")
    out.update(peak())
    warm = [s["ms"] for s in out["steps"] if not s["warmup"]]
    out["warm_step_ms_mean"] = sum(warm) / len(warm) if warm else None
    out["tokens_per_second_warm"] = (batch * workload.spec.seq_len / (out["warm_step_ms_mean"] / 1e3)
                                     if warm else None)
    out["losses_finite"] = all(s["loss"] == s["loss"] for s in out["steps"])
    out["total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    del model, optimizer
    free_cuda()
    return out


def step_l3_b1(run_dir: Path) -> dict:
    out = _l3_steps(1)
    if out["oom"]:
        write_json(run_dir / "l3_b1.json", out)
        log("B1 x T2048 does not fit: stopping (the canonical batch must be revisited)")
        sys.exit(RC_B1_OOM)
    return out


def step_l3_b2(run_dir: Path) -> dict:
    return _l3_steps(2)


def step_l3_identity(run_dir: Path) -> dict:
    """Structural identity on the whole model: one canonical step (loss + backward,
    no optimizer) of eager, structural identity and eager again, every parameter
    gradient compared on the GPU.

    Session 1 found the default step not reproducible even eager against eager
    (losses bitwise, gradients not: the backward has nondeterministic kernels, and
    ``use_deterministic_algorithms(warn_only=True)`` alone did not change that).
    So two modes: ``default`` (the canonical backend; structural identity should
    differ from eager exactly as much as eager differs from itself) and
    ``deterministic_math_sdpa`` (SDPA pinned to the math backend, deterministic
    algorithms, a fixed cuBLAS workspace -- set by the driver for this child), in
    which eager must reproduce itself bitwise and structural identity must equal it.
    """
    torch = _torch()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.sites import structural_identity_kernels
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

    workload = Qwen3_5Workload()
    data = workload.batch_for(seed=0)

    def one(kernels, math_sdpa: bool):
        model, _ = workload.build_patched(kernels)
        if math_sdpa:
            with sdpa_kernel(SDPBackend.MATH):
                loss = workload.loss(model, data)
                loss.backward()
        else:
            loss = workload.loss(model, data)
            loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        counts = workload.last_build.observed()
        del model
        free_cuda()
        return loss.detach().clone(), grads, counts

    def compare(a, b) -> dict:
        differing, worst = [], (None, 0.0)
        for name in a[1]:
            if torch.equal(a[1][name], b[1][name]):
                continue
            differing.append(name)
            rel = float((a[1][name].double() - b[1][name].double()).norm()
                        / b[1][name].double().norm().clamp(min=1e-30))
            if rel > worst[1]:
                worst = (name, rel)
        layers = sorted({int(n.split(".")[2]) for n in differing if n.startswith("model.layers.")})
        return {"loss_bitwise": bool(torch.equal(a[0], b[0])), "loss": [float(a[0]), float(b[0])],
                "params": len(a[1]), "params_not_bitwise": len(differing),
                "layers_not_bitwise": layers, "worst_param": worst[0], "worst_rel_l2": worst[1],
                "all_bitwise": bool(torch.equal(a[0], b[0])) and not differing}

    out = {}
    for mode in ("default", "deterministic_math_sdpa"):
        math_sdpa = mode != "default"
        if math_sdpa:
            # warn_only: an op with no deterministic kernel warns (in the session log)
            # rather than ending the step.
            torch.use_deterministic_algorithms(True, warn_only=True)
        eager = one(KernelSet(registry=workload.site_registry), math_sdpa)
        identity = one(structural_identity_kernels(workload.site_registry), math_sdpa)
        out[mode] = {"identity_vs_eager": compare(identity, eager), "identity_counts": identity[2]}
        del identity
        again = one(KernelSet(registry=workload.site_registry), math_sdpa)
        out[mode]["eager_vs_eager"] = compare(again, eager)
        del eager, again
        free_cuda()
        log(f"  l3 identity ({mode}): " + json.dumps(
            {k: (v["all_bitwise"], v["params_not_bitwise"], v["worst_rel_l2"])
             for k, v in out[mode].items() if isinstance(v, dict) and "all_bitwise" in v}))
    return out


def step_l3_smoke(run_dir: Path) -> dict:
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.smoke import run_smoke

    report = run_smoke()
    report.write(run_dir / "smoke_report.json")
    if not report.ok:
        raise RuntimeError(f"L3 smoke failed: {report.failure}")
    return {"status": report.status, "result": report.result, "diagnostics": report.diagnostics,
            "dispatch": {k: v for k, v in report.effective.items() if "dispatch" in k}}


#: Section 2.1's primitive table: the roles each primitive was expected to serve.
EXPECTED_ROLES = {
    "linear_no_bias": {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                       "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                       "down_proj", "lm_head"},
    "gemma_rmsnorm": {"input_layernorm", "post_attention_layernorm", "norm", "q_norm",
                      "k_norm"},
    "gated_rmsnorm": {"norm"},
    "causal_conv1d": {"causal_conv1d_fn"},
    "l2norm": {"l2norm"},
    "gdn_gating": {"chunk_gated_delta_rule"},
    "partial_rope": {"apply_rotary_pos_emb"},
    "causal_gqa_attention": {"scaled_dot_product_attention"},
    "swiglu": {"act_fn"},
    "cross_entropy": {"fixed_cross_entropy"},
}


def step_harvest(run_dir: Path) -> dict:
    from evograd.benchmark.topdown.common.manifest import summarize, write_manifest
    from evograd.benchmark.topdown.qwen3_5_4b.harvest import harvest, snapshot
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest
    from evograd.opdecl.models import QWEN3_5_4B

    peak_reset()
    started = time.perf_counter()
    data = harvest.run_harvest()
    # Not ``harvest.json``: that name is this step's own result file.
    path = run_dir / "harvest_manifest.json"
    write_manifest(data, path)
    log(summarize(data, label="Qwen3.5 harvest"))
    snap = snapshot.extract(data)
    text = json.dumps(snap, indent=2) + "\n"
    snapshot.SNAPSHOT_PATH.write_text(text, encoding="utf-8")
    (run_dir / "snapshot.json").write_text(text, encoding="utf-8")
    loaded = snapshot.load()  # verifies the hash as written
    declared = {site: manifest.dims_for(site, QWEN3_5_4B, batch=1, seq=2048)
                for site in manifest.SITES}
    problems = snapshot.check_declared_dims(loaded, declared)
    roles = {generic: sorted({r for c in entry["configurations"] for r in c["roles"]})
             for generic, entry in loaded["level1"].items()}
    role_diff = {generic: {"unexpected": sorted(set(roles.get(generic, ())) - expected),
                           "missing": sorted(expected - set(roles.get(generic, ())))}
                 for generic, expected in EXPECTED_ROLES.items()}
    level1 = {generic: [{"roles": c["roles"], "dims": c["dims"], "frequency": c["frequency"],
                         "dtype": c["dtype"], "component": c["provenance"]["component"]}
                        for c in entry["configurations"]]
              for generic, entry in loaded["level1"].items()}
    observed = {site: loaded["tasks"][manifest.task_key(site)]["observed"]
                for site in manifest.SITES}
    result = {"manifest": str(path), "manifest_hash": data["manifest_hash"],
              "workload_id": data["workload_id"], "snapshot_hash": loaded["snapshot_hash"],
              "snapshot_path": str(snapshot.SNAPSHOT_PATH.relative_to(REPO)),
              "declared_dims_problems": problems, "counts_by_task": data["counts_by_task"],
              "observed_whole_model_counts": {s: o["whole_model_invocations"]
                                              for s, o in observed.items()},
              "observed_by_layer_type": {s: o["by_layer_type"] for s, o in observed.items()},
              "level1": level1, "roles": roles, "roles_vs_table_2_1": role_diff,
              "validation_loss": data["validation"]["loss"],
              "wall_time_s": time.perf_counter() - started, **peak()}
    if problems:
        raise RuntimeError(f"declared dims disagree with the harvest: {problems}")
    return result


def _compare(actual, expected) -> dict:
    a, b = actual.detach().float(), expected.detach().float()
    diff = (a - b).abs()
    return {"max_abs_err": float(diff.max()) if diff.numel() else 0.0,
            "ref_absmax": float(b.abs().max()) if b.numel() else 0.0,
            "rel_l2": float((a.double() - b.double()).norm()
                            / b.double().norm().clamp(min=1e-30)),
            "shape_equal": tuple(a.shape) == tuple(b.shape),
            "dtype_equal": actual.dtype == expected.dtype}


def _op_case(op, workload, device: str = "cuda") -> dict:
    """Oracle vs runtime_forward on one case: outputs and every gradient (recorded only)."""
    torch = _torch()
    from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
    from evograd.opdecl.oracle import oracle, resolve_runtime_forward

    peak_reset()
    tick = time.perf_counter()
    values = make_case_inputs(op, workload, device=device)
    ref_out, ref_grads = oracle(op, values)
    ref_out = as_output_tuple(op, ref_out)
    leaves, args = {}, []
    active = {a.name for a in op.active_args()}
    for arg in op.args:
        value = values.get(arg.name, getattr(arg, "default", None))
        if arg.name in active:
            value = value.detach().clone().requires_grad_(True)
            leaves[arg.grad_name] = value
        args.append(value)
    run_out = as_output_tuple(op, resolve_runtime_forward(op)(*args))
    douts = upstream_grad_values(op, values)
    douts = douts if isinstance(douts, tuple) else (douts,)
    grads = torch.autograd.grad(run_out, list(leaves.values()), douts, allow_unused=True)
    record = {"dims": dict(workload.dims), "dtype": workload.dtype, "results": {}}
    for out, actual, expected in zip(op.outputs, run_out, ref_out):
        entry = _compare(actual, expected)
        atol, rtol = op.tolerance_for(workload, out.name)
        entry["within_provisional"] = bool(torch.allclose(actual.float(), expected.float(),
                                                          atol=atol, rtol=rtol))
        record["results"][out.name] = entry
    for name, grad in zip(leaves, grads):
        if grad is None:
            record["results"][name] = {"error": "runtime produced no gradient"}
            continue
        entry = _compare(grad, ref_grads[name])
        atol, rtol = op.tolerance_for(workload, name)
        entry["within_provisional"] = bool(torch.allclose(grad.float(), ref_grads[name].float(),
                                                          atol=atol, rtol=rtol))
        record["results"][name] = entry
    record.update(peak())
    record["wall_time_s"] = time.perf_counter() - tick
    return record


def _liger_cross_entropy(op, workload) -> dict:
    """Liger's cross-entropy pair on the observed 248,320-wide rows, against the oracle."""
    torch = _torch()
    from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
    from evograd.opdecl.oracle import oracle

    baseline = op.performance_baselines["liger"]
    forward, backward = baseline.pair_factory()
    values = make_case_inputs(op, workload, device="cuda")
    ref_out, ref_grads = oracle(op, values)
    ref_out = as_output_tuple(op, ref_out)
    out, saved = forward(*[values[name] for name in baseline.forward_args])
    out = out if isinstance(out, tuple) else (out,)
    douts = upstream_grad_values(op, values)
    grads = backward(douts if not isinstance(douts, tuple) or len(douts) > 1 else douts[0],
                     saved if isinstance(saved, tuple) else (saved,),
                     *[values[name] for name in baseline.backward_extras])
    grads = grads if isinstance(grads, tuple) else (grads,)
    result = {"dims": dict(workload.dims), "outputs": {}, "grads": {}}
    for spec, actual, expected in zip(op.outputs, out, ref_out):
        result["outputs"][spec.name] = _compare(actual, expected)
    for (name, expected), actual in zip(ref_grads.items(), grads):
        if actual is not None:
            result["grads"][name] = _compare(actual, expected)
    result.update(peak())
    return result


def step_ops_smoke(run_dir: Path, level2: bool = False) -> dict:
    """L1 cases (and the Liger CE line) here; ``level2`` the L2 cases instead.

    Two processes, not one: cuDNN's SDPA backward caches its plan per shape and
    ignores the upstream gradient's strides (found in session 1), and the L1
    attention case (contiguous dO) and the L2 gated attention (transposed dO,
    as the model hands it) share a shape -- whichever ran second got wrong
    gradients.
    """
    from evograd.benchmark.cases import observed_workloads
    from evograd.benchmark.core.registry import TASKS
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level1.manifest import OBSERVED_BINDINGS
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest
    from evograd.opdecl.models import rederive_dims

    cases = []
    for binding in OBSERVED_BINDINGS:
        for workload in observed_workloads(WORKLOAD, binding.task, op=TASKS[binding.task]):
            cases.append(("L1", binding.task, TASKS[binding.task], workload, OBSERVED))
    if level2:
        cases = [("L2", task, TASKS[task], workload, OBSERVED)
                 for task in manifest.TASKS for workload in TASKS[task].benchmark_suites[OBSERVED]]
    results = []
    for level, name, op, workload, suite in cases:
        log(f"ops smoke {level} {name} {workload.dims}")
        try:
            record = _op_case(op, workload)
            record.update({"level": level, "task": name, "suite": suite, "ok": True,
                           "rederives": (workload.provenance is not None
                                         and workload.dims == rederive_dims(workload.provenance))})
        except Exception as exc:  # one case must not take the step down
            record = {"level": level, "task": name, "suite": suite, "dims": workload.dims,
                      "ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-3000:]}
        results.append(record)
        free_cuda()
    liger = None
    ce = [w for w in observed_workloads(WORKLOAD, "cross_entropy", op=TASKS["cross_entropy"])]
    if ce and not level2:
        try:
            liger = _liger_cross_entropy(TASKS["cross_entropy"], ce[0])
            liger["ok"] = True
        except Exception as exc:
            liger = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()[-3000:]}
    free_cuda()
    return {"cases": results, "failed": [r["task"] for r in results if not r["ok"]],
            "not_within_provisional": sorted({
                f"{r['task']}:{k}" for r in results if r["ok"]
                for k, v in r["results"].items() if not v.get("within_provisional", True)}),
            "liger_cross_entropy_248k": liger}



def step_capture(run_dir: Path) -> dict:
    from evograd.benchmark.topdown.qwen3_5_4b.layer.capture import (
        run_capture, write_captures)

    peak_reset()
    tick = time.perf_counter()
    results = run_capture(manifest_path=run_dir / "harvest_manifest.json")
    written = write_captures(results, run_dir)
    return {"written": [str(p) for p in written],
            "artifacts": {str(i): {"artifact_hash": a.payload["artifact_hash"],
                                   "content_hash": a.payload["content_hash"],
                                   "layer_kind": a.identity["layer_kind"],
                                   "bytes": (run_dir / f"layer{i}.pt").stat().st_size,
                                   "full_model_loss": m["full_model_validation"]["loss"]}
                          for i, (a, m) in results.items()},
            "wall_time_s": time.perf_counter() - tick, **peak()}


def step_replay(run_dir: Path) -> dict:
    torch = _torch()
    from evograd.benchmark import TASKS
    from evograd.benchmark.topdown.qwen3_5_4b.architecture import REPRESENTATIVE_LAYERS
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import capture_tools, manifest
    from evograd.benchmark.topdown.qwen3_5_4b.layer.artifact import load_canonical
    from evograd.evaluation.workloads.qwen3_5_4b.layer.replay import run_replay
    from evograd.opdecl import bind_shape
    from evograd.opdecl.models import QWEN3_5_4B
    from evograd.opdecl.oracle import resolve_runtime_forward

    out = {"replay": {}, "sites": {}}
    for kind, index in sorted(REPRESENTATIVE_LAYERS.items(), key=lambda kv: kv[1]):
        path = run_dir / f"layer{index}.pt"
        artifact = load_canonical(path, layer_index=index)
        report = run_replay(artifact, device="cuda", noise_repeats=2)
        write_json(run_dir / f"replay_layer{index}.json", report)
        comparisons = report["comparisons"]
        out["replay"][str(index)] = {
            "status": report["status"],
            "output_bitwise": comparisons["output"].get("bitwise_identical"),
            "grad_input_bitwise": comparisons["grad_input"].get("bitwise_identical"),
            "param_grads_bitwise": all(v.get("bitwise_identical")
                                       for v in comparisons["param_grads"].values()),
            "param_grads_not_bitwise": sorted(k for k, v in comparisons["param_grads"].items()
                                              if not v.get("bitwise_identical")),
            "max_param_grad_rel_err": max((v.get("max_rel_err_vs_scale") or 0.0)
                                          for v in comparisons["param_grads"].values()),
        }
        del artifact, report
        free_cuda()
        source = capture_tools.captured_source(path, device="cuda")
        for site in manifest.SITES_BY_BLOCK[kind]:
            op = TASKS[manifest.task_key(site)]
            entry = {"layer": index}
            try:
                call = manifest.captures()[site](source)
                dims = manifest.dims_for(site, QWEN3_5_4B, batch=1, seq=2048)
                entry["shapes_ok"] = all(
                    tuple(call.inputs[a.name].shape) == bind_shape(a.shape, dims)
                    for a in op.args if getattr(a, "shape", None) is not None)
                produced = resolve_runtime_forward(op)(
                    *[call.inputs.get(a.name, getattr(a, "default", None)) for a in op.args])
                produced = produced if isinstance(produced, tuple) else (produced,)
                entry["production_bitwise"] = {
                    name: bool(torch.equal(value, call.outputs[name]))
                    for name, value in zip(op.output_names, produced)}
                entry["ok"] = True
                del call, produced
            except Exception as exc:
                entry.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                             traceback=traceback.format_exc()[-2000:])
            out["sites"][f"{site}@{index}"] = entry
            free_cuda()
        del source
        free_cuda()
    return out


# ── session 3 ────────────────────────────────────────────────────────────────


def _run(cmd: list[str], log_path: Path, timeout: int, env=None) -> dict:
    tick = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"killed after {timeout}s\n{exc.stdout or ''}\n{exc.stderr or ''}")
        rc = "timeout"
    wall = time.perf_counter() - tick
    log(f"  rc={rc} in {wall:.0f}s ({log_path.name})")
    return {"rc": rc, "wall_s": wall, "log": str(log_path)}


def step_t2(run_dir: Path) -> dict:
    from evograd.benchmark import TASKS
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest

    out_dir = run_dir / "t2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in manifest.TASKS:
        op = TASKS[task]
        baseline = next(iter(sorted(op.performance_baselines)), "none")
        report = out_dir / f"{task}.json"
        if (out_dir / f"{task}.DONE").exists():
            results[task] = json.loads((out_dir / f"{task}.DONE").read_text() or "{}")
            continue
        log(f"t2 {task} (eager, torch.compile, baseline {baseline})")
        cmd = [sys.executable, "-m", "evograd.evaluation.tier2.cli", "--op", task, "--suite",
               OBSERVED, "--baseline", baseline, "--out", str(report)]
        entry = _run(cmd, out_dir / f"{task}.log", timeout=2400)
        entry["baseline"] = baseline
        entry["report"] = str(report) if report.exists() else None
        # A provider the gate refused at this shape
        # is not timed; a second run with the gate off gives it a number, filed
        # apart and labelled diagnostic -- never a pass.
        refused = []
        if report.exists():
            for case in json.loads(report.read_text()).get("cases", []):
                refused += [name for name, row in case["providers"].items()
                            if not row.get("ok") and "correctness" in str(row.get("error"))]
        if refused:
            diagnostic = out_dir / f"{task}.nocheck.json"
            log(f"t2 {task}: gate refused {refused}; diagnostic run with --no-check")
            entry["gate_refused"] = refused
            entry["diagnostic"] = _run([*cmd[:-2], "--no-check", "--out", str(diagnostic)],
                                       out_dir / f"{task}.nocheck.log", timeout=2400)
            entry["diagnostic_report"] = str(diagnostic) if diagnostic.exists() else None
        results[task] = entry
        (out_dir / f"{task}.DONE").write_text(json.dumps(entry))
    return results



def step_l3_baselines(run_dir: Path) -> dict:
    out_dir = run_dir / "l3"
    out_dir.mkdir(parents=True, exist_ok=True)
    cli = [sys.executable, "-m", "evograd.evaluation.tier3.cli", "--model", WORKLOAD,
           "--baseline", "none", "--loss-steps", "20"]
    runs = {
        "main": ["--structural-identity", "--compile-site", "gated_delta_rule",
                 "--timeout", "2400"],
        # Alone, so a compile that exceeds 20 minutes or fails costs only itself.
        "whole_model_compile": ["--whole-model-compile", "--timeout", "1200"],
    }
    results = {}
    for name, extra in runs.items():
        report = out_dir / f"{name}.json"
        marker = out_dir / f"{name}.DONE"
        if marker.exists():
            results[name] = json.loads(marker.read_text() or "{}")
            continue
        log(f"l3 baselines: {name}")
        entry = _run([*cli, *extra, "--out", str(report)], out_dir / f"{name}.log",
                     timeout=7200 if name == "main" else 3000)
        entry["report"] = str(report) if report.exists() else None
        results[name] = entry
        marker.write_text(json.dumps(entry))
    return results


def step_table(run_dir: Path) -> dict:
    sys.path.insert(0, str(REPO / "scripts" / "qwen3_5_4b"))
    from baseline_table import build_table  # noqa: E402  (scripts directory)

    text = build_table(run_dir)
    (run_dir / "BASELINES.md").write_text(text, encoding="utf-8")
    return {"table": str(run_dir / "BASELINES.md"), "lines": text.count("\n")}


FUNCTIONS = {"l3_b1": step_l3_b1, "l3_b2": step_l3_b2, "l3_smoke": step_l3_smoke,
             "harvest": step_harvest, "ops_smoke": step_ops_smoke,
             "ops_smoke_l2": lambda run_dir: step_ops_smoke(run_dir, level2=True),
             "l3_identity": step_l3_identity, "capture": step_capture, "replay": step_replay,
             "t2": step_t2, "l3_baselines": step_l3_baselines,
             "table": step_table}


def run_child(run_dir: Path, name: str) -> int:
    import evograd

    log(f"{name}: child running evograd from {evograd.__file__}")
    tick = time.perf_counter()
    try:
        result = FUNCTIONS[name](run_dir)
    except SystemExit:
        raise
    except Exception as exc:
        write_json(run_dir / f"{name}.failed.json",
                   {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        log(f"{name}: FAILED {type(exc).__name__}: {exc}")
        return 1
    result = {"wall_time_s_step": time.perf_counter() - tick, **(result or {})}
    write_json(run_dir / f"{name}.json", result)
    (run_dir / f"{name}.DONE").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {time.perf_counter() - tick:.1f}s\n",
        encoding="utf-8")
    log(f"{name}: done in {time.perf_counter() - tick:.1f}s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", choices=STEPS, default=None)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), default=None,
                        help="stop after this session's steps")
    parser.add_argument("--child", choices=STEPS, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.child:
        return run_child(run_dir, args.child)
    torch = _torch()
    if not torch.cuda.is_available():
        log("no CUDA device visible; this driver runs on a GPU node")
        return 2
    import evograd

    log(f"evograd from {evograd.__file__}; device {torch.cuda.get_device_name()}")
    if not str(evograd.__file__).startswith(str(REPO / "src")):
        log("evograd is not imported from this worktree; export PYTHONPATH first")
        return 2
    last = {1: SESSION1, 2: SESSION1 + SESSION2, 3: STEPS}.get(args.session, STEPS)
    env = {k: v for k, v in os.environ.items() if k != "EVOGRAD_SIDECARS"}
    for name in STEPS:
        if name not in last or (args.only and name not in args.only):
            continue
        if (run_dir / f"{name}.DONE").exists():
            log(f"{name}: already done; skipping")
            continue
        log(f"{name}: start (child, timeout {TIMEOUTS[name]}s)")
        child_env = dict(env)
        if name == "l3_identity":
            # Deterministic cuBLAS for this child only: it must be set before
            # CUDA initializes, and the timed steps must not inherit it.
            child_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            proc = subprocess.run([sys.executable, __file__, "--run-dir", str(run_dir),
                                   "--child", name], env=child_env, timeout=TIMEOUTS[name])
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
            write_json(run_dir / f"{name}.failed.json", {"error": f"killed after {TIMEOUTS[name]}s"})
        if rc == RC_B1_OOM:
            log("STOP: B1 x T2048 is out of memory; see l3_b1.json")
            return RC_B1_OOM
        if rc != 0:
            log(f"{name}: child rc={rc}; continuing with the next step")
    log("driver finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
