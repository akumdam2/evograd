"""Run the four-part protocol: calibrate on references, freeze, then judge.

    python -m ...tier3.protocol4_cli calibrate --out policy.json [--steps 1000]
    python -m ...tier3.protocol4_cli holdout   --policy policy.json --candidate PATH
    python -m ...tier3.protocol4_cli smoke     --steps 20 --window 10 --checkpoints 0,20

Two child job kinds, each in its own process so no run can inherit another's
allocator state or CUDA context:

    step   eager reference and one provider on the first training batch, in the
           same process (two 0.6B models fit); returns B and C as scalars
    train  one provider, the full training plan; returns the two curves

The parent computes D from the curves, derives thresholds from references only,
freezes the policy, and only then runs the holdout seeds -- compile and the
candidate alike -- against the frozen file. Sequential, one provider at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .protocol4 import (
    HARD_METRICS,
    MARGIN,
    SCHEMA_VERSION,
    SCREENING_METRICS,
    SCREENING_PLAN,
    Protocol4Policy,
    capture_first_step,
    check,
    derive_policy,
    is_screening,
    load_policy,
    step_distances,
)
from .simple import PatchSet, compiled_trusted_kernels, kernels_from_patches, parse_patch_specs
from .training import TrainingPlan, run_training, training_distances

CALIBRATION_SEEDS = (0, 1, 2)
HOLDOUT_SEEDS = (11, 17)
CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_QWEN3_P4_TIMEOUT", "7200"))

PRETRAINED = "pretrained:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
REAL_TEXT = "wikitext-2-raw:b08601e04326c79dfdd32d625aee71d232d685c3"


def workload_config(args) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": "qwen3_0_6b", "device": args.device, "dtype": args.dtype,
        "seed": 0, "data_seed": 0, "attn_implementation": "sdpa",
        "weights": PRETRAINED, "data": REAL_TEXT,
    }
    if getattr(args, "layers", None):
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    if getattr(args, "tokens", None):
        config["seq_len"] = args.tokens
    return config


def plan_from(args) -> TrainingPlan:
    return TrainingPlan(
        steps=args.steps, window=args.window,
        checkpoints=tuple(int(c) for c in args.checkpoints.split(",")),
        learning_rate=args.learning_rate,
    )


DEFAULT_SITES = ("qkv_norm_rope",)


def sites_of(args) -> tuple[str, ...]:
    text = getattr(args, "sites", None) or ",".join(DEFAULT_SITES)
    return tuple(s.strip() for s in text.split(",") if s.strip())


def patches_of(args) -> dict[str, str]:
    """The candidate provider's ``{site: compile|liger|PATH}``; legacy --candidate = QKV path."""
    patches = parse_patch_specs(getattr(args, "patch", None) or ())
    if not patches and getattr(args, "candidate", None):
        patches = {sites_of(args)[0]: args.candidate}
    return patches


def build_kernels(workload, provider: str, candidate: str | None, *,
                  sites: tuple[str, ...] = DEFAULT_SITES, patches: dict[str, str] | None = None):
    """eager | compile (torch.compile of every site in ``sites``) | candidate (``patches``)."""
    from evograd.evaluation.tier3.patch import KernelSet

    registry = workload.site_registry
    if provider == "eager":
        return KernelSet(registry=registry)
    if provider == "compile":
        return compiled_trusted_kernels(PatchSet(tuple(sites), (), {}), registry)
    if provider == "candidate":
        if not patches:
            if not candidate:
                raise ValueError("--candidate or --patch is required for the candidate provider")
            patches = {sites[0]: candidate}
        if tuple(sorted(patches)) != tuple(sorted(sites)):
            raise ValueError(f"candidate patches {sorted(patches)} but the policy's sites are "
                             f"{sorted(sites)}; B/C must be calibrated for the exact patch set")
        return kernels_from_patches(patches, registry)
    raise ValueError(f"unknown provider {provider!r}")


def local_checks(workload, kernels, *, device: str, data_seed: int) -> dict[str, Any]:
    """Part A for one provider: site preflight, purity, every live boundary, counts."""
    from . import boundary, purity
    from .gate import _boundary_reason

    preflight = workload.site_preflight(kernels, device=device)
    pure = purity.run_for(kernels, workload, device=device)
    local = boundary.validate_all_invocations(workload, kernels, data_seed=data_seed)
    ok = bool(preflight.get("ok", True)) and bool(pure.get("ok", False)) and bool(local.get("ok", False))
    return {"ok": ok,
            "site_preflight": preflight,
            "provider_purity": {"ok": bool(pure.get("ok", False))},
            "live_boundary": {k: local.get(k) for k in ("ok", "failure_count", "sites")},
            "live_boundary_reason": None if local.get("ok") else _boundary_reason(local)}


# ── child jobs ───────────────────────────────────────────────────────────────


def job_step(args) -> dict[str, Any]:
    """B and C: eager reference vs one provider, first training batch, no update."""
    from .workload import Qwen3Workload

    workload = Qwen3Workload.from_config(json.loads(args.config_json))
    ids, labels = workload.train_batches(args.data_seed).batch(0)
    reference = capture_first_step(workload, build_kernels(workload, "eager", None), ids, labels)
    kernels = build_kernels(workload, args.provider, args.candidate,
                            sites=sites_of(args), patches=patches_of(args) or None)
    provider = capture_first_step(workload, kernels, ids, labels)
    distances = step_distances(provider, reference, labels, device=args.device)
    patch_set = PatchSet.of(kernels, layers=workload.spec.arch["num_hidden_layers"])
    live = {k: v for k, v in patch_set.expected_counts.items()
            if k in patch_set.patched or k in patch_set.supporting}
    distances.update({
        "provider": args.provider, "data_seed": args.data_seed,
        "patch_set": patch_set.to_dict(),
        "kernel_origin": [s.origin for s in kernels.sources],
        "kernel_sources": [{"site": s.site, "origin": s.origin} for s in kernels.sources],
        "provenance": provider.get("provenance"),
        "observed_counts": provider["counts"],
        "expected_live_counts": live,
        "counts_ok": all(provider["counts"].get(k) == v for k, v in live.items()),
    })
    if getattr(args, "with_local", False) and kernels.patched:
        distances["local"] = local_checks(workload, kernels, device=args.device,
                                          data_seed=args.data_seed)
    return distances


def job_train(args) -> dict[str, Any]:
    """D: one provider through the whole training plan."""
    from .workload import Qwen3Workload

    workload = Qwen3Workload.from_config(json.loads(args.config_json))
    plan = plan_from(args)
    kernels = build_kernels(workload, args.provider, args.candidate,
                            sites=sites_of(args), patches=patches_of(args) or None)
    result = run_training(
        workload, kernels, plan=plan,
        train_batches=workload.train_batches(args.data_seed),
        validation_batches=workload.validation_batches(),
        validation_steps=workload.validation_batches().batches_per_epoch,
        data_seed=args.data_seed,
        evaluator_factory=workload.eager_evaluator,
        progress=lambda m: print(m, file=sys.stderr, flush=True),
    )
    result.update({"provider": args.provider,
                   "kernel_origin": [s.origin for s in kernels.sources]})
    return result


def _isolated(kind: str, *, config_json: str, provider: str, data_seed: int,
              plan: TrainingPlan, candidate: str | None, device: str,
              tag: str, sites: tuple[str, ...] = DEFAULT_SITES,
              patches: dict[str, str] | None = None, with_local: bool = False) -> dict[str, Any]:
    from evograd.pipelines.shared.runner import evograd_env

    out = Path(os.environ.get("TMPDIR", "/tmp")) / f"p4-{os.getpid()}-{tag}.json"
    command = [sys.executable, "-m", __spec__.name, kind,
               "--config-json", config_json, "--provider", provider,
               "--data-seed", str(data_seed), "--device", device,
               "--steps", str(plan.steps), "--window", str(plan.window),
               "--checkpoints", ",".join(map(str, plan.checkpoints)),
               "--learning-rate", str(plan.learning_rate),
               "--sites", ",".join(sites),
               "--result-json", str(out)]
    for site, spec in (patches or {}).items():
        command += ["--patch", f"{site}={spec}"]
    if with_local:
        command += ["--with-local"]
    if candidate:
        command += ["--candidate", candidate]
    started = time.time()
    completed = subprocess.run(command, env=evograd_env(), text=True,
                               capture_output=True, timeout=CHILD_TIMEOUT)
    if completed.returncode != 0 or not out.exists():
        raise RuntimeError(f"{kind} child for {provider} seed {data_seed} failed "
                           f"({completed.returncode}):\n{completed.stderr[-4000:]}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    payload["child_seconds"] = time.time() - started
    return payload


def measure_provider(provider: str, *, config_json: str, data_seed: int,
                     plan: TrainingPlan, candidate: str | None, device: str,
                     reference_train: dict[str, Any] | None,
                     log, sites: tuple[str, ...] = DEFAULT_SITES,
                     patches: dict[str, str] | None = None, screening: bool = False,
                     with_local: bool = False) -> dict[str, Any]:
    """B, C and D for one provider at one seed. D needs the eager curve.

    ``screening=True``: A (when ``with_local``), B and C only -- no training run.
    """
    log(f"  [{provider} seed {data_seed}] step B/C{' + local A' if with_local else ''} ...")
    step = _isolated("step", config_json=config_json, provider=provider,
                     data_seed=data_seed, plan=plan, candidate=candidate,
                     device=device, tag=f"step-{provider}-{data_seed}",
                     sites=sites, patches=patches, with_local=with_local)
    log(f"     kl={step['kl_mean']:.4e} grad={step['global_grad_rel_l2']:.4e} "
        f"counts_ok={step.get('counts_ok')} local_ok={(step.get('local') or {}).get('ok')} "
        f"({step['child_seconds']:.0f}s)")
    if screening:
        return dict(step)
    log(f"  [{provider} seed {data_seed}] train D, {plan.steps} steps ...")
    train = _isolated("train", config_json=config_json, provider=provider,
                      data_seed=data_seed, plan=plan, candidate=candidate,
                      device=device, tag=f"train-{provider}-{data_seed}",
                      sites=sites, patches=patches)
    log(f"     windows={[round(w, 5) for w in train['train_window_nll'][:4]]}... "
        f"val={ {k: round(v, 5) for k, v in train['validation_nll'].items()} } "
        f"({train['child_seconds']:.0f}s)")
    measured: dict[str, Any] = {**step, "train": train}
    if reference_train is not None:
        measured.update(training_distances(train, reference_train))
    measured["non_finite_training_steps"] = train.get("non_finite_steps", [])
    return measured


# ── parent commands ──────────────────────────────────────────────────────────


def command_calibrate(args) -> int:
    from .workload import Qwen3Workload

    from evograd.evaluation.tier3.gate.numerics import environment_fingerprint, fingerprint_hash

    config = workload_config(args)
    config_json = json.dumps(config, sort_keys=True)
    workload = Qwen3Workload.from_config(config)
    plan = plan_from(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    sites = sites_of(args)
    screening = bool(getattr(args, "screening", False))
    log(f"protocol/4 {'A/B/C screening ' if screening else ''}calibration on "
        f"{workload.spec.workload_id}, sites {list(sites)}")
    log(f"plan {SCREENING_PLAN if screening else plan.to_dict()}")
    reused = _reusable_cells(getattr(args, "reuse_cells", None), sites=sites, log=log)
    cells = []
    for seed in seeds:
        ref_train = None
        if not screening:
            log(f"seed {seed}: eager reference train ...")
            ref_train = _isolated("train", config_json=config_json, provider="eager",
                                  data_seed=seed, plan=plan, candidate=None,
                                  device=args.device, tag=f"train-eagerA-{seed}", sites=sites)
            log(f"   eager windows={[round(w, 5) for w in ref_train['train_window_nll'][:4]]}... "
                f"val={ {k: round(v, 5) for k, v in ref_train['validation_nll'].items()} } "
                f"({ref_train['child_seconds']:.0f}s)")
        repeat = reused.get(("repeated_eager", seed))
        if repeat is None:
            repeat = measure_provider("eager", config_json=config_json, data_seed=seed,
                                      plan=plan, candidate=None, device=args.device,
                                      reference_train=ref_train, log=log, sites=sites,
                                      screening=screening)
        else:
            log(f"  [eager seed {seed}] repeated-eager step reused from --reuse-cells")
        compiled = reused.get(("compile", seed))
        if compiled is None:
            compiled = measure_provider("compile", config_json=config_json, data_seed=seed,
                                        plan=plan, candidate=None, device=args.device,
                                        reference_train=ref_train, log=log, sites=sites,
                                        screening=screening)
        else:
            log(f"  [compile seed {seed}] compile step reused from --reuse-cells")
        cells.append({"seed": seed, "eager_reference_train": ref_train,
                      "repeated_eager": repeat, "compile": compiled})
        _save(out.with_suffix(".partial.json"), {"cells": cells, "plan": plan.to_dict()})

    patch_set = PatchSet.of(build_kernels(workload, "compile", None, sites=sites),
                            layers=workload.spec.arch["num_hidden_layers"])
    policy = derive_policy(
        compile_distances=[c["compile"] for c in cells],
        repeat_distances=[c["repeated_eager"] for c in cells],
        workload_id=workload.spec.workload_id, workload_hash=workload.spec.workload_hash,
        dtype=str(workload.spec.dtype).replace("torch.", ""),
        environment_hash=fingerprint_hash(environment_fingerprint()),
        patch_set=patch_set, data_identity_digest=workload.data_identity_digest(),
        training_plan=(SCREENING_PLAN if screening else plan.to_dict()), margin=MARGIN,
        metrics=(SCREENING_METRICS if screening else HARD_METRICS),
        notes={"calibration_seeds": list(seeds), "candidate_free": True,
               "screening": screening, "sites": list(sites),
               "reused_cells": sorted(f"{k[0]}@{k[1]}" for k in reused),
               "references": ["eager", "repeated eager",
                              "site-compiled " + "+".join(sites)],
               "provenance": ("local checks after Liger-Kernel / FlashAttention test "
                              "practice; trained-model comparison after Cut Cross-Entropy / "
                              "FlashMask; KL anchor and trajectory thresholds are this "
                              "project's design choices")},
    )
    _save(out, {"schema": SCHEMA_VERSION, "policy": policy.to_dict(),
                "workload_config": config, "data_identity": workload.data_identity(),
                "cells": cells, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("frozen thresholds: " + " ".join(f"{k}={v:.4e}" for k, v in policy.thresholds.items()))
    log("binding terms   : " + " ".join(f"{k}={v['binding_term']}" for k, v in policy.derivation.items()))
    log(f"wrote {out}")
    return 0


def command_holdout(args) -> int:
    """Judge compile and the candidate on seeds the calibration never saw."""
    from .workload import Qwen3Workload
    from evograd.evaluation.tier3.gate.numerics import environment_fingerprint, fingerprint_hash

    payload = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    policy = Protocol4Policy.from_dict(payload["policy"])
    config = payload["workload_config"]
    config_json = json.dumps(config, sort_keys=True)
    workload = Qwen3Workload.from_config(config)
    screening = is_screening(policy)
    # A screening policy carries SCREENING_PLAN, not a TrainingPlan; the default
    # plan below only supplies the learning rate of the optional diagnostic run.
    plan = TrainingPlan() if screening else TrainingPlan(
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in policy.training_plan.items()})
    log = lambda m: print(m, flush=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    providers = [p for p in args.providers.split(",") if p]
    sites = tuple(policy.patch_set.patched)
    patches = patches_of(args) or None
    if patches and tuple(sorted(patches)) != tuple(sorted(sites)):
        raise SystemExit(f"--patch covers {sorted(patches)} but the policy is for {sorted(sites)}")

    patch_set = PatchSet.of(build_kernels(workload, "compile", None, sites=sites),
                            layers=workload.spec.arch["num_hidden_layers"])
    policy.require_binding(
        workload_id=workload.spec.workload_id, workload_hash=workload.spec.workload_hash,
        dtype=str(workload.spec.dtype).replace("torch.", ""),
        environment_hash=fingerprint_hash(environment_fingerprint()),
        patch_set=patch_set, data_identity_digest=workload.data_identity_digest(),
        training_plan=(SCREENING_PLAN if screening else plan.to_dict()),
    )
    log(("screening " if screening else "") + "policy bound: "
        + " ".join(f"{k}={v:.4e}" for k, v in policy.thresholds.items()))

    results = []
    references: dict[int, dict[str, Any]] = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, Any] = {}
    for seed in seeds:
        ref_train = None
        if not screening:
            log(f"holdout seed {seed}: eager reference train ...")
            ref_train = _isolated("train", config_json=config_json, provider="eager",
                                  data_seed=seed, plan=plan, candidate=None,
                                  device=args.device, tag=f"hold-eager-{seed}", sites=sites)
            references[seed] = ref_train
        for provider in providers:
            measured = measure_provider(provider, config_json=config_json, data_seed=seed,
                                        plan=plan, candidate=args.candidate,
                                        device=args.device, reference_train=ref_train, log=log,
                                        sites=sites, patches=patches, screening=screening,
                                        with_local=True)
            verdict = check(policy, measured)
            local = measured.get("local") or {}
            if screening and not local.get("ok", False):
                verdict = dict(verdict, ok=False, failed_at="local_A",
                               reason=(f"part A failed: preflight={local.get('site_preflight', {}).get('ok')} "
                                       f"purity={local.get('provider_purity', {}).get('ok')} "
                                       f"live_boundary={local.get('live_boundary', {}).get('ok')} "
                                       f"{local.get('live_boundary_reason') or ''}"))
            if screening and not measured.get("counts_ok", True):
                verdict = dict(verdict, ok=False, failed_at="invocation_counts",
                               reason=f"observed {measured.get('observed_counts')} != expected "
                                      f"{measured.get('expected_live_counts')}")
            log(f"  => {provider} seed {seed}: {'PASS' if verdict['ok'] else 'FAIL'} "
                + (f"({verdict['reason']})" if not verdict["ok"] else "")
                + " ratios " + " ".join(f"{k}={v:.3f}" for k, v in verdict["ratios"].items()))
            results.append({"seed": seed, "provider": provider, "measured": measured,
                            "verdict": verdict})
            _save(out, holdout_payload(args.policy, results, references, diagnostics))
    steps = int(getattr(args, "diagnostic_steps", 0) or 0)
    if screening and steps:
        # A SHORT loss trajectory, recorded and never judged: eager and every
        # provider at the first holdout seed, validation only at the last step.
        short = TrainingPlan(steps=steps, window=steps, checkpoints=(steps,),
                             learning_rate=plan.learning_rate)
        for provider in ["eager", *providers]:
            log(f"  diagnostic trajectory: {provider}, {steps} steps at seed {seeds[0]} ...")
            diagnostics[provider] = _isolated(
                "train", config_json=config_json, provider=provider, data_seed=seeds[0],
                plan=short, candidate=args.candidate, device=args.device,
                tag=f"diag-{provider}-{seeds[0]}", sites=sites, patches=patches)
            diagnostics[provider]["label"] = "diagnostic only; not part of the verdict"
            _save(out, holdout_payload(args.policy, results, references, diagnostics))
    log(f"wrote {out}")
    return 0 if all(r["verdict"]["ok"] for r in results) else 1


def _reusable_cells(path, *, sites: tuple[str, ...], log) -> dict[tuple[str, int], dict[str, Any]]:
    """Step-only cells from an earlier calibration on the same workload.

    The repeated-eager step distance does not depend on any patch set, so it is
    reused for every seed the file has; the compile step distance is reused only
    when that calibration's patch set is exactly ``sites``. Only the single-step
    metrics are taken -- a training curve in the file is left where it is.
    """
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    same_sites = tuple(payload["policy"]["patch_set"]["patched"]) == tuple(sites)
    keep = ("kl_mean", "kl_mean_raw", "kl_roundoff_negative", "kl_std", "kl_max_position",
            "valid_positions", "global_grad_rel_l2", "grad_presence", "missing_grads",
            "finite", "logits_rel_l2_valid", "loss_provider", "loss_reference",
            "loss_abs_delta", "provider", "data_seed", "patch_set", "kernel_origin",
            "observed_counts", "child_seconds")
    reused: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in payload.get("cells", []):
        seed = int(cell["seed"])
        for kind in ("repeated_eager", "compile"):
            if kind == "compile" and not same_sites:
                continue
            row = cell.get(kind)
            if row and "kl_mean" in row and "global_grad_rel_l2" in row:
                reused[(kind, seed)] = {k: row[k] for k in keep if k in row}
                reused[(kind, seed)]["reused_from"] = str(path)
    log(f"reusing {len(reused)} step cells from {path} (compile reused: {same_sites})")
    return reused


def holdout_payload(policy_file, results: list[dict[str, Any]],
                    references: dict[int, dict[str, Any]],
                    diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a holdout run writes: the verdicts *and* the eager curves they were judged against.

    A verdict on part D is a distance to one eager training run. Without that
    run's own curve in the file, a D failure cannot be told apart from a failure
    of the reference -- which is exactly what happened at holdout seed 17 on
    2026-09-05, where only the window deltas and the eager validation NLL had
    survived. The reference curves are stored per seed, whole.
    """
    return {"schema": SCHEMA_VERSION, "policy_file": str(policy_file), "results": results,
            "eager_reference_train": {str(seed): ref for seed, ref in references.items()},
            "diagnostic_trajectory": dict(diagnostics or {})}


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protocol4")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "holdout", "step", "train"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda")
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--layers", type=int, default=None)
        p.add_argument("--tokens", type=int, default=None)
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--window", type=int, default=50)
        p.add_argument("--checkpoints", default="0,100,500,1000")
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--candidate", default=None)
        p.add_argument("--sites", default=",".join(DEFAULT_SITES),
                       help="comma-separated patch set the compile reference and the candidate cover")
        p.add_argument("--patch", action="append", default=[], metavar="SITE=compile|liger|PATH",
                       help="the candidate provider's route per site (repeatable)")
        p.add_argument("--screening", action="store_true",
                       help="A/B/C only: no training runs; the policy carries SCREENING_PLAN")
        p.add_argument("--reuse-cells", default=None,
                       help="an earlier calibration file whose step cells (repeated eager; compile "
                            "when the patch set matches) are reused instead of re-measured")
        p.add_argument("--diagnostic-steps", type=int, default=0,
                       help="holdout --screening: record a short loss trajectory (diagnostic)")
        p.add_argument("--with-local", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--out", default=None)
        p.add_argument("--seeds", default=None)
        p.add_argument("--providers", default="compile,candidate")
        p.add_argument("--policy", default=None)
        # child-only
        p.add_argument("--config-json", default=None, help=argparse.SUPPRESS)
        p.add_argument("--provider", default=None, help=argparse.SUPPRESS)
        p.add_argument("--data-seed", type=int, default=0, help=argparse.SUPPRESS)
        p.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("step", "train"):
        result = job_step(args) if args.command == "step" else job_train(args)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result, default=str), encoding="utf-8")
        return 0
    if args.seeds is None:
        args.seeds = ",".join(map(str, CALIBRATION_SEEDS if args.command == "calibrate"
                                  else HOLDOUT_SEEDS))
    if args.command == "calibrate":
        return command_calibrate(args)
    return command_holdout(args)


if __name__ == "__main__":
    raise SystemExit(main())
