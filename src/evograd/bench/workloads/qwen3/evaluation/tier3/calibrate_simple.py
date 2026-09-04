"""Calibrate the simplified Tier-3 gate for one exact (workload, dtype, patch set).

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.evaluation.tier3.calibrate_simple \
        run --sites qkv_norm_rope --layers 4 --tokens 256 --out policy.json

Two comparisons, both reference-only:

    N   eager vs an independently rebuilt eager
        the device's own run-to-run drift, and nothing else
    T   eager vs the *matched* trusted replacement
        the declared pair through ``bind``, at exactly the sites the candidate
        will replace -- not at every registered site

A candidate cannot reach this module. It takes a patch set, not a program, and
the only providers it can build are eager and the trusted replacement for that
patch set.

Holdout seeds do not contribute to any threshold; they are run afterwards
against the frozen policy, and a policy whose holdout fails is not a policy.

Every cell runs in its own child process, so seeds and references cannot
accumulate host or CUDA memory across a sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from evograd.bench.tier3_gate.numerics import (
    environment_fingerprint,
    fingerprint_hash,
)
from .simple import (
    HARD_METRICS,
    SAFETY_MARGIN,
    SCHEMA_VERSION,
    PatchSet,
    derive_simple_policy,
    matched_trusted_kernels,
    measure,
)
from .workload import Qwen3Workload

CALIBRATION_SEEDS = (0, 1, 2, 3)
HOLDOUT_SEEDS = (11, 17)
REPEATS = 4
LEARNING_RATE = 1e-4
CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_QWEN3_CAL_TIMEOUT", "3600"))


def _eager_kernels(workload):
    from evograd.bench.tier3_patch import KernelSet

    return KernelSet(registry=workload.site_registry)


def _patch_set(workload, sites: tuple[str, ...]) -> PatchSet:
    from evograd.bench.tier3_patch import KernelSet, patch

    kernels = KernelSet(registry=workload.site_registry)
    for site in sites:
        kernels = patch(kernels, site, workload.site_registry.require(site).default)
    return PatchSet.of(kernels, layers=workload.spec.arch["num_hidden_layers"])


def _step(workload, kernels, *, data_seed: int) -> dict[str, Any]:
    from .gate import capture_step

    model, _provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ids, labels = workload.batch_for(seed=data_seed)
    result = capture_step(model, optimizer, ids, labels)
    built = workload.last_build
    result["counts"] = built.observed() if built else {}
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_cell(seed: int, repeat: int, *, workload_config: dict[str, Any],
             sites: tuple[str, ...]) -> dict[str, Any]:
    """Reference noise and matched trusted drift for one (seed, repeat).

    Three independent builds. E/E genuinely needs two, because it is about what
    the device does across runs; the trusted replacement gets its own so its
    patching cannot be confused with a restored one.
    """
    from evograd.ops import OPS

    workload = Qwen3Workload.from_config(workload_config)
    patch_set = _patch_set(workload, sites)

    eager_a = _step(workload, _eager_kernels(workload), data_seed=seed)
    eager_b = _step(workload, _eager_kernels(workload), data_seed=seed)
    noise = measure(eager_b, eager_a)
    del eager_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trusted_kernels = matched_trusted_kernels(dict(OPS), patch_set,
                                              workload.site_registry)
    trusted = _step(workload, trusted_kernels, data_seed=seed)
    trusted_counts = trusted.get("counts", {})
    drift = measure(trusted, eager_a)
    del trusted, eager_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed, "repeat": repeat,
        "patch_set": patch_set.to_dict(),
        "trusted_observed_counts": trusted_counts,
        "noise": {k: noise[k] for k in HARD_METRICS},
        "drift": {k: drift[k] for k in HARD_METRICS},
        "noise_full": _slim(noise),
        "drift_full": _slim(drift),
    }


def _slim(metrics: dict[str, Any]) -> dict[str, Any]:
    """Everything but the big nested structures, for the record."""
    return {k: v for k, v in metrics.items()
            if k not in ("grad_presence",) and not isinstance(v, (list,))}


# ── isolation ────────────────────────────────────────────────────────────────


def _run_isolated(argv: list[str]) -> dict[str, Any]:
    from evograd.pipelines.shared.runner import evograd_env

    with_json = Path(os.environ.get("TMPDIR", "/tmp")) / f"t3simple-{os.getpid()}-{argv[-1]}.json"
    command = [sys.executable, "-m", __spec__.name, *argv[:-1],
               "--result-json", str(with_json)]
    completed = subprocess.run(command, env=evograd_env(), text=True,
                               capture_output=True, timeout=CHILD_TIMEOUT)
    if completed.returncode != 0 or not with_json.exists():
        raise RuntimeError(
            f"calibration child failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-3000:]}"
        )
    payload = json.loads(with_json.read_text(encoding="utf-8"))
    with_json.unlink(missing_ok=True)
    return payload


def _cells(seeds, repeats, *, config_json: str, sites: tuple[str, ...],
           isolate: bool, config: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for seed in seeds:
        for repeat in range(repeats):
            if isolate:
                cell = _run_isolated([
                    "cell", "--config-json", config_json,
                    "--sites", ",".join(sites),
                    "--seed", str(seed), "--repeat", str(repeat),
                    f"{seed}-{repeat}",
                ])
            else:
                cell = run_cell(seed, repeat, workload_config=config, sites=sites)
            print(f"  seed {seed} repeat {repeat}: "
                  + " ".join(f"{k}={cell['noise'][k]:.3e}/{cell['drift'][k]:.3e}"
                             for k in HARD_METRICS), flush=True)
            cells.append(cell)
    return cells


# ── the sweep ────────────────────────────────────────────────────────────────


def calibrate(*, config: dict[str, Any], sites: tuple[str, ...],
              calibration_seeds=CALIBRATION_SEEDS, holdout_seeds=HOLDOUT_SEEDS,
              repeats: int = REPEATS, margin: float = SAFETY_MARGIN,
              isolate: bool = True) -> dict[str, Any]:
    workload = Qwen3Workload.from_config(config)
    patch_set = _patch_set(workload, sites)
    config_json = json.dumps(config, sort_keys=True)

    print(f"calibration cells for patch set {patch_set.key!r} "
          f"(noise/drift per metric):", flush=True)
    cells = _cells(calibration_seeds, repeats, config_json=config_json,
                   sites=sites, isolate=isolate, config=config)
    environment = environment_fingerprint()
    policy = derive_simple_policy(
        reference_noise=[c["noise"] for c in cells],
        trusted_drift=[c["drift"] for c in cells],
        workload_id=workload.spec.workload_id,
        workload_hash=workload.spec.workload_hash,
        dtype=str(workload.spec.dtype).replace("torch.", ""),
        environment_hash=fingerprint_hash(environment),
        patch_set=patch_set,
        margin=margin,
        notes={
            "calibration_seeds": list(calibration_seeds),
            "holdout_seeds": list(holdout_seeds),
            "repeats": repeats,
            "comparisons": {
                "N": "unmodified eager vs an independently rebuilt unmodified eager",
                "T": ("unmodified eager vs the matched trusted replacement "
                      "(declared pair through bind, at the candidate's sites only)"),
            },
            "formula": ("threshold = max(max reference noise, max matched trusted "
                        "drift, metric floor) * margin"),
            "candidate_free": True,
            "layers": workload.spec.arch["num_hidden_layers"],
            "sequence_length": config.get("seq_len"),
        },
    )

    print(f"holdout cells for patch set {patch_set.key!r}:", flush=True)
    holdout = _cells(holdout_seeds, repeats, config_json=config_json,
                     sites=sites, isolate=isolate, config=config)

    from .simple import check

    holdout_results = []
    for cell in holdout:
        for label in ("noise", "drift"):
            verdict = check(policy, {**cell[f"{label}_full"], **cell[label],
                                     "missing_grads": [], "grad_presence": {},
                                     "finite": {"ok": True}})
            holdout_results.append({
                "seed": cell["seed"], "repeat": cell["repeat"], "comparison": label,
                "ok": verdict["ok"], "reason": verdict["reason"],
                "measured": verdict["measured"], "ratios": verdict["ratios"],
            })
    holdout_ok = all(r["ok"] for r in holdout_results)

    return {
        "schema": SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "environment": environment,
        "workload_config": config,
        "sites_requested": list(sites),
        "calibration_cells": cells,
        "holdout_cells": holdout,
        "holdout_results": holdout_results,
        "holdout_ok": holdout_ok,
        "accepted": holdout_ok,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibrate_simple")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "cell"):
        p = sub.add_parser(name)
        p.add_argument("--sites", default="", help="comma separated patched sites")
        p.add_argument("--layers", type=int, default=None)
        p.add_argument("--tokens", type=int, default=None)
        p.add_argument("--batch", type=int, default=None)
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--device", default="cuda")
        p.add_argument("--config-json", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--repeat", type=int, default=0)
        p.add_argument("--repeats", type=int, default=REPEATS)
        p.add_argument("--margin", type=float, default=SAFETY_MARGIN)
        p.add_argument("--no-isolate", action="store_true")
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser


def _config(args) -> dict[str, Any]:
    if args.config_json:
        return json.loads(args.config_json)
    config: dict[str, Any] = {"model": "qwen3_0_6b", "device": args.device,
                              "dtype": args.dtype, "seed": 0, "data_seed": 0,
                              "attn_implementation": "sdpa"}
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.tokens is not None:
        config["seq_len"] = args.tokens
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    sites = tuple(s for s in args.sites.split(",") if s)
    config = _config(args)

    if args.command == "cell":
        cell = run_cell(args.seed, args.repeat, workload_config=config, sites=sites)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(cell, default=str),
                                              encoding="utf-8")
        return 0

    report = calibrate(config=config, sites=sites, repeats=args.repeats,
                       margin=args.margin, isolate=not args.no_isolate)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    thresholds = report["policy"]["thresholds"]
    print(f"patch set {report['policy']['patch_set']['key']}: "
          + " ".join(f"{k}={v:.4e}" for k, v in thresholds.items())
          + f" | holdout {'OK' if report['holdout_ok'] else 'FAILED'}")
    return 0 if report["holdout_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
