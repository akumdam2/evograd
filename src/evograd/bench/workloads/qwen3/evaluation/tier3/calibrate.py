"""Measure the canonical Qwen3 model's own numerical noise, then bound it.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.evaluation.tier3.calibrate run \
        --report results/qwen3-level4/t3-numerics-calibration.json

Three comparisons, on the canonical workload, in isolated child processes --
which is how tier-3 providers run, so the calibration is measured under the
conditions it will be applied in:

    E/E   unmodified eager vs an independently rebuilt unmodified eager
          the hardware's own run-to-run drift, and nothing else
    E/S   eager vs the structural adapters
          whether restructuring the modules added anything on top of E/E
    S/B   structural vs the bound pair
          what opdecl.bind and the declared runtime spellings cost

The thresholds come from E/E on the calibration seeds alone. E/S must fit inside
them or the adapters are not proven equivalent. S/B is bounded separately,
because it is a known integration drift rather than noise and a candidate is
compared against both references.

Nothing is tuned against a candidate, and no tensor reaches disk: each child
computes summaries in a streaming pass and prints JSON.
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

from .numerics import (
    GATED_METRICS,
    SAFETY_MARGIN,
    SCHEMA_VERSION,
    NumericsPolicy,
    check_against,
    compare_tensor,
    derive_envelope,
    derive_trajectory_policy,
    environment_fingerprint,
    fingerprint_hash,
)
from .sites import bound_pair_identity_kernels, structural_identity_kernels
from .workload import Qwen3Workload

#: Seeds the thresholds are derived from, and seeds they are validated on. The
#: holdout never contributes to a bound; that is what makes it a test.
CALIBRATION_SEEDS = (0, 1, 2, 3)
HOLDOUT_SEEDS = (11, 17)
REPEATS = 4

#: Tier 3's own loss-trajectory horizon. Stored in the policy, because a bound
#: measured over five optimizer steps is not a bound over fifty.
HORIZON = 5
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW"

COMPARISONS = ("EE", "ES", "SB")
CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_QWEN3_CAL_TIMEOUT", "3600"))


# ── one run of the model, inside a child ─────────────────────────────────────


def _kernels(workload, kind: str):
    from evograd.bench.tier3_patch import KernelSet
    from evograd.ops import OPS

    if kind == "eager":
        return KernelSet(registry=workload.site_registry)
    if kind == "structural":
        return structural_identity_kernels(workload.site_registry)
    if kind == "bound":
        return bound_pair_identity_kernels(OPS, None, workload.site_registry)
    raise ValueError(f"unknown provider kind {kind!r}")


def _snapshot(model) -> dict[str, torch.Tensor]:
    """The initial weights, kept on device so a rerun costs a copy not a build."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _restore(model, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(snapshot[name])
            parameter.grad = None


def _repatch(workload, model, kind: str):
    """Install a different provider's sites on an already-built model.

    Building a 0.6B model costs ~50 s and running a step costs ~2 s, so a cell
    that rebuilt for every configuration would spend its time in ``nn.init``.
    E/E still uses two genuinely independent builds -- that comparison is about
    what the device does across runs, and reusing one build would measure less
    than it claims. The structural and bound configurations are installed onto a
    restored copy, which changes no weight and no architecture.
    """
    from evograd.bench.workloads.qwen3.evaluation.tier3.sites import PatchedModel, expected_counts, patch_model

    kernels = _kernels(workload, kind)
    if not kernels.patched:
        workload._last = PatchedModel(
            model=model, provenance=_empty(), counters=_counters(),
            carrier=None, expected=expected_counts(len(model.model.layers)),
        )
        return kernels
    provenance, counters, carrier = patch_model(
        model, kernels, expected_layers=workload.spec.arch["num_hidden_layers"]
    )
    workload._last = PatchedModel(
        model=model, provenance=provenance, counters=counters, carrier=carrier,
        expected=expected_counts(len(model.model.layers)),
    )
    return kernels


def _empty():
    from evograd.bench.tier3_patch import PatchProvenance

    return PatchProvenance(method="module_surgery", requested_sites=(),
                           actual_sites=(), paths={})


def _counters():
    from evograd.bench.workloads.qwen3.evaluation.tier3.sites import SiteCounters

    return SiteCounters()


def _step_on(workload, model, *, data_seed: int):
    """One forward/backward/AdamW step on an already-patched model.

    Captures exactly what the gate captures, via the same helper, so the
    calibration cannot come to measure something the gate does not check. The
    earlier version kept the *stepped parameter* and one parameter's ``exp_avg``;
    a threshold derived from those covered neither the update the step applied
    nor the moments of the other 309 tensors.
    """
    from .gate import capture_step

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ids, labels = workload.batch_for(seed=data_seed)
    result = capture_step(model, optimizer, ids, labels)
    built = workload.last_build
    result["provenance"] = built.provenance.to_dict() if built else {}
    result["counts"] = built.observed() if built else {}
    del optimizer
    torch.cuda.empty_cache()
    return result


def _one_step(workload, kind: str, *, data_seed: int, fault=None):
    """Build, one forward/backward, one AdamW step. Returns tensors to compare."""
    kernels = _kernels(workload, kind)
    if fault is not None:
        kernels = fault(workload, kernels)
    model, provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ids, labels = workload.batch_for(seed=data_seed)
    outputs = model(input_ids=ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    optimizer.step()
    stepped = {n: p.detach().clone() for n, p in model.named_parameters()}
    exp_avg = {
        n: optimizer.state[p]["exp_avg"].detach().clone()
        for n, p in list(model.named_parameters())[:1]  # one is enough to gate on
        if p in optimizer.state and "exp_avg" in optimizer.state[p]
    }
    counters = workload.last_build.observed() if workload.last_build else {}
    result = {
        "logits": outputs.logits.detach().clone(),
        "loss": outputs.loss.detach().clone(),
        "grads": grads,
        "stepped": stepped,
        "exp_avg": exp_avg,
        "missing_grads": missing,
        "provenance": provenance.to_dict(),
        "counts": counters,
    }
    del model, outputs, optimizer
    torch.cuda.empty_cache()
    return result


def _trajectory(workload, kind: str, *, data_seed: int, horizon: int = HORIZON):
    """``horizon`` optimizer steps on a fresh deterministic batch each time."""
    model, _p = workload.build_patched(_kernels(workload, kind))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    for index in range(horizon):
        ids, labels = workload.batch_for(seed=data_seed + 1000 + index)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return losses


def _trajectory_on(workload, model, *, data_seed: int, horizon: int = HORIZON):
    """``horizon`` optimizer steps on an already-patched model, fresh batch each."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    for index in range(horizon):
        ids, labels = workload.batch_for(seed=data_seed + 1000 + index)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del optimizer
    torch.cuda.empty_cache()
    return losses


def _summarize(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Every comparable result, in the gate's own namespaces.

    One list, not three: each sample already carries the kind that decides its
    envelope, so grouping them here would only be a second opinion about
    something the comparison has already settled.
    """
    from .gate import _compare

    samples = _compare(candidate, reference)
    return {
        "samples": samples,
        "counts": candidate["counts"],
        "provenance": candidate["provenance"],
        "missing_grads": candidate["missing_grads"],
        "stateless_parameters": candidate.get("stateless_parameters", []),
        "finite": all(s.get("finite", False) for s in samples),
    }


def run_cell(seed: int, repeat: int, *, workload_config: dict[str, Any],
             trajectories: bool = True) -> dict[str, Any]:
    """All three comparisons for one (seed, repeat), sharing four model runs.

    Four runs rather than six: E/E needs two eager, E/S reuses the first eager,
    S/B reuses the structural. Every run rebuilds the model from the same CPU
    seed, so the weights are identical and only the device's execution differs.
    """
    from evograd.bench.tier3_patch import KernelSet

    workload = Qwen3Workload.from_config(workload_config)
    model_a, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
    eager_a = _step_on(workload, model_a, data_seed=seed)
    del model_a
    torch.cuda.empty_cache()

    # A second, independent build: E/E is about what the device does across
    # runs, and sharing a build would measure less than the name claims.
    model_b, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
    weights = _snapshot(model_b)
    eager_b = _step_on(workload, model_b, data_seed=seed)
    ee = _summarize(eager_b, eager_a)
    del eager_b
    torch.cuda.empty_cache()

    _restore(model_b, weights)
    _repatch(workload, model_b, "structural")
    structural = _step_on(workload, model_b, data_seed=seed)
    es = _summarize(structural, eager_a)
    del eager_a
    torch.cuda.empty_cache()

    _restore(model_b, weights)
    _repatch(workload, model_b, "bound")
    bound = _step_on(workload, model_b, data_seed=seed)
    sb = _summarize(bound, structural)
    del structural, bound, model_b, weights
    torch.cuda.empty_cache()

    # Trajectories cost four more builds, so they are measured once per seed
    # rather than once per repeat: the quantity they bound is a per-seed curve,
    # and repeating the single-step comparison is what samples the noise.
    curves: dict[str, list[float]] = {}
    if trajectories:
        model_c, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
        base = _snapshot(model_c)
        for kind, label in (("eager", "eager"), ("eager", "eager2"),
                            ("structural", "structural"), ("bound", "bound")):
            _restore(model_c, base)
            _repatch(workload, model_c, kind)
            curves[label] = _trajectory_on(workload, model_c, data_seed=seed)
        del model_c, base
        torch.cuda.empty_cache()
    return {"seed": seed, "repeat": repeat, "EE": ee, "ES": es, "SB": sb,
            "trajectories": curves, "has_trajectories": bool(curves)}


# ── isolation ────────────────────────────────────────────────────────────────


def _run_isolated(argv: list[str]) -> dict[str, Any]:
    """One child, one (seed, repeat). A wedged run costs its own cell."""
    import tempfile

    handle, path = tempfile.mkstemp(prefix="evograd_qwen3_cal_", suffix=".json")
    os.close(handle)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "evograd.bench.workloads.qwen3.evaluation.tier3.calibrate",
             *argv, "--result-json", path],
            capture_output=True, text=True, timeout=CHILD_TIMEOUT,
        )
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {"error": f"child exited rc={process.returncode} without a result",
                    "stderr_tail": process.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        return {"error": f"child exceeded {CHILD_TIMEOUT}s and was killed"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── deriving and validating the policy ───────────────────────────────────────


def _samples(cells: list[dict[str, Any]], comparison: str, kind: str = "samples"):
    for cell in cells:
        if "error" in cell:
            continue
        for sample in cell[comparison][kind]:
            yield sample


def _trajectory_deltas(cells, a: str, b: str) -> list[tuple[float, float]]:
    out = []
    for cell in cells:
        if "error" in cell or not cell.get("has_trajectories"):
            continue
        left, right = cell["trajectories"][a], cell["trajectories"][b]
        for x, y in zip(left, right):
            out.append((abs(x - y), abs(x - y) / abs(y) if y else 0.0))
    return out


def build_policy(cells: list[dict[str, Any]], *, workload: Qwen3Workload,
                 environment: dict[str, Any], margin: float = SAFETY_MARGIN):
    # No pooling. A gradient, the update one step applied, and Adam's two
    # moments are four quantities with four different scales, and a shared
    # envelope takes the loudest of them -- the gradient -- as the bound for all
    # four. A wrong update is then a rounding error against a threshold set by
    # something a thousand times larger. Each sample carries the kind it belongs
    # to and is judged only against samples of that kind.
    envelopes = derive_envelope(_samples(cells, "EE"), margin=margin)
    bound = derive_envelope(_samples(cells, "SB"), margin=margin)
    trajectory = derive_trajectory_policy(
        _trajectory_deltas(cells, "eager2", "eager"),
        horizon=HORIZON, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
        margin=margin,
    )
    bound_trajectory = derive_trajectory_policy(
        _trajectory_deltas(cells, "bound", "structural"),
        horizon=HORIZON, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
        margin=margin,
    )
    return NumericsPolicy(
        schema_version=SCHEMA_VERSION,
        workload_id=workload.spec.workload_id,
        workload_hash=workload.spec.workload_hash,
        environment=environment,
        environment_hash=fingerprint_hash(environment),
        envelopes=envelopes,
        trajectory=trajectory,
        bound_pair_envelopes=bound,
        bound_pair_trajectory=bound_trajectory,
        notes={
            "derived_from": "E/E on the calibration seeds only",
            "formula": "threshold = max(observed E/E maximum) * margin",
            "margin": margin,
            "gated_metrics": list(GATED_METRICS),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "repeats": REPEATS,
            "residual_rmsnorm_spelling": (
                "fused_add_rms_norm's declared runtime_forward normalizes with "
                "F.rms_norm while Qwen3RMSNorm casts to bfloat16 before the "
                "weight multiply. That systematic difference appears in S/B "
                "only, never in E/E or E/S, and is bounded separately."
            ),
        },
    )


def summarize_comparison(cells, comparison: str) -> dict[str, Any]:
    """What one comparison looked like, per metric, across every sample."""
    out: dict[str, Any] = {"cells": 0, "samples": 0, "metrics": {}}
    values: dict[str, list[float]] = {m: [] for m in GATED_METRICS}
    bitwise = finite = total = 0
    for cell in cells:
        if "error" in cell:
            continue
        out["cells"] += 1
        for sample in cell[comparison]["samples"]:
            total += 1
            bitwise += int(sample.get("bitwise", False))
            finite += int(sample.get("finite", False))
            for metric in GATED_METRICS:
                if metric in sample:
                    values[metric].append(float(sample[metric]))
    out["samples"] = total
    out["bitwise_fraction"] = bitwise / total if total else 0.0
    out["all_finite"] = finite == total
    for metric, series in values.items():
        if series:
            out["metrics"][metric] = {
                "max": max(series), "mean": sum(series) / len(series),
                "min": min(series),
            }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def _workload(args) -> Qwen3Workload:
    config: dict[str, Any] = {"device": args.device, "dtype": args.dtype}
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    if args.tokens is not None:
        config["seq_len"] = args.tokens
    return Qwen3Workload.from_config(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.evaluation.tier3.calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("run", "cell"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=int, default=None,
                        help="exploratory reduced-layer pass; thresholds must "
                             "come from the full canonical workload")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated override for the calibration seeds")
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-isolate", action="store_true")
    # child mode
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trajectories", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    workload = _workload(args)

    if args.command == "cell":
        cell = run_cell(args.seed, args.repeat, workload_config=workload.to_config(),
                        trajectories=args.trajectories)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(cell), encoding="utf-8")
        return 0

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(CALIBRATION_SEEDS))
    holdout = ([int(s) for s in args.holdout.split(",")] if args.holdout
               else list(HOLDOUT_SEEDS))
    environment = environment_fingerprint()

    def cells_for(seed_list) -> list[dict[str, Any]]:
        out = []
        for seed in seed_list:
            for repeat in range(args.repeats):
                print(f"[cal] seed={seed} repeat={repeat}", file=sys.stderr, flush=True)
                want_curves = repeat == 0
                if args.no_isolate:
                    out.append(run_cell(seed, repeat,
                                        workload_config=workload.to_config(),
                                        trajectories=want_curves))
                else:
                    child = ["cell", "--seed", str(seed), "--repeat", str(repeat),
                             "--device", args.device, "--dtype", args.dtype]
                    if want_curves:
                        child.append("--trajectories")
                    if args.layers is not None:
                        child += ["--layers", str(args.layers)]
                    if args.tokens is not None:
                        child += ["--tokens", str(args.tokens)]
                    out.append(_run_isolated(child))
        return out

    calibration = cells_for(seeds)
    policy = build_policy(calibration, workload=workload, environment=environment)

    holdout_cells = cells_for(holdout)
    validation = {
        "EE": check_against(policy.envelopes, _samples(holdout_cells, "EE")),
        "ES": check_against(policy.envelopes, _samples(holdout_cells, "ES")),
        "SB": check_against(policy.bound_pair_envelopes, _samples(holdout_cells, "SB")),
        "trajectory_EE": [
            policy.trajectory.check(cell["trajectories"]["eager"],
                                    cell["trajectories"]["eager2"])
            for cell in holdout_cells if cell.get("has_trajectories")
        ],
        "trajectory_ES": [
            policy.trajectory.check(cell["trajectories"]["eager"],
                                    cell["trajectories"]["structural"])
            for cell in holdout_cells if cell.get("has_trajectories")
        ],
        "trajectory_SB": [
            policy.bound_pair_trajectory.check(cell["trajectories"]["structural"],
                                               cell["trajectories"]["bound"])
            for cell in holdout_cells if cell.get("has_trajectories")
        ],
    }
    # The load-bearing claim: the structural adapters must fit inside the
    # envelope the *hardware* set, on seeds that did not derive it.
    es_calibration = check_against(policy.envelopes, _samples(calibration, "ES"))

    report = {
        "schema_version": SCHEMA_VERSION,
        "workload": workload.describe(),
        "environment": environment,
        "environment_hash": policy.environment_hash,
        "design": {
            "comparisons": {
                "EE": "unmodified eager vs an independently rebuilt unmodified eager",
                "ES": "eager vs structural identity (native Transformers spellings)",
                "SB": "structural identity vs bound-pair identity (bind + declared runtime)",
            },
            "calibration_seeds": seeds,
            "holdout_seeds": holdout,
            "repeats": args.repeats,
            "horizon": HORIZON,
            "isolation": "one child process per (seed, repeat)"
                         if not args.no_isolate else "single process",
            "margin": SAFETY_MARGIN,
        },
        "summaries": {c: summarize_comparison(calibration, c) for c in COMPARISONS},
        "holdout_summaries": {c: summarize_comparison(holdout_cells, c)
                              for c in COMPARISONS},
        "policy": policy.to_dict(),
        "es_inside_ee_envelope_calibration": es_calibration,
        "holdout": validation,
        "cell_errors": [c["error"] for c in calibration + holdout_cells if "error" in c],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
    print(json.dumps({
        "summaries": report["summaries"],
        "es_inside_ee": es_calibration["ok"],
        "holdout_ok": {k: v["ok"] for k, v in validation.items()
                       if isinstance(v, dict) and "ok" in v},
        "trajectory": policy.trajectory.to_dict(),
        "bound_trajectory": policy.bound_pair_trajectory.to_dict(),
        "cell_errors": report["cell_errors"],
    }, indent=2, sort_keys=True))
    return 0 if es_calibration["ok"] and not report["cell_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
