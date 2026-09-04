"""Negative controls: does the calibrated gate still reject a wrong model?

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.evaluation.tier3.controls \
        --report results/qwen3-level4/t3-negative-controls.json

The envelope is derived from reference runs only, which is what makes it honest
and also what makes it untested: nothing in its derivation ever saw a wrong
kernel. So every fault in :mod:`.faults` is injected into an otherwise
structural provider and put through the same gate a candidate will face, on the
**holdout** seeds -- the ones that did not set the thresholds.

The number this produces is the gate's sensitivity: the smallest magnitude of
each fault kind that is rejected on every seed tested. A fault that survives is
reported as surviving. Thresholds are never loosened to accommodate a control,
and never tightened to catch one -- either would make the envelope a fit to the
faults rather than a measurement of the hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .faults import catalogue, smallest_rejected, state_catalogue
from .gate import (
    DEFAULT_ARTIFACT,
    build_references,
    check_model_correctness,
    load_policy,
)
from .sites import structural_identity_kernels
from .workload import Qwen3Workload

CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_QWEN3_CTRL_TIMEOUT", "3600"))


def run_one(fault, *, seed: int, policy, workload_config: dict[str, Any],
            workload=None, references=None) -> dict[str, Any]:
    """One fault, one seed, through the real gate."""
    workload = workload or Qwen3Workload.from_config(workload_config)
    kernels = fault.apply(workload)
    try:
        verdict = check_model_correctness(
            workload, kernels, policy=policy, data_seed=seed,
            # The trajectory is a second, slower gate; the controls exercise the
            # single-step envelope, which is the one a candidate meets first.
            check_trajectory=False, references=references,
        )
    except Exception as exc:  # a fault that crashes is also a rejection
        return {"fault": fault.to_dict(), "seed": seed, "rejected": True,
                "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "fault": fault.to_dict(),
        "seed": seed,
        "rejected": not verdict["ok"],
        # Which stage refused it matters as much as that one did: a fault the
        # purity gate catches never reaches the model, and a fault only the
        # envelopes catch says the earlier, cheaper stages are blind to it.
        "failed_at": verdict.get("failed_at"),
        "reason": verdict.get("reason"),
        "vs_eager_exceeded": len(verdict.get("vs_eager", {}).get("exceeded", [])),
        "vs_bound_exceeded": len(verdict.get("vs_bound_pair", {}).get("exceeded", [])),
        "worst_vs_eager": verdict.get("vs_eager", {}).get("worst"),
        "finite": verdict.get("finite", {}).get("ok"),
    }


def _isolated(argv: list[str]) -> dict[str, Any]:
    import tempfile

    handle, path = tempfile.mkstemp(prefix="evograd_qwen3_ctrl_", suffix=".json")
    os.close(handle)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "evograd.bench.workloads.qwen3.evaluation.tier3.controls",
             *argv, "--result-json", path],
            capture_output=True, text=True, timeout=CHILD_TIMEOUT,
        )
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {"error": f"child exited rc={process.returncode} without a result",
                    "stderr_tail": process.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": f"child exceeded {CHILD_TIMEOUT}s"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.evaluation.tier3.controls",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--magnitudes", default=None,
                        help="comma-separated fault magnitudes for the scaled "
                             "kinds; the catalogue's three by default")
    parser.add_argument("--seeds", default="11,17",
                        help="holdout seeds; these never derived a threshold")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--fault", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--magnitude", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--site", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    return parser


def _workload_config(args) -> dict[str, Any]:
    config: dict[str, Any] = {"device": args.device, "dtype": args.dtype,
                              "calibration_path": str(args.calibration)}
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    return config


def _catalogue(args):
    """The fault list, optionally narrowed to the magnitudes asked for."""
    if not getattr(args, "magnitudes", None):
        return catalogue()
    return catalogue(magnitudes=tuple(
        float(m) for m in str(args.magnitudes).split(",")))


def run_state_fault(fault, *, seed: int, policy, workload, references
                    ) -> dict[str, Any]:
    """One optimizer-state defect, judged by the same envelopes.

    These never pass through a kernel, so they are injected into a *captured*
    correct step: the gradients stay exactly right and only the update, a
    moment, or the step counter is wrong. That is the whole point -- a gate
    that pooled these with gradients would be looking at the right number and
    the wrong scale.
    """
    from .gate import _compare, _step
    from evograd.bench.tier3_gate.numerics import check_against, combined_envelope

    try:
        captured = _step(workload, structural_identity_kernels(workload.site_registry),
                         data_seed=seed,
                         learning_rate=policy.trajectory.learning_rate)
        samples = _compare(fault.apply(captured), references["eager"])
        bound = combined_envelope(policy.envelopes, policy.bound_pair_envelopes)
        verdict = check_against(bound, samples)
    except Exception as exc:
        return {"fault": fault.to_dict(), "seed": seed, "rejected": True,
                "reason": f"{type(exc).__name__}: {exc}"}
    groups = sorted({e.get("group") for e in verdict["exceeded"] if e.get("group")})
    return {
        "fault": fault.to_dict(), "seed": seed, "rejected": not verdict["ok"],
        "failed_at": "numerical_envelopes" if not verdict["ok"] else None,
        "exceeded_groups": groups[:8],
        "exceeded": len(verdict["exceeded"]),
        "reason": (verdict["exceeded"][0].get("reason")
                   or f"{len(verdict['exceeded'])} samples outside "
                      f"{len(groups)} namespaces") if not verdict["ok"] else None,
    }


def run_positive_controls(*, seed: int, policy, workload, references
                          ) -> list[dict[str, Any]]:
    """The two providers that must pass, run through the identical gate.

    A control suite that only shows rejections has not shown the gate can say
    yes. These are the same two references the calibration was taken on, so a
    failure here is a defect in the gate rather than in a provider.
    """
    from evograd.ops import OPS

    from .sites import bound_pair_identity_kernels

    out = []
    for label, kernels in (
        ("structural_identity", structural_identity_kernels(workload.site_registry)),
        ("bound_pair_identity", bound_pair_identity_kernels(
            OPS, None, workload.site_registry)),
    ):
        try:
            verdict = check_model_correctness(
                workload, kernels, policy=policy, data_seed=seed,
                check_trajectory=False, references=references,
            )
            out.append({"provider": label, "seed": seed, "admitted": verdict["ok"],
                        "failed_at": verdict.get("failed_at"),
                        "reason": verdict.get("reason")})
        except Exception as exc:
            out.append({"provider": label, "seed": seed, "admitted": False,
                        "reason": f"{type(exc).__name__}: {exc}"})
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    config = _workload_config(args)

    if args.seed is not None and args.fault is None:
        # Child mode: every fault for one seed, sharing one pair of references.
        # Rebuilding eager and bound per fault would spend the run in nn.init.
        policy = load_policy(args.calibration)
        workload = Qwen3Workload.from_config(config)
        references = build_references(workload, policy=policy, data_seed=args.seed)
        records = [
            run_one(fault, seed=args.seed, policy=policy, workload_config=config,
                    workload=workload, references=references)
            for fault in _catalogue(args)
        ]
        records += [
            run_state_fault(fault, seed=args.seed, policy=policy,
                            workload=workload, references=references)
            for fault in state_catalogue()
        ]
        records += run_positive_controls(seed=args.seed, policy=policy,
                                         workload=workload, references=references)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(records), encoding="utf-8")
        return 0

    policy = load_policy(args.calibration)
    seeds = [int(s) for s in args.seeds.split(",")]
    results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[ctrl] every fault, holdout seed={seed}", file=sys.stderr, flush=True)
        if args.no_isolate:
            workload = Qwen3Workload.from_config(config)
            references = build_references(workload, policy=policy, data_seed=seed)
            results += [
                run_one(fault, seed=seed, policy=policy, workload_config=config,
                        workload=workload, references=references)
                for fault in _catalogue(args)
            ]
            results += [
                run_state_fault(fault, seed=seed, policy=policy,
                                workload=workload, references=references)
                for fault in state_catalogue()
            ]
            results += run_positive_controls(seed=seed, policy=policy,
                                             workload=workload, references=references)
        else:
            child = ["--seed", str(seed), "--device", args.device,
                     "--dtype", args.dtype, "--calibration", str(args.calibration)]
            if args.magnitudes:
                child += ["--magnitudes", str(args.magnitudes)]
            if args.layers is not None:
                child += ["--layers", str(args.layers)]
            batch = _isolated(child)
            results += batch if isinstance(batch, list) else [batch]

    positives = [r for r in results if "provider" in r]
    clean = [r for r in results if "error" not in r and "fault" in r]
    report = {
        "schema_version": "evograd-qwen3-t3-negative-controls/2",
        "positive_controls": positives,
        "positive_controls_ok": all(r.get("admitted") for r in positives),
        "gate_stages": list(__import__(
            "evograd.bench.workloads.qwen3.evaluation.tier3.gate", fromlist=["STAGES"]).STAGES),
        "rejected_at": {
            r["fault"]["name"]: r.get("failed_at") for r in clean
            if r.get("rejected")
        },
        "calibration": str(args.calibration),
        "environment_hash": policy.environment_hash,
        "workload_id": policy.workload_id,
        "holdout_seeds": seeds,
        "policy": "no threshold was changed to accommodate any control",
        "results": results,
        "sensitivity": smallest_rejected(clean),
        "all_rejected": all(r.get("rejected") for r in clean),
        "control_kinds": sorted({r["fault"]["name"] for r in clean}),
        "survivors": [
            {"fault": r["fault"], "seed": r["seed"], "reason": r.get("reason")}
            for r in clean if not r.get("rejected")
        ],
        "errors": [r["error"] for r in results if "error" in r],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
    print(json.dumps({
        "sensitivity": report["sensitivity"],
        "all_rejected": report["all_rejected"],
        "survivors": report["survivors"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
