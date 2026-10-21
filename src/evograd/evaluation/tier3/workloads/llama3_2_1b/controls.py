"""Negative controls: does the calibrated gate still reject a wrong model?

    PYTHONPATH=src python -m evograd.evaluation.tier3.workloads.llama3_2_1b.controls \
        --report results/llama3-level3/t3-negative-controls.json

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
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .faults import (
    catalogue,
    diagnostic_catalogue,
    observability,
    smallest_rejected,
    state_catalogue,
)
from .gate import (
    DEFAULT_ARTIFACT,
    build_references,
    check_model_correctness,
    load_policy,
)
from .sites import structural_identity_kernels
from .workload import Llama3Workload

CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_LLAMA3_CTRL_TIMEOUT", "3600"))


def run_one(fault, *, seed: int, policy, workload_config: dict[str, Any],
            workload=None, references=None) -> dict[str, Any]:
    """One fault, one seed, through the real gate."""
    try:
        workload = workload or Llama3Workload.from_config(workload_config)
        kernels = fault.apply(workload)
        verdict = check_model_correctness(
            workload, kernels, policy=policy, data_seed=seed,
            # The trajectory is a second, slower gate; the controls exercise the
            # single-step envelope, which is the one a candidate meets first.
            check_trajectory=False, references=references,
        )
    except Exception as exc:
        return _fault_error(fault, seed, f"{type(exc).__name__}: {exc}")
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


def _fault_error(fault, seed: int, reason: str) -> dict[str, Any]:
    return {"fault": fault.to_dict(), "seed": seed, "rejected": False,
            "errored": True, "error": reason, "reason": reason}


def _resource_failure(record: dict[str, Any]) -> bool:
    """Classify an errored control using specific failure signatures only."""
    if not record.get("errored"):
        return False
    reason = str(record.get("error", ""))
    return (reason.startswith(("MemoryError:", "OutOfMemoryError:",
                               "child exited rc=-9 ", "child exceeded "))
            or "CUDA out of memory" in reason)


def _release() -> None:
    """Release tensors between controls in the explicit in-process debug mode."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _isolated(argv: list[str]) -> dict[str, Any]:
    import tempfile

    handle, path = tempfile.mkstemp(prefix="evograd_llama3_ctrl_", suffix=".json")
    os.close(handle)
    try:
        process = subprocess.run(
            [sys.executable, "-m", "evograd.evaluation.tier3.workloads.llama3_2_1b.controls",
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
        prog="python -m evograd.evaluation.tier3.workloads.llama3_2_1b.controls",
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
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fault-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--positive-only", action="store_true", help=argparse.SUPPRESS)
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
    from evograd.evaluation.tier3.gate.numerics import check_against, combined_envelope

    try:
        captured = _step(workload, structural_identity_kernels(workload.site_registry),
                         data_seed=seed,
                         learning_rate=policy.trajectory.learning_rate)
        damaged = fault.apply(captured)
        # Measured before the verdict, and reported whatever the verdict is.
        # A control that changed no stored bit was not missed by the gate; it
        # never reached the model, and the two must never read alike.
        evidence = observability(captured["stored"], damaged["stored"])
        samples = _compare(damaged, references["eager"])
        bound = combined_envelope(policy.envelopes, policy.bound_pair_envelopes)
        verdict = check_against(bound, samples)
    except Exception as exc:
        return _fault_error(fault, seed, f"{type(exc).__name__}: {exc}")
    groups = sorted({e.get("group") for e in verdict["exceeded"] if e.get("group")})
    return {
        "fault": fault.to_dict(), "seed": seed, "rejected": not verdict["ok"],
        "failed_at": "numerical_envelopes" if not verdict["ok"] else None,
        "exceeded_groups": groups[:8],
        "exceeded": len(verdict["exceeded"]),
        "observability": evidence,
        "classification": classify(evidence, rejected=not verdict["ok"],
                                   scope=getattr(fault, "scope", "stored_parameters")),
        "reason": (verdict["exceeded"][0].get("reason")
                   or f"{len(verdict['exceeded'])} samples outside "
                      f"{len(groups)} namespaces") if not verdict["ok"] else None,
    }


#: What a control result means once observability is known.
UNOBSERVABLE = "unobservable: the fault changed no stored bit"
PARTLY_OBSERVABLE = "partly observable: quantization erased the fault on some roles"
DETECTED = "observable and rejected"
MISSED = "observable and NOT rejected: the gate or the envelope is at fault"
OPTIMIZER_SCOPE = "optimizer-state fault: no stored-parameter footprint by design"


def classify(evidence: dict[str, Any], *, rejected: bool,
             scope: str = "stored_parameters") -> str:
    """Separate "the gate missed it" from "there was nothing to miss".

    Only a fault that is *supposed* to move a parameter can be judged by
    whether it moved one. Corrupting Adam's second moment changes no weight and
    is not thereby invisible -- it is visible in the moment, which is its own
    envelope namespace.
    """
    if scope != "stored_parameters":
        return (f"{DETECTED} ({OPTIMIZER_SCOPE})" if rejected
                else f"{MISSED} ({OPTIMIZER_SCOPE})")
    if not evidence["observable"]:
        return UNOBSERVABLE
    if rejected:
        return DETECTED
    if evidence["roles_with_no_stored_change"]:
        return PARTLY_OBSERVABLE
    return MISSED


def run_positive_controls(*, seed: int, policy, workload, references
                          ) -> list[dict[str, Any]]:
    """The two providers that must pass, run through the identical gate.

    A control suite that only shows rejections has not shown the gate can say
    yes. These are the same two references the calibration was taken on, so a
    failure here is a defect in the gate rather than in a provider.
    """
    from evograd.benchmark import TASKS

    from .sites import bound_pair_identity_kernels

    out = []
    for label, kernels in (
        ("structural_identity", structural_identity_kernels(workload.site_registry)),
        ("bound_pair_identity", bound_pair_identity_kernels(
            TASKS, None, workload.site_registry)),
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
                        "errored": True, "error": f"{type(exc).__name__}: {exc}",
                        "reason": f"{type(exc).__name__}: {exc}"})
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    config = _workload_config(args)
    kernel_faults = _catalogue(args)
    state_faults = state_catalogue()
    diagnostic_faults = diagnostic_catalogue()
    faults = kernel_faults + state_faults + diagnostic_faults
    required_faults = len(kernel_faults) + len(state_faults)

    if args.seed is not None:
        # Each child handles one control. A fault's OOM or timeout cannot poison
        # the CUDA context used by later controls or by the positive controls.
        policy = load_policy(args.calibration)
        workload = Llama3Workload.from_config(config)
        references = build_references(workload, policy=policy, data_seed=args.seed)
        if args.positive_only:
            records = run_positive_controls(seed=args.seed, policy=policy,
                                            workload=workload, references=references)
        elif args.fault_index is not None:
            if not 0 <= args.fault_index < len(faults):
                raise ValueError(f"fault index {args.fault_index} out of range")
            fault = faults[args.fault_index]
            if args.fault_index < len(kernel_faults):
                record = run_one(fault, seed=args.seed, policy=policy,
                                 workload_config=config, workload=workload,
                                 references=references)
            else:
                record = run_state_fault(fault, seed=args.seed, policy=policy,
                                         workload=workload, references=references)
            if args.fault_index >= required_faults:
                record["required"] = False
            records = [record]
        else:
            raise ValueError("child must select --positive-only or --fault-index")
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(records), encoding="utf-8")
        return 0

    policy = load_policy(args.calibration)
    seeds = [int(s) for s in args.seeds.split(",")]
    results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[ctrl] holdout seed={seed}", file=sys.stderr, flush=True)
        if args.no_isolate:
            workload = Llama3Workload.from_config(config)
            references = build_references(workload, policy=policy, data_seed=seed)
            results += run_positive_controls(seed=seed, policy=policy,
                                             workload=workload, references=references)
            _release()
            for index, fault in enumerate(faults):
                if index < len(kernel_faults):
                    record = run_one(fault, seed=seed, policy=policy,
                                     workload_config=config, workload=workload,
                                     references=references)
                else:
                    record = run_state_fault(fault, seed=seed, policy=policy,
                                             workload=workload, references=references)
                if index >= required_faults:
                    record["required"] = False
                results.append(record)
                _release()
        else:
            child = ["--seed", str(seed), "--device", args.device,
                     "--dtype", args.dtype, "--calibration", str(args.calibration)]
            if args.magnitudes:
                child += ["--magnitudes", str(args.magnitudes)]
            if args.layers is not None:
                child += ["--layers", str(args.layers)]
            positives = _isolated([*child, "--positive-only"])
            results += positives if isinstance(positives, list) else [positives]
            for index, fault in enumerate(faults):
                batch = _isolated([*child, "--fault-index", str(index)])
                if isinstance(batch, list) and len(batch) == 1:
                    record = batch[0]
                elif isinstance(batch, dict) and "error" in batch:
                    record = _fault_error(fault, seed, batch["error"])
                    record["stderr_tail"] = batch.get("stderr_tail")
                else:
                    record = _fault_error(fault, seed, "child returned no single fault record")
                if index >= required_faults:
                    record["required"] = False
                results.append(record)

    positives = [r for r in results if "provider" in r]
    every = [r for r in results if "fault" in r]
    # Required controls must be rejected. Diagnostics are run and reported and
    # judged on their observability instead -- `wrong_update` is one, because
    # bfloat16 erases it before it reaches model state.
    required = [r for r in every if r.get("required", True)]
    clean = [r for r in required if not r.get("errored")]
    errored = [r for r in every if r.get("errored")]
    diagnostics = [r for r in every if not r.get("required", True)]
    positive_errors = [r for r in positives if r.get("errored")]
    top_level_errors = [r for r in results if "fault" not in r and "provider" not in r]
    expected_required = len(seeds) * required_faults
    expected_positives = len(seeds) * 2
    complete = (len(clean) == expected_required
                and len(positives) == expected_positives
                and not positive_errors and not top_level_errors)
    errors = [r["error"] for r in results if "error" in r]
    report = {
        "schema_version": "evograd-llama3-t3-negative-controls/3",
        "positive_controls": positives,
        "positive_controls_ok": (len(positives) == expected_positives
                                 and all(r.get("admitted") for r in positives)),
        "coverage": {
            "complete": complete,
            "required_controls": expected_required,
            "judged": len(clean),
            "errored": len([r for r in required if r.get("errored")]),
            "resource_failures": sum(_resource_failure(r) for r in required),
            "positive_controls": len(positives),
            "positive_controls_errored": len(positive_errors) + len(top_level_errors),
            "note": "Sensitivity is withheld until every required control is judged.",
        },
        "errored_controls": errored,
        "gate_stages": list(__import__(
            "evograd.evaluation.tier3.workloads.llama3_2_1b.gate", fromlist=["STAGES"]).STAGES),
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
        "sensitivity": (smallest_rejected(clean)
                        if len(clean) == expected_required and clean else None),
        "all_rejected": (len(clean) == expected_required and bool(clean)
                         and all(r.get("rejected") for r in clean)),
        "control_kinds": sorted({r.name for r in faults[:required_faults]}),
        "diagnostics": diagnostics,
        "diagnostic_kinds": sorted({r["fault"]["name"] for r in diagnostics}),
        "observability_policy": (
            "Every optimizer-state control records how many stored elements "
            "its fault actually changed. A control that changed none was not "
            "missed by the gate -- it never reached the model -- and is "
            "classified `unobservable` rather than counted as a failure to "
            "detect."
        ),
        "survivors": [
            {"fault": r["fault"], "seed": r["seed"], "reason": r.get("reason"),
             "classification": r.get("classification"),
             "observability": r.get("observability")}
            for r in clean if not r.get("rejected")
        ],
        "errors": errors,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
    print(json.dumps({
        "coverage": report["coverage"],
        "sensitivity": report["sensitivity"],
        "all_rejected": report["all_rejected"],
        "positive_controls_ok": report["positive_controls_ok"],
        "survivors": report["survivors"],
        "errors": report["errors"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
