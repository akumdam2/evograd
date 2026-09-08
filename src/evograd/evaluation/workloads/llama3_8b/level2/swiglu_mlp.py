"""Judging an implementation of ``llama3_swiglu_mlp`` against the captured boundary.

The case itself belongs to :mod:`evograd.benchmark.topdown.llama3_8b.levels.level2.swiglu_mlp.capture`.
What is here is the judgment: reference-versus-production comparison,
the tolerance each result is held to, the repeated-noise measurement
that tolerance is calibrated from, and the report and CLI that carry
the verdict.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from evograd.benchmark.topdown.llama3_8b.levels.level3.artifact import ArtifactError, load_canonical
from evograd.evaluation.workloads.llama3_8b.level3.replay import declared_gate, required_tolerance
from evograd.benchmark.topdown.llama3_8b.harvest.snapshot import load as load_snapshot
from evograd.benchmark.topdown.llama3_8b.levels.level2.swiglu_mlp.capture import (
    MlpExtractionError,
    TASK_NAME,
    check_provenance,
    derive_mlp_invocation,
)

REPORT_SCHEMA = "evograd-llama3-mlp-verify/1"

# --------------------------------------------------------------------------
# verification of the Level-2 declaration against the capture
# --------------------------------------------------------------------------


#: The declaration's suite for *this* workload. Three of the four Level-2
#: declarations are shared with Qwen3, so ``op.benchmark[0]`` is whichever
#: architecture was harvested first -- Qwen3's shape, not Llama's. The observed
#: suite is the one this workload is declared to run at.
OBSERVED_SUITE = "llama_3_8b_observed"


def _declared_case(op):
    """The single ``llama_3_8b_observed`` workload, or a problem describing why not."""
    cases = op.benchmark_workloads(OBSERVED_SUITE)
    if len(cases) != 1:
        return None, (
            f"expected one {OBSERVED_SUITE} case on {op.name}, found {len(cases)}; "
            "the declaration carries no observed suite for this workload"
        )
    return cases[0], None


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?

    Pure data: no tensors, no GPU. The declaration derives its benchmark dims
    from the snapshot at import time, so this can only fail if one of them was
    edited by hand -- which is exactly what it is here to catch.
    """
    from evograd.benchmark import get_task

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    declared, missing = _declared_case(get_task(TASK_NAME))
    if declared is None:
        return [missing]
    problems: list[str] = []
    batch, seq, hidden = harvest["input_shapes"][0]["shape"]
    for dim, expected in (
        ("B", batch),
        ("T", seq),
        ("H", hidden),
        ("I", harvest["attrs"]["intermediate_size"]),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(
                f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}"
            )
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(
            f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}"
        )
    return problems


def _reference_pass(forward, payload: dict[str, Any], device: str):
    x = payload["input"].to(device).detach().clone().requires_grad_(True)
    weights = {
        name: payload["weights"][name].to(device).detach().clone().requires_grad_(True)
        for name in ("gate_weight", "up_weight", "down_weight")
    }
    out = forward(x, weights["gate_weight"], weights["up_weight"], weights["down_weight"])
    out.backward(payload["grad_output"].to(device))
    return (
        out.detach().clone(),
        x.grad.detach().clone(),
        {name: tensor.grad.detach().clone() for name, tensor in weights.items()},
    )


def _gate_reason(record: dict[str, Any]) -> str:
    if "required_t" in record:
        return (
            f"required_t={record['required_t']:.3e} > declared base "
            f"{record['declared_base']:.3e} (atol={record['atol']}, rtol={record['rtol']})"
        )
    return (
        f"max_rel_err_vs_scale={record.get('max_rel_err_vs_scale')} > "
        f"{record.get('tolerance')}"
    )


def run_verify(
    payload: dict[str, Any],
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown.llama3_8b.levels.level2.swiglu_mlp.reference import (
        llama3_swiglu_mlp_forward_hf,
        llama3_swiglu_mlp_forward_ref,
    )

    from evograd.evaluation.workloads.llama3_8b.level3.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
        declared_gate,
    )
    from evograd.benchmark.topdown.llama3_8b.levels.level3.prepare import (
        validate_noise_repeats,
    )

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise MlpExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)

    op = get_task(TASK_NAME)
    _case, _missing = _declared_case(op)
    if _case is None:
        raise MlpExtractionError(_missing)
    declared = _case.dims
    # Two independent questions, deliberately not merged: does this artifact
    # match the snapshot it claims to come from (``check_provenance``), and does
    # the *declaration* still match the canonical snapshot. The second is about
    # the tracked snapshot whatever artifact is being verified, so a shrunken
    # debug capture does not make the declaration look wrong.
    shape_problems = declaration_problems()
    weight_shapes = {
        "gate_weight": list(payload["weights"]["gate_weight"].shape),
        "up_weight": list(payload["weights"]["up_weight"].shape),
        "down_weight": list(payload["weights"]["down_weight"].shape),
    }
    hidden = payload["arch"]["hidden_size"]
    intermediate = payload["arch"]["intermediate_size"]
    for name, expected in (
        ("gate_weight", [intermediate, hidden]),
        ("up_weight", [intermediate, hidden]),
        ("down_weight", [hidden, intermediate]),
    ):
        if weight_shapes[name] != expected:
            shape_problems.append(
                f"captured {name} {weight_shapes[name]} != {expected} implied by the "
                "captured hidden/intermediate widths"
            )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out, grad_x, weight_grads = _reference_pass(llama3_swiglu_mlp_forward_ref, payload, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    # The operator's own declared tolerance, per result, applied exactly as the
    # benchmark harness applies it -- `allclose(atol, rtol)`, not a
    # scale-normalized proxy. Taken from the declaration rather than chosen
    # here, so it cannot be tuned to fit this result.
    # This workload's observed case, not ``benchmark[0]`` -- see attention.py.
    case, missing = _declared_case(op)
    if case is None:
        raise MlpExtractionError(missing)
    declared_tol = {
        # Named, so ``ReductionScaledAtol`` can find it; an unnamed call reaches
        # the hook as ``result_name=None`` and silently keeps the base.
        "output": op.tolerance_for(case, "out"),
        "grad_input": op.tolerance_for(case, "dx"),
        **{
            name: op.tolerance_for(case, f"d{name}")
            for name in ("gate_weight", "up_weight", "down_weight")
        },
    }
    base = op.tolerances[case.dtype][0]

    comparisons = {
        "output": declared_gate(out, payload["output"].to(device), declared_tol["output"], base),
        "grad_input": declared_gate(
            grad_x, payload["grad_input"].to(device), declared_tol["grad_input"], base
        ),
        "weight_grads": {
            name: declared_gate(
                weight_grads[name],
                payload["weight_grads"][name].to(device),
                declared_tol[name],
                base,
            )
            for name in sorted(weight_grads)
        },
    }

    # The BF16 spelling Transformers actually ran. Same computation, so it gets
    # the tight replay tolerances -- this is the wiring check, and it is expected
    # to be bitwise identical.
    hf_out, hf_grad_x, hf_weight_grads = _reference_pass(
        llama3_swiglu_mlp_forward_hf, payload, device
    )
    hf_comparisons = {
        "output": compare_tensors(hf_out, payload["output"].to(device), FORWARD_TOL),
        "grad_input": compare_tensors(hf_grad_x, payload["grad_input"].to(device), GRADIENT_TOL),
        "weight_grads": {
            name: compare_tensors(
                hf_weight_grads[name], payload["weight_grads"][name].to(device), GRADIENT_TOL
            )
            for name in sorted(hf_weight_grads)
        },
    }

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "reference compared against itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [
            _reference_pass(llama3_swiglu_mlp_forward_ref, payload, device)
            for _ in range(noise_repeats)
        ]
        noise["output"] = _max_noise(
            _noise(passes[i][0], passes[0][0]) for i in range(1, noise_repeats)
        )
        noise["grad_input"] = _max_noise(
            _noise(passes[i][1], passes[0][1]) for i in range(1, noise_repeats)
        )
        noise["weight_grads"] = {
            name: _max_noise(
                _noise(passes[i][2][name], passes[0][2][name]) for i in range(1, noise_repeats)
            )
            for name in sorted(weight_grads)
        }

    failures = [f"provenance: {p}" for p in provenance_problems]
    failures += [f"shape: {p}" for p in shape_problems]
    for label, group in (("declared reference", comparisons), ("BF16 spelling", hf_comparisons)):
        for name in ("output", "grad_input"):
            record = group[name]
            if not record["within_tolerance"]:
                failures.append(f"{label} {name}: {_gate_reason(record)}")
        for name, record in group["weight_grads"].items():
            if not record["within_tolerance"]:
                failures.append(f"{label} {name} gradient: {_gate_reason(record)}")

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": TASK_NAME,
        "identity": payload["identity"],
        "provenance_chain": payload["provenance_chain"],
        "provenance_validated": not provenance_problems,
        "snapshot_hash": load_snapshot(snapshot_path)["snapshot_hash"],
        "declared_dims": declared,
        "tolerances": {
            "metric": "max|a-b| / max|b| (reference scale)",
            "declared_reference": {
                "values": declared_tol,
                "source": (
                    f"the {TASK_NAME} declaration's bfloat16 tolerance and its "
                    "per-gradient multipliers"
                ),
                "why": (
                    "the declared contract accumulates the gate/up product in "
                    "float32 and LlamaMLP does not, so these are deliberately "
                    "not the same computation; the question is whether the "
                    "reference is a valid answer for the operator, which is what "
                    "this tolerance was declared to answer"
                ),
            },
            "hf_spelling": {
                "forward": FORWARD_TOL,
                "gradient": GRADIENT_TOL,
                "why": (
                    "the same computation Transformers ran, so it is held to the "
                    "Level-3 replay tolerances: one BF16 unit roundoff forward, "
                    "one epsilon on gradients. This is the wiring check."
                ),
            },
        },
        "comparisons": comparisons,
        "hf_spelling_comparisons": hf_comparisons,
        "noise_floor": noise,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu_name": (
                torch.cuda.get_device_properties(torch.cuda.current_device()).name
                if device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
        "diagnostics": {
            "note": "diagnostic only -- one unwarmed pass, not a benchmark result",
            "forward_backward_wall_time_s": elapsed,
            "peak_allocated_bytes": peak,
        },
    }


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{value:.3e}"


def summarize_verify(report: dict[str, Any]) -> str:
    comparisons = report["comparisons"]
    lines = [
        f"[{report['status'].upper()}] {report['task']} against the captured LlamaMLP",
        f"  provenance validated: {report['provenance_validated']}  "
        f"snapshot {report['snapshot_hash'][:16]}...",
    ]
    for link in report["provenance_chain"]:
        lines.append(f"    -> {link}")
    lines += [
        "",
        f"  declared dims {report['declared_dims']}",
        "  declared reference (float32-accumulated), against the operator's "
        "declared BF16 tolerance:",
        f"  output      rel {_fmt(comparisons['output']['max_rel_err_vs_scale'])}  "
        f"required_t {_fmt(comparisons['output'].get('required_t'))} <= base "
        f"{comparisons['output'].get('declared_base')}",
        f"  grad x      rel {_fmt(comparisons['grad_input']['max_rel_err_vs_scale'])}  "
        f"required_t {_fmt(comparisons['grad_input'].get('required_t'))}",
    ]
    for name, record in comparisons["weight_grads"].items():
        lines.append(
            f"  grad {name:<12} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"required_t {_fmt(record.get('required_t'))}"
        )
    hf = report["hf_spelling_comparisons"]
    lines += [
        "",
        "  the BF16 spelling Transformers ran -- same computation, tight tolerance:",
        f"    output rel {_fmt(hf['output']['max_rel_err_vs_scale'])}  "
        f"bitwise {hf['output']['bitwise_identical']}",
        f"    grad x rel {_fmt(hf['grad_input']['max_rel_err_vs_scale'])}  "
        f"bitwise {hf['grad_input']['bitwise_identical']}",
    ]
    for name, record in hf["weight_grads"].items():
        lines.append(
            f"    grad {name:<12} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"bitwise {record['bitwise_identical']}"
        )
    noise = report["noise_floor"]
    if noise.get("measured"):
        worst = max(
            (v for v in noise["weight_grads"].values() if v is not None), default=0.0
        )
        lines += [
            "",
            f"  measured noise floor ({noise['repeats']} reference runs): "
            f"output {_fmt(noise['output'])}  grad x {_fmt(noise['grad_input'])}  "
            f"weight grads {_fmt(worst)}",
        ]
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# tolerance calibration
# --------------------------------------------------------------------------

#: Results the gate covers: the output and every gradient, named as the
#: declaration names them.
RESULT_NAMES = ("out", "dx", "dgate_weight", "dup_weight", "ddown_weight")


def _pair_pass(forward, x, gate_weight, up_weight, down_weight, dout):
    """Forward and backward through one spelling, returning every result."""
    leaves = {
        "x": x.detach().clone().requires_grad_(True),
        "gate_weight": gate_weight.detach().clone().requires_grad_(True),
        "up_weight": up_weight.detach().clone().requires_grad_(True),
        "down_weight": down_weight.detach().clone().requires_grad_(True),
    }
    out = forward(**leaves)
    out.backward(dout)
    return {
        "out": out.detach().clone(),
        "dx": leaves["x"].grad.detach().clone(),
        "dgate_weight": leaves["gate_weight"].grad.detach().clone(),
        "dup_weight": leaves["up_weight"].grad.detach().clone(),
        "ddown_weight": leaves["down_weight"].grad.detach().clone(),
    }


def _calibration_case(
    label: str,
    dtype: str,
    tensors: dict[str, Any],
    multipliers: dict[str, tuple[float, float]],
    repeats: int = 3,
):
    from evograd.benchmark.topdown.llama3_8b.levels.level2.swiglu_mlp.reference import (
        llama3_swiglu_mlp_forward_hf,
        llama3_swiglu_mlp_forward_ref,
    )

    reference = _pair_pass(llama3_swiglu_mlp_forward_ref, **tensors)
    production = _pair_pass(llama3_swiglu_mlp_forward_hf, **tensors)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    # Run-to-run noise of the production spelling, so the recommendation is not
    # calibrated against a number that moves on its own.
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(llama3_swiglu_mlp_forward_hf, **tensors)
        for name in RESULT_NAMES:
            noise[name] = max(
                noise[name],
                required_tolerance(
                    again[name], production[name], multipliers.get(name, (1.0, 1.0))
                )["required_t"],
            )
    return {
        "label": label,
        "dtype": dtype,
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "results": results,
        "production_noise_required_t": noise,
        "worst_required_t": max(r["required_t"] for r in results.values()),
    }


def run_calibration(
    source: Path,
    *,
    device: str = "cuda",
    skip_canonical: bool = False,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Measure what a correct BF16 implementation actually needs.

    Two populations, because they answer different halves of the question: the
    declared correctness workloads are what every test run gates on, and the
    canonical [2, 2048, 4096] invocation is the one the benchmark times and the
    only one at production width -- a tolerance calibrated on 32-wide cases
    would say nothing about a 14336-long contraction.
    """
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.benchmark import get_task

    op = get_task(TASK_NAME)
    multipliers = {
        name: tuple(op.tolerance_multipliers.get(name, (1.0, 1.0))) for name in RESULT_NAMES
    }
    cases: list[dict[str, Any]] = []

    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            _calibration_case(
                f"correctness {workload.dims}",
                workload.dtype,
                {
                    "x": values["x"],
                    "gate_weight": values["gate_weight"],
                    "up_weight": values["up_weight"],
                    "down_weight": values["down_weight"],
                    "dout": values["dout"],
                },
                multipliers,
            )
        )

    canonical = None
    if not skip_canonical:
        payload, _ = derive_mlp_invocation(source, device=device, snapshot_path=snapshot_path)
        canonical = _calibration_case(
            "canonical layer-16 invocation",
            payload["arch"]["dtype"],
            {
                "x": payload["input"].to(device),
                "gate_weight": payload["weights"]["gate_weight"].to(device),
                "up_weight": payload["weights"]["up_weight"].to(device),
                "down_weight": payload["weights"]["down_weight"].to(device),
                "dout": payload["grad_output"].to(device),
            },
            multipliers,
        )
        cases.append(canonical)

    by_result = {
        name: max(case["results"][name]["required_t"] for case in cases)
        for name in RESULT_NAMES
    }
    bf16_cases = [case for case in cases if case["dtype"] == "bfloat16"]
    bf16_by_result = {
        name: max((case["results"][name]["required_t"] for case in bf16_cases), default=0.0)
        for name in RESULT_NAMES
    }
    fp32_cases = [case for case in cases if case["dtype"] == "float32"]
    fp32_by_result = {
        name: max((case["results"][name]["required_t"] for case in fp32_cases), default=0.0)
        for name in RESULT_NAMES
    }
    declared = {
        "float32": op.tolerances.get("float32"),
        "bfloat16": op.tolerances.get("bfloat16"),
        "multipliers": op.tolerance_multipliers,
    }
    return {
        "schema_version": "evograd-llama3-mlp-tolerance/1",
        "task": TASK_NAME,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|)) using the declaration's multipliers"
        ),
        "multipliers": {name: list(value) for name, value in multipliers.items()},
        "compared": "declared float32-accumulated forward vs runtime_forward (the HF BF16 spelling)",
        "cases": cases,
        "worst_required_t": {
            "overall": by_result,
            "bfloat16": bf16_by_result,
            "float32": fp32_by_result,
        },
        "canonical_worst_required_t": canonical["worst_required_t"] if canonical else None,
        "declared_tolerances": declared,
    }


def summarize_calibration(report: dict[str, Any]) -> str:
    lines = [
        f"tolerance calibration for {report['task']}",
        f"  metric: {report['metric']}",
        f"  comparing: {report['compared']}",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"  {case['label']}  [{case['dtype']}]")
        for name in RESULT_NAMES:
            record = case["results"][name]
            lines.append(
                f"    {name:<14} required_t {record['required_t']:.3e}   "
                f"max_abs {record['max_abs_err']:.3e}   "
                f"noise {case['production_noise_required_t'][name]:.3e}"
            )
    lines.append("")
    for group, values in report["worst_required_t"].items():
        worst = max(values.values()) if values else 0.0
        lines.append(f"  worst required_t ({group}): {worst:.3e}   per result: " + ", ".join(
            f"{name}={value:.2e}" for name, value in values.items()
        ))
    lines.append(f"  declared: {report['declared_tolerances']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.benchmark.topdown.llama3_8b.levels.level2.swiglu_mlp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(sp):
        sp.add_argument(
            "--source", type=Path, default=Path("results/llama3-level4/layer16.pt")
        )
        sp.add_argument("--device", default="cuda")
        return sp

    derive = source_args(sub.add_parser("derive", help="describe the derived invocation"))
    derive.add_argument("--metadata-out", type=Path, default=None)

    verify = source_args(
        sub.add_parser("verify", help="check the Level-2 reference against the capture")
    )
    verify.add_argument("--report", type=Path, default=None)
    verify.add_argument("--noise-repeats", type=int, default=4)

    calibrate = source_args(
        sub.add_parser("calibrate", help="measure the tolerance a correct BF16 spelling needs")
    )
    calibrate.add_argument("--report", type=Path, default=None)
    calibrate.add_argument(
        "--skip-canonical",
        action="store_true",
        help="calibrate on the declared correctness workloads only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "derive":
        try:
            payload, metadata = derive_mlp_invocation(args.source, device=args.device)
        except (MlpExtractionError, ArtifactError) as exc:
            print(f"derivation failed: {exc}", file=sys.stderr)
            return 1
        print(f"derived {TASK_NAME} from {args.source} (no tensors written)")
        for link in payload["provenance_chain"]:
            print(f"  -> {link}")
        print(f"  content    {payload['content_hash']}")
        print(f"  derivation {payload['derivation_hash']}")
        if args.metadata_out is not None:
            Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.metadata_out).write_text(
                json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.metadata_out}")
        return 0

    from evograd.benchmark.topdown.llama3_8b.levels.level3.prepare import validate_noise_repeats
    if args.command == "calibrate":
        try:
            report = run_calibration(
                args.source, device=args.device, skip_canonical=args.skip_canonical
            )
        except (MlpExtractionError, ArtifactError) as exc:
            print(f"calibration failed: {exc}", file=sys.stderr)
            return 1
        print(summarize_calibration(report))
        if args.report is not None:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {args.report}")
        return 0

    try:
        validate_noise_repeats(args.noise_repeats)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        payload, _ = derive_mlp_invocation(args.source, device=args.device)
        report = run_verify(payload, device=args.device, noise_repeats=args.noise_repeats)
    except (MlpExtractionError, ArtifactError) as exc:
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(summarize_verify(report))
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
