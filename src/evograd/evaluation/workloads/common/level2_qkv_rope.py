"""Judging an implementation of one model's q/k/v projection and rotary boundary.

The case itself belongs to that model's ``levels.level2.qkv_rope.capture``.
What is here is the judgment. Written once; ``descriptor`` names the model.
"""

from __future__ import annotations

from evograd.evaluation.workloads.common.replay import declared_gate, required_tolerance

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch


#: The boundary this module judges.
#: Boundary-specific and identical in every model that presents it;
#: `_bind` asserts the model's capture module still agrees.
OUTPUT_NAMES = ('q', 'k', 'v')
GRAD_NAMES = ('dx', 'dq_weight', 'dk_weight', 'dv_weight')

BOUNDARY = "qkv_rope"

#: Combined with each model's prefix.
REPORT_SCHEMA_SUFFIX = "qkv-verify/1"

# --------------------------------------------------------------------------
# verification and calibration
# --------------------------------------------------------------------------

ARG_ORDER = ("x", "q_weight", "k_weight", "v_weight", "cos", "sin")
ACTIVE_ARGS = ARG_ORDER[:4]


def _bind(descriptor):
    """This model's capture module for the qkv_rope boundary."""
    cap = descriptor.capture(BOUNDARY)
    assert cap.OUTPUT_NAMES == OUTPUT_NAMES, (OUTPUT_NAMES, cap.OUTPUT_NAMES)
    assert cap.GRAD_NAMES == GRAD_NAMES, (GRAD_NAMES, cap.GRAD_NAMES)
    return cap


def _pair_pass(forward, inputs: dict[str, torch.Tensor], output_grads):
    """Forward and backward through one spelling, keeping the observed layout.

    No ``eps`` argument: ``llama3_qkv_rope`` declares none, because there is no
    normalization inside the boundary for one to belong to.
    """
    leaves = {name: inputs[name].detach().clone().requires_grad_(True) for name in ACTIVE_ARGS}
    outputs = forward(*(leaves.get(name, inputs.get(name)) for name in ARG_ORDER))
    torch.autograd.backward(tuple(outputs), tuple(output_grads))
    result = {name: out.detach().clone() for name, out in zip(OUTPUT_NAMES, outputs)}
    result.update(
        {f"d{name}": leaves[name].grad.detach().clone() for name in ACTIVE_ARGS}
    )
    return result


def run_verify(
    payload: dict[str, Any],
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
    descriptor,
) -> dict[str, Any]:
    """Check both spellings against what Transformers computed.

    The production spelling is the same computation the model ran, so it is held
    to the Level-3 replay tolerances. The declared reference carries the rotation
    in float32 and is therefore deliberately a different computation, held to
    the operator's own declared gate.
    """
    cap = _bind(descriptor)
    TASK_NAME = cap.TASK_NAME
    QkvExtractionError = cap.QkvExtractionError
    check_provenance = cap.check_provenance
    derive_qkv_invocation = cap.derive_qkv_invocation
    declaration_problems = cap.declaration_problems
    load_snapshot = descriptor.module("harvest.snapshot").load
    ArtifactError = descriptor.module("levels.level3.artifact").ArtifactError
    from evograd.benchmark import get_task
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

    from evograd.evaluation.workloads.common.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
    )
    validate_noise_repeats = descriptor.module(
        "levels.level3.prepare"
    ).validate_noise_repeats

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QkvExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    declaration_issues = declaration_problems(snapshot_path)

    op = get_task(TASK_NAME)
    case = op.benchmark[0]
    base = op.tolerances[case.dtype][0]
    declared_tol = {name: op.tolerance_for(case, name) for name in OUTPUT_NAMES}
    declared_tol.update({name: op.tolerance_for(case, name) for name in GRAD_NAMES})

    inputs = {k: v.to(device) for k, v in payload["inputs"].items()}
    output_grads = [payload["output_grads"][f"d{n}"].to(device) for n in OUTPUT_NAMES]
    captured = {
        **{n: payload["outputs"][n].to(device) for n in OUTPUT_NAMES},
        **{n: payload["grads"][n].to(device) for n in GRAD_NAMES},
    }
    names = OUTPUT_NAMES + GRAD_NAMES

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    production = _pair_pass(
        resolve_runtime_forward(op), inputs, output_grads
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    production_comparisons = {
        name: compare_tensors(
            production[name], captured[name],
            FORWARD_TOL if name in OUTPUT_NAMES else GRADIENT_TOL,
        )
        for name in names
    }

    reference = _pair_pass(resolve_forward(op), inputs, output_grads)
    reference_comparisons = {
        name: declared_gate(reference[name], captured[name], declared_tol[name], base)
        for name in names
    }

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "production spelling vs itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [
            _pair_pass(resolve_runtime_forward(op), inputs, output_grads)
            for _ in range(noise_repeats)
        ]
        noise["results"] = {
            name: _max_noise(
                _noise(passes[i][name], passes[0][name]) for i in range(1, noise_repeats)
            )
            for name in names
        }

    failures = [f"provenance: {p}" for p in provenance_problems]
    failures += [f"declaration: {p}" for p in declaration_issues]
    for label, group in (("production", production_comparisons), ("declared reference", reference_comparisons)):
        for name, record in group.items():
            if not record["within_tolerance"]:
                failures.append(
                    f"{label} {name}: max_rel_err_vs_scale="
                    f"{record.get('max_rel_err_vs_scale')}, required_t="
                    f"{record.get('required_t')}, tolerance={record.get('tolerance')}"
                )

    return {
        "schema_version": descriptor.schema(REPORT_SCHEMA_SUFFIX),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": TASK_NAME,
        "identity": payload["identity"],
        "provenance_chain": payload["provenance_chain"],
        "provenance_validated": not provenance_problems,
        "snapshot_hash": load_snapshot(snapshot_path)["snapshot_hash"],
        "declared_dims": case.dims,
        "attrs": payload["attrs"],
        "tolerances": {
            "production": {"forward": FORWARD_TOL, "gradient": GRADIENT_TOL},
            "declared_reference": {
                name: list(value) for name, value in declared_tol.items()
            },
            "declared_base": base,
        },
        "comparisons": {
            "production": production_comparisons,
            "declared_reference": reference_comparisons,
        },
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
    lines = [
        f"[{report['status'].upper()}] {report['task']} against the captured boundary",
        f"  provenance validated: {report['provenance_validated']}  "
        f"snapshot {report['snapshot_hash'][:16]}...",
    ]
    for link in report["provenance_chain"]:
        lines.append(f"    -> {link}")
    lines += ["", f"  declared dims {report['declared_dims']}", f"  attrs {report['attrs']}", ""]
    lines.append("  production spelling (the same computation the model ran):")
    for name, record in report["comparisons"]["production"].items():
        lines.append(
            f"    {name:<16} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"bitwise {record['bitwise_identical']}  stride_ok {record['stride_match']}"
        )
    lines.append("")
    lines.append("  declared float32 reference, against the operator's allclose gate:")
    for name, record in report["comparisons"]["declared_reference"].items():
        lines.append(
            f"    {name:<16} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"required_t {_fmt(record.get('required_t'))} <= base "
            f"{record.get('declared_base')}"
        )
    noise = report["noise_floor"]
    if noise.get("measured"):
        lines += ["", f"  measured noise floor ({noise['repeats']} production runs):"]
        lines.append("    " + "  ".join(f"{n}={_fmt(v)}" for n, v in noise["results"].items()))
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


def _calibration_case(label, dtype, inputs, output_grads, multipliers, repeats: int = 3):
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

    names = OUTPUT_NAMES + GRAD_NAMES
    reference = _pair_pass(resolve_forward(op), inputs, output_grads)
    production = _pair_pass(resolve_runtime_forward(op), inputs, output_grads)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in names
    }
    noise = {name: 0.0 for name in names}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(resolve_runtime_forward(op), inputs, output_grads)
        for name in names:
            noise[name] = max(
                noise[name],
                required_tolerance(
                    again[name], production[name], multipliers.get(name, (1.0, 1.0))
                )["required_t"],
            )
    return {
        "label": label,
        "dtype": dtype,
        "results": results,
        "production_noise_required_t": noise,
        "worst_required_t": max(r["required_t"] for r in results.values()),
    }


def run_calibration(
    source: Path,
    *,
    device: str = "cuda",
    skip_canonical: bool = False,
    skip_observed: bool = False,
    snapshot_path: Path | None = None,
    descriptor,
) -> dict[str, Any]:
    cap = _bind(descriptor)
    TASK_NAME = cap.TASK_NAME
    QkvExtractionError = cap.QkvExtractionError
    check_provenance = cap.check_provenance
    derive_qkv_invocation = cap.derive_qkv_invocation
    declaration_problems = cap.declaration_problems
    load_snapshot = descriptor.module("harvest.snapshot").load
    ArtifactError = descriptor.module("levels.level3.artifact").ArtifactError
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.benchmark import get_task

    op = get_task(TASK_NAME)
    names = OUTPUT_NAMES + GRAD_NAMES
    multipliers = {
        name: tuple(op.tolerance_multipliers.get(name, (1.0, 1.0))) for name in names
    }
    cases = []
    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            _calibration_case(
                f"correctness {workload.dims}",
                workload.dtype,
                {name: values[name] for name in ARG_ORDER},
                [values[f"d{n}"] for n in OUTPUT_NAMES],
                multipliers,
            )
        )

    # The shape the model runs, on synthetic inputs at its own widths. Without
    # this case the calibration sees only the grid -- whose longest reduction is
    # 64 tokens against the model's 4096 -- and the harvested capture, whose
    # gradients arrive near zero (`ref_absmax` 0.0, errors ~1e-07) and so cannot
    # exercise a gradient tolerance at all. That blind spot is how this operator
    # came to reject a correct kernel at production width while every number in
    # its calibration artifact looked comfortable.
    if not skip_observed:
        for workload in op.benchmark:
            values = make_case_inputs(op, workload, device=device)
            cases.append(
                _calibration_case(
                    f"observed {workload.dims} (synthetic)",
                    workload.dtype,
                    {name: values[name] for name in ARG_ORDER},
                    [values[f"d{n}"] for n in OUTPUT_NAMES],
                    multipliers,
                    repeats=2,
                )
            )

    if not skip_canonical:
        payload, _ = derive_qkv_invocation(source, device=device, snapshot_path=snapshot_path)
        cases.append(
            _calibration_case(
                "canonical layer-16 invocation",
                str(payload["outputs"]["q"].dtype).replace("torch.", ""),
                {k: v.to(device) for k, v in payload["inputs"].items()},
                [payload["output_grads"][f"d{n}"].to(device) for n in OUTPUT_NAMES],
                multipliers,
                repeats=2,
            )
        )

    def worst(subset):
        return {
            name: max((c["results"][name]["required_t"] for c in subset), default=0.0)
            for name in names
        }

    return {
        "schema_version": descriptor.schema("qkv-tolerance/1"),
        "task": TASK_NAME,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|))"
        ),
        "compared": "declared float32 reference vs runtime_forward (the HF BF16 spelling)",
        "multipliers": {name: list(value) for name, value in multipliers.items()},
        "cases": cases,
        "worst_required_t": {
            "overall": worst(cases),
            "bfloat16": worst([c for c in cases if c["dtype"] == "bfloat16"]),
            "float32": worst([c for c in cases if c["dtype"] == "float32"]),
        },
        "declared_tolerances": {
            "float32": op.tolerances.get("float32"),
            "bfloat16": op.tolerances.get("bfloat16"),
            "multipliers": op.tolerance_multipliers,
        },
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
        for name, record in case["results"].items():
            lines.append(
                f"    {name:<16} required_t {record['required_t']:.3e}   "
                f"rel_vs_scale {_fmt(record['max_rel_err_vs_scale'])}   "
                f"noise {case['production_noise_required_t'][name]:.3e}   "
                f"min_ma@1e-2 {record['minimal_atol_multiplier']['1e-02']:.2f}"
            )
    lines.append("")
    for group, values in report["worst_required_t"].items():
        worst = max(values.values()) if values else 0.0
        lines.append(
            f"  worst required_t ({group}): {worst:.3e}   "
            + ", ".join(f"{n}={v:.2e}" for n, v in values.items())
        )
    lines.append(f"  declared: {report['declared_tolerances']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser(descriptor) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m evograd.evaluation.workloads.{descriptor.name}.level2.qkv_rope",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(sp):
        sp.add_argument("--source", type=Path, default=descriptor.artifact_path)
        sp.add_argument("--device", default="cuda")
        return sp

    derive = source_args(sub.add_parser("derive", help="describe the derived invocation"))
    derive.add_argument("--metadata-out", type=Path, default=None)

    verify = source_args(sub.add_parser("verify", help="check both spellings against the capture"))
    verify.add_argument("--report", type=Path, default=None)
    verify.add_argument("--noise-repeats", type=int, default=4)

    calibrate = source_args(sub.add_parser("calibrate", help="measure the required tolerance"))
    calibrate.add_argument("--report", type=Path, default=None)
    calibrate.add_argument("--skip-canonical", action="store_true")
    calibrate.add_argument(
        "--skip-observed",
        action="store_true",
        help="omit the model-width synthetic case (it needs a GPU with room for it)",
    )
    return parser


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None, *, descriptor) -> int:
    cap = _bind(descriptor)
    TASK_NAME = cap.TASK_NAME
    QkvExtractionError = cap.QkvExtractionError
    check_provenance = cap.check_provenance
    derive_qkv_invocation = cap.derive_qkv_invocation
    declaration_problems = cap.declaration_problems
    load_snapshot = descriptor.module("harvest.snapshot").load
    ArtifactError = descriptor.module("levels.level3.artifact").ArtifactError
    parser = build_parser(descriptor)
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            payload, metadata = derive_qkv_invocation(args.source, device=args.device)
            print(f"derived {TASK_NAME} from {args.source} (no tensors written)")
            for link in payload["provenance_chain"]:
                print(f"  -> {link}")
            print(f"  content    {payload['content_hash']}")
            print(f"  derivation {payload['derivation_hash']}")
            _write(args.metadata_out, metadata)
            return 0

        if args.command == "calibrate":
            report = run_calibration(
                args.source,
                device=args.device,
                skip_canonical=args.skip_canonical,
                skip_observed=args.skip_observed,
                descriptor=descriptor,
            )
            print(summarize_calibration(report))
            _write(args.report, report)
            return 0

        validate_noise_repeats = descriptor.module("levels.level3.prepare").validate_noise_repeats
        try:
            validate_noise_repeats(args.noise_repeats)
        except ValueError as exc:
            parser.error(str(exc))
        payload, _ = derive_qkv_invocation(args.source, device=args.device)
        report = run_verify(
            payload, device=args.device, noise_repeats=args.noise_repeats,
            descriptor=descriptor,
        )
    except (QkvExtractionError, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1

