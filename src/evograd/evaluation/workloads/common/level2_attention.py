"""Judging an implementation of one model's attention boundary.

The case itself -- the capture, the provenance checks, the declaration check --
belongs to that model's ``levels.level2.attention.capture``. What is here is the
judgment. Written once; ``descriptor`` names the model.
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


#: The boundary this module judges, as named under ``levels/level2/``.
BOUNDARY = "attention"

#: Combined with each model's prefix, e.g. ``evograd-llama3-…``.
REPORT_SCHEMA_SUFFIX = "attention-verify/1"

# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _pair_pass(forward, q, k, v, o_weight, dout):
    """Forward and backward through one spelling, keeping the observed layout."""
    leaves = {
        "q": q.detach().clone().requires_grad_(True),
        "k": k.detach().clone().requires_grad_(True),
        "v": v.detach().clone().requires_grad_(True),
        "o_weight": o_weight.detach().clone().requires_grad_(True),
    }
    out = forward(**leaves)
    out.backward(dout)
    return {
        "out": out.detach().clone(),
        "dq": leaves["q"].grad.detach().clone(),
        "dk": leaves["k"].grad.detach().clone(),
        "dv": leaves["v"].grad.detach().clone(),
        "do_weight": leaves["o_weight"].grad.detach().clone(),
    }


def run_verify(
    payload: dict[str, Any],
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
    include_dense_reference: bool = True,
    descriptor,
) -> dict[str, Any]:
    """Check both spellings against what Transformers computed.

    Two comparisons with two separately justified tolerances, as for the MLP.
    The production spelling is *the same call* the model made, so it is held to
    the Level-3 replay tolerances -- one BF16 unit roundoff forward, one epsilon
    on gradients. The declared dense reference computes the softmax in float32
    over a materialized score matrix, which is deliberately a different
    computation, and is held to the operator's declared tolerance.
    """
    cap = descriptor.capture(BOUNDARY)
    TASK_NAME = cap.TASK_NAME
    RESULT_NAMES = cap.RESULT_NAMES
    AttentionExtractionError = cap.AttentionExtractionError
    _declared_case = cap._declared_case
    check_provenance = cap.check_provenance
    declaration_problems = cap.declaration_problems
    load_snapshot = descriptor.module("harvest.snapshot").load
    from evograd.benchmark import get_task
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

    from evograd.evaluation.workloads.common.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
        declared_gate,
    )
    validate_noise_repeats = descriptor.module("levels.level3.prepare").validate_noise_repeats

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AttentionExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    shape_problems = declaration_problems(snapshot_path)

    op = get_task(TASK_NAME)
    # This workload's observed case, not ``benchmark[0]``: the declaration is
    # shared with Qwen3, whose shape is the one that would come back.
    case, missing = _declared_case(op)
    if case is None:
        raise AttentionExtractionError(missing)
    base = op.tolerances[case.dtype][0]
    # Every result names itself. ``tolerance_for(case)`` with no result name
    # reaches ``ReductionScaledAtol`` as ``result_name=None``, which matches no
    # entry in its ``result_dims`` -- so the hook silently returns the unscaled
    # base and ``out`` alone is gated at anchor width.
    declared_tol = {
        name: op.tolerance_for(case, name)
        for name in ("out", "dq", "dk", "dv", "do_weight")
    }

    tensors = {
        "q": payload["q"].to(device),
        "k": payload["k"].to(device),
        "v": payload["v"].to(device),
        "o_weight": payload["o_weight"].to(device),
        "dout": payload["grad_output"].to(device),
    }
    captured = {
        "out": payload["out"].to(device),
        **{name: tensor.to(device) for name, tensor in payload["grads"].items()},
    }

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    production = _pair_pass(resolve_runtime_forward(op), **tensors)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    production_comparisons = {
        name: compare_tensors(
            production[name], captured[name], FORWARD_TOL if name == "out" else GRADIENT_TOL
        )
        for name in RESULT_NAMES
    }

    dense_comparisons = None
    if include_dense_reference:
        dense = _pair_pass(resolve_forward(op), **tensors)
        dense_comparisons = {
            name: declared_gate(dense[name], captured[name], declared_tol[name], base)
            for name in RESULT_NAMES
        }
        del dense

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "production spelling vs itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [
            _pair_pass(resolve_runtime_forward(op), **tensors) for _ in range(noise_repeats)
        ]
        noise["results"] = {
            name: _max_noise(
                _noise(passes[i][name], passes[0][name]) for i in range(1, noise_repeats)
            )
            for name in RESULT_NAMES
        }

    failures = [f"provenance: {p}" for p in provenance_problems]
    failures += [f"declaration: {p}" for p in shape_problems]
    for name, record in production_comparisons.items():
        if not record["within_tolerance"]:
            failures.append(
                f"production {name}: max_rel_err_vs_scale="
                f"{record.get('max_rel_err_vs_scale')} > {record['tolerance']}"
            )
    for name, record in (dense_comparisons or {}).items():
        if not record["within_tolerance"]:
            failures.append(
                f"declared reference {name}: required_t={record.get('required_t')} > "
                f"base {record.get('declared_base')}"
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
        "sdpa_attrs": payload["attrs"],
        "tolerances": {
            "metric": "max|a-b| / max|b| (reference scale) for production; allclose for the reference",
            "production": {
                "forward": FORWARD_TOL,
                "gradient": GRADIENT_TOL,
                "why": (
                    "the same SDPA call the model made, so it is held to the "
                    "Level-3 replay tolerances"
                ),
            },
            "declared_reference": {
                "values": {name: list(value) for name, value in declared_tol.items()},
                "why": (
                    "the dense float32-softmax spelling is deliberately a "
                    "different computation; the question is whether it is a "
                    "valid answer for the operator"
                ),
            },
        },
        "comparisons": {
            "production": production_comparisons,
            "declared_reference": dense_comparisons,
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
        f"[{report['status'].upper()}] {report['task']} against the captured attention boundary",
        f"  provenance validated: {report['provenance_validated']}  "
        f"snapshot {report['snapshot_hash'][:16]}...",
    ]
    for link in report["provenance_chain"]:
        lines.append(f"    -> {link}")
    lines += [
        "",
        f"  declared dims {report['declared_dims']}",
        f"  sdpa {report['sdpa_attrs']}",
        "",
        "  production spelling (the same SDPA call the model made):",
    ]
    for name, record in report["comparisons"]["production"].items():
        lines.append(
            f"    {name:<10} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"bitwise {record['bitwise_identical']}  stride_ok {record['stride_match']}"
        )
    dense = report["comparisons"]["declared_reference"]
    if dense:
        lines += ["", "  declared dense reference, against the operator's allclose gate:"]
        for name, record in dense.items():
            lines.append(
                f"    {name:<10} rel {_fmt(record['max_rel_err_vs_scale'])}  "
                f"required_t {_fmt(record.get('required_t'))} <= base "
                f"{record.get('declared_base')}"
            )
    noise = report["noise_floor"]
    if noise.get("measured"):
        lines += ["", f"  measured noise floor ({noise['repeats']} production runs):"]
        lines.append(
            "    " + "  ".join(f"{n}={_fmt(v)}" for n, v in noise["results"].items())
        )
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# tolerance calibration
# --------------------------------------------------------------------------


def _calibration_case(label, dtype, tensors, multipliers, repeats: int = 3, *, op, RESULT_NAMES):
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

    reference = _pair_pass(resolve_forward(op), **tensors)
    production = _pair_pass(resolve_runtime_forward(op), **tensors)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(resolve_runtime_forward(op), **tensors)
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
    descriptor,
) -> dict[str, Any]:
    cap = descriptor.capture(BOUNDARY)
    TASK_NAME = cap.TASK_NAME
    RESULT_NAMES = cap.RESULT_NAMES
    derive_attention_invocation = cap.derive_attention_invocation
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.benchmark import get_task

    op = get_task(TASK_NAME)
    multipliers = {
        name: tuple(op.tolerance_multipliers.get(name, (1.0, 1.0))) for name in RESULT_NAMES
    }
    cases = []
    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            _calibration_case(
                f"correctness {workload.dims}",
                workload.dtype,
                {k: values[k] for k in ("q", "k", "v", "o_weight", "dout")},
                multipliers,
                op=op,
                RESULT_NAMES=RESULT_NAMES,
            )
        )

    canonical = None
    if not skip_canonical:
        payload, _ = derive_attention_invocation(
            source, device=device, snapshot_path=snapshot_path
        )
        canonical = _calibration_case(
            f"canonical layer-{descriptor.representative_layer} invocation",
            str(payload["q"].dtype).replace("torch.", ""),
            {
                "q": payload["q"].to(device),
                "k": payload["k"].to(device),
                "v": payload["v"].to(device),
                "o_weight": payload["o_weight"].to(device),
                "dout": payload["grad_output"].to(device),
            },
            multipliers,
            repeats=2,
            op=op,
            RESULT_NAMES=RESULT_NAMES,
        )
        cases.append(canonical)

    def worst(subset):
        return {
            name: max((c["results"][name]["required_t"] for c in subset), default=0.0)
            for name in RESULT_NAMES
        }

    return {
        "schema_version": descriptor.schema("attention-tolerance/1"),
        "task": TASK_NAME,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|))"
        ),
        "compared": "declared dense float32-softmax forward vs runtime_forward (SDPA)",
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


def summarize_calibration(report: dict[str, Any], *, descriptor) -> str:
    cap = descriptor.capture(BOUNDARY)
    RESULT_NAMES = cap.RESULT_NAMES
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
                f"    {name:<10} required_t {record['required_t']:.3e}   "
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
        prog=f"python -m evograd.evaluation.workloads.{descriptor.name}.level2.attention",
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
    verify.add_argument(
        "--skip-dense-reference",
        action="store_true",
        help="skip the materialized-score reference (it needs several GiB at the canonical shape)",
    )

    calibrate = source_args(sub.add_parser("calibrate", help="measure the required tolerance"))
    calibrate.add_argument("--report", type=Path, default=None)
    calibrate.add_argument("--skip-canonical", action="store_true")
    return parser


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None, *, descriptor) -> int:
    cap = descriptor.capture(BOUNDARY)
    TASK_NAME = cap.TASK_NAME
    RESULT_NAMES = cap.RESULT_NAMES
    AttentionExtractionError = cap.AttentionExtractionError
    derive_attention_invocation = cap.derive_attention_invocation
    ArtifactError = descriptor.module("levels.level3.artifact").ArtifactError
    parser = build_parser(descriptor)
    args = parser.parse_args(argv)

    try:
        if args.command == "derive":
            payload, metadata = derive_attention_invocation(args.source, device=args.device)
            print(f"derived {TASK_NAME} from {args.source} (no tensors written)")
            for link in payload["provenance_chain"]:
                print(f"  -> {link}")
            print(f"  content    {payload['content_hash']}")
            print(f"  derivation {payload['derivation_hash']}")
            _write(args.metadata_out, metadata)
            return 0

        if args.command == "calibrate":
            report = run_calibration(
                args.source, device=args.device, skip_canonical=args.skip_canonical,
                descriptor=descriptor,
            )
            print(summarize_calibration(report, descriptor=descriptor))
            _write(args.report, report)
            return 0

        validate_noise_repeats = descriptor.module("levels.level3.prepare").validate_noise_repeats
        try:
            validate_noise_repeats(args.noise_repeats)
        except ValueError as exc:
            parser.error(str(exc))
        payload, _ = derive_attention_invocation(args.source, device=args.device)
        report = run_verify(
            payload,
            device=args.device,
            noise_repeats=args.noise_repeats,
            include_dense_reference=not args.skip_dense_reference,
            descriptor=descriptor,
        )
    except (AttentionExtractionError, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1

