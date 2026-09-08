"""Judging an implementation of ``llama3_residual_rmsnorm`` against the captured boundary.

The case itself belongs to :mod:`evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.capture`.
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
from evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.capture import (
    DIRECT_GRADS,
    OUTPUT_NAMES,
    ResidualExtractionError,
    TASK_NAME,
    check_provenance,
    declaration_problems,
    derive_residual_invocation,
)

REPORT_SCHEMA = "evograd-llama3-residual-verify/1"

# --------------------------------------------------------------------------
# verification and calibration
# --------------------------------------------------------------------------


def transformers_spelling(x, residual, weight, eps=1e-6):
    """The exact ``LlamaRMSNorm`` spelling, after the residual add.

    ``LlamaRMSNorm`` computes the variance in float32 but applies the learned
    weight *after* casting back to the input dtype. ``F.rms_norm`` -- the
    declaration's ``runtime_forward`` -- is a different kernel with a different
    rounding, so this third spelling exists to say which of the two the model
    actually ran, and by how much they differ. It is Qwen-specific and stays
    here rather than in the generic declaration.
    """
    summed = x + residual
    wide = summed.to(torch.float32)
    variance = wide.pow(2).mean(-1, keepdim=True)
    wide = wide * torch.rsqrt(variance + eps)
    return weight * wide.to(summed.dtype), summed


def _pair_pass(forward, x, residual, weight, dout, dsummed, eps):
    """Forward and backward through one spelling, with both upstream gradients."""
    leaves = {
        "x": x.detach().clone().requires_grad_(True),
        "r": residual.detach().clone().requires_grad_(True),
        "weight": weight.detach().clone().requires_grad_(True),
    }
    out, summed = forward(leaves["x"], leaves["r"], leaves["weight"], eps)
    torch.autograd.backward((out, summed), (dout, dsummed))
    return {
        "out": out.detach().clone(),
        "summed": summed.detach().clone(),
        "dx": leaves["x"].grad.detach().clone(),
        "dr": leaves["r"].grad.detach().clone(),
        "dweight": leaves["weight"].grad.detach().clone(),
    }


def run_verify(
    payload: dict[str, Any],
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.reference import (
        llama3_residual_rmsnorm_forward_ref,
        llama3_residual_rmsnorm_runtime_ref,
    )

    from evograd.evaluation.workloads.llama3_8b.level3.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
    )
    from evograd.benchmark.topdown.llama3_8b.levels.level3.prepare import (
        validate_noise_repeats,
    )

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ResidualExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    declaration_issues = declaration_problems(snapshot_path)

    op = get_task(TASK_NAME)
    case = op.benchmark_workloads("llama_3_8b_observed")[0]
    base = op.tolerance_for(case)[0]
    declared_tol = {
        name: op.tolerance_for(case, name)
        for name in ("out", "summed", "dx", "dr", "dweight")
    }

    eps = float(payload["attrs"]["eps"])
    args = (
        payload["inputs"]["x"].to(device),
        payload["inputs"]["r"].to(device),
        payload["inputs"]["weight"].to(device),
        payload["output_grads"]["dout"].to(device),
        payload["output_grads"]["dsummed"].to(device),
        eps,
    )
    captured = {
        "out": payload["outputs"]["out"].to(device),
        "summed": payload["outputs"]["summed"].to(device),
        "dx": payload["grads"]["dx"].to(device),
        "dtotal": payload["grads"]["dtotal"].to(device),
        "dweight": payload["grads"]["dweight"].to(device),
    }

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    transformers = _pair_pass(transformers_spelling, *args)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    runtime = _pair_pass(llama3_residual_rmsnorm_runtime_ref, *args)
    reference = _pair_pass(llama3_residual_rmsnorm_forward_ref, *args)

    def compare_group(results, tolerance_of):
        """Every directly observable quantity, against the capture.

        ``dtotal`` is compared against the spelling's own ``dx``: the contract
        says ``dx == dr == dtotal``, and ``summed``'s gradient in the layer is
        ``dtotal``, so this is the check that the two output paths were
        combined -- not a restatement of the ``dx`` comparison, because the two
        captured tensors come from different places in the layer graph.
        """
        checks = {}
        for name in OUTPUT_NAMES:
            checks[name] = tolerance_of(name, results[name], captured[name])
        for name in DIRECT_GRADS:
            checks[name] = tolerance_of(name, results[name], captured[name])
        checks["dtotal"] = tolerance_of("dx", results["dx"], captured["dtotal"])
        return checks

    transformers_checks = compare_group(
        transformers,
        lambda name, a, b: compare_tensors(
            a, b, FORWARD_TOL if name in OUTPUT_NAMES else GRADIENT_TOL
        ),
    )
    runtime_checks = compare_group(
        runtime, lambda name, a, b: declared_gate(a, b, declared_tol[name], base)
    )
    reference_checks = compare_group(
        reference, lambda name, a, b: declared_gate(a, b, declared_tol[name], base)
    )

    # `dr` cannot be compared against the model: layer 14's residual feeds
    # input_layernorm as well, so its gradient there is not this boundary's.
    # What the contract claims about it -- dr == dx -- is provable in isolation.
    isolated = {
        spelling: {
            "dr_equals_dx_bitwise": bool(torch.equal(results["dr"], results["dx"])),
            "max_abs_dr_minus_dx": float(
                (results["dr"].float() - results["dx"].float()).abs().max()
            ),
        }
        for spelling, results in (
            ("transformers", transformers),
            ("runtime_forward", runtime),
            ("declared_reference", reference),
        )
    }

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "transformers spelling vs itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [_pair_pass(transformers_spelling, *args) for _ in range(noise_repeats)]
        noise["results"] = {
            name: _max_noise(
                _noise(passes[i][name], passes[0][name]) for i in range(1, noise_repeats)
            )
            for name in ("out", "summed", "dx", "dr", "dweight")
        }

    failures = [f"provenance: {p}" for p in provenance_problems]
    failures += [f"declaration: {p}" for p in declaration_issues]
    for label, group in (
        ("transformers", transformers_checks),
        ("runtime_forward", runtime_checks),
        ("declared reference", reference_checks),
    ):
        for name, record in group.items():
            if not record["within_tolerance"]:
                failures.append(
                    f"{label} {name}: max_rel_err_vs_scale="
                    f"{record.get('max_rel_err_vs_scale')}, required_t="
                    f"{record.get('required_t')}, tolerance={record.get('tolerance')}"
                )
    for spelling, record in isolated.items():
        if not record["dr_equals_dx_bitwise"]:
            failures.append(
                f"{spelling}: dr != dx bitwise (max abs "
                f"{record['max_abs_dr_minus_dx']}), which the contract requires"
            )

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": TASK_NAME,
        "identity": payload["identity"],
        "provenance_chain": payload["provenance_chain"],
        "provenance_validated": not provenance_problems,
        "snapshot_hash": load_snapshot(snapshot_path)["snapshot_hash"],
        "declared_case": {"dims": case.dims, "dtype": case.dtype},
        "attrs": payload["attrs"],
        "tolerances": {
            "transformers": {"forward": FORWARD_TOL, "gradient": GRADIENT_TOL},
            "declared": {name: list(value) for name, value in declared_tol.items()},
            "declared_base": base,
        },
        "comparisons": {
            "transformers": transformers_checks,
            "runtime_forward": runtime_checks,
            "declared_reference": reference_checks,
        },
        "isolated_dresidual_proof": {
            "why": (
                "layer 14's decoder input feeds both the residual add and "
                "input_layernorm, so its gradient in the layer is this "
                "boundary's dresidual plus a path outside the boundary; a "
                "direct comparison would be a claim the graph cannot support"
            ),
            "results": isolated,
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
    lines += [
        "",
        f"  declared case {report['declared_case']}",
        f"  attrs {report['attrs']}",
        "",
    ]
    for label, group in report["comparisons"].items():
        lines.append(f"  {label}:")
        for name, record in group.items():
            lines.append(
                f"    {name:<10} rel {_fmt(record['max_rel_err_vs_scale'])}  "
                f"bitwise {record['bitwise_identical']}  "
                f"required_t {_fmt(record.get('required_t'))}"
            )
    proof = report["isolated_dresidual_proof"]
    lines += ["", "  dresidual (not directly comparable; proved in isolation):"]
    for spelling, record in proof["results"].items():
        lines.append(
            f"    {spelling:<20} dr == dx bitwise: {record['dr_equals_dx_bitwise']}"
        )
    noise = report["noise_floor"]
    if noise.get("measured"):
        lines += ["", f"  measured noise floor ({noise['repeats']} runs):"]
        lines.append("    " + "  ".join(f"{n}={_fmt(v)}" for n, v in noise["results"].items()))
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


RESULT_NAMES = ("out", "summed", "dx", "dr", "dweight")


def _calibration_case(label, dtype, args, multipliers, repeats: int = 3):
    from evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.reference import (
        llama3_residual_rmsnorm_forward_ref,
        llama3_residual_rmsnorm_runtime_ref,
    )

    reference = _pair_pass(llama3_residual_rmsnorm_forward_ref, *args)
    production = _pair_pass(llama3_residual_rmsnorm_runtime_ref, *args)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(llama3_residual_rmsnorm_runtime_ref, *args)
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
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.benchmark import get_task

    op = get_task(TASK_NAME)
    cases = []
    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        # The declaration's own gate for this workload, including its
        # per-workload atol/rtol override and the dweight hook.
        multipliers = {
            name: tuple(
                t / b if b else 1.0
                for t, b in zip(op.tolerance_for(workload, name), op.tolerance_for(workload))
            )
            for name in RESULT_NAMES
        }
        cases.append(
            _calibration_case(
                f"correctness {workload.dims}",
                workload.dtype,
                (
                    values["x"],
                    values["r"],
                    values["weight"],
                    values["dout"],
                    values["dsummed"],
                    float(values["eps"]),
                ),
                multipliers,
            )
        )

    if not skip_canonical:
        payload, _ = derive_residual_invocation(
            source, device=device, snapshot_path=snapshot_path
        )
        case = op.benchmark_workloads("llama_3_8b_observed")[0]
        multipliers = {
            name: tuple(
                t / b if b else 1.0
                for t, b in zip(op.tolerance_for(case, name), op.tolerance_for(case))
            )
            for name in RESULT_NAMES
        }
        cases.append(
            _calibration_case(
                "canonical layer-16 invocation",
                "bfloat16",
                (
                    payload["inputs"]["x"].to(device),
                    payload["inputs"]["r"].to(device),
                    payload["inputs"]["weight"].to(device),
                    payload["output_grads"]["dout"].to(device),
                    payload["output_grads"]["dsummed"].to(device),
                    float(payload["attrs"]["eps"]),
                ),
                multipliers,
                repeats=2,
            )
        )

    def worst(subset):
        return {
            name: max((c["results"][name]["required_t"] for c in subset), default=0.0)
            for name in RESULT_NAMES
        }

    return {
        "schema_version": "evograd-fused-add-rms-norm-tolerance/1",
        "task": TASK_NAME,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|)), using the declaration's own "
            "per-workload tolerance and dweight hook as the multipliers"
        ),
        "compared": "declared primitive forward vs runtime_forward (F.rms_norm)",
        "cases": cases,
        "worst_required_t": {
            "overall": worst(cases),
            "bfloat16": worst([c for c in cases if c["dtype"] == "bfloat16"]),
            "float32": worst([c for c in cases if c["dtype"] == "float32"]),
            "float16": worst([c for c in cases if c["dtype"] == "float16"]),
        },
        "declared_tolerances": {
            "tolerances": op.tolerances,
            "has_tolerance_hook": op.tolerance_hook is not None,
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
                f"    {name:<10} required_t {record['required_t']:.3e}   "
                f"rel_vs_scale {_fmt(record['max_rel_err_vs_scale'])}   "
                f"noise {case['production_noise_required_t'][name]:.3e}"
            )
    lines.append("")
    for group, values in report["worst_required_t"].items():
        worst = max(values.values()) if values else 0.0
        lines.append(
            f"  worst required_t ({group}): {worst:.3e}   "
            + ", ".join(f"{n}={v:.2e}" for n, v in values.items())
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(sp):
        sp.add_argument("--source", type=Path, default=Path("results/llama3-level4/layer16.pt"))
        sp.add_argument("--device", default="cuda")
        return sp

    derive = source_args(sub.add_parser("derive", help="describe the derived invocation"))
    derive.add_argument("--metadata-out", type=Path, default=None)

    verify = source_args(sub.add_parser("verify", help="check the spellings against the capture"))
    verify.add_argument("--report", type=Path, default=None)
    verify.add_argument("--noise-repeats", type=int, default=4)

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            payload, metadata = derive_residual_invocation(args.source, device=args.device)
            print(f"derived {TASK_NAME} from {args.source} (no tensors written)")
            for link in payload["provenance_chain"]:
                print(f"  -> {link}")
            print(f"  content    {payload['content_hash']}")
            print(f"  derivation {payload['derivation_hash']}")
            _write(args.metadata_out, metadata)
            return 0

        if args.command == "calibrate":
            report = run_calibration(
                args.source, device=args.device, skip_canonical=args.skip_canonical
            )
            print(summarize_calibration(report))
            _write(args.report, report)
            return 0

        from evograd.benchmark.topdown.llama3_8b.levels.level3.prepare import validate_noise_repeats
        try:
            validate_noise_repeats(args.noise_repeats)
        except ValueError as exc:
            parser.error(str(exc))
        payload, _ = derive_residual_invocation(args.source, device=args.device)
        report = run_verify(payload, device=args.device, noise_repeats=args.noise_repeats)
    except (ResidualExtractionError, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
