"""Numerical calibration for the observed Qwen3 Level-2 boundaries.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.calibrate inventory \
        --report results/qwen3-level4/l2-inventory.json

One question, asked of three declarations: **what does a correct implementation
actually have to be allowed, at the shape the model runs?**

The comparison is always the same pair, and neither half is a candidate:

    reference   ``op.forward``          the declared, float32-accumulated oracle
    production  ``op.runtime_forward``  the exact spelling Transformers executes

Both are differentiated by ``torch.autograd`` from *identical* inputs and
*identical* upstream gradients, which is what tier 1's
``pytorch_autograd_provider`` and tier 2's ``check_module`` each compare. A
tolerance derived here is therefore the floor: no implementation of these
semantics can agree with the oracle more closely than the reference
implementation does, so anything tighter would reject PyTorch itself.

Three populations, because they answer different halves of the question:

* the declared **correctness** workloads -- small, CPU-runnable, what every test
  run gates on;
* the **qwen3_0_6b_observed** workloads -- the exact shapes the benchmark times,
  and the only ones at production width;
* the harvested **layer-14** invocation, whose tensors are the model's own
  rather than synthetic, which is how a scale problem is told apart from a
  reduction-length one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

TASKS = ("qwen3_qkv_norm_rope", "qwen3_attention", "qwen3_swiglu_mlp")

#: Repeats used to establish the run-to-run noise floor. A tolerance below the
#: floor is not a tolerance, it is a coin flip.
DEFAULT_REPEATS = 4


# ── one pass through one spelling ────────────────────────────────────────────


def _results(op, forward, values: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Run one spelling and collect every declared output and gradient.

    Leaves are cloned per call so the two spellings cannot share a graph, and
    the upstream gradients come from ``values`` unchanged, so the only thing
    that differs between the reference and production passes is the forward.
    """
    from evograd.opdecl.inputs import as_output_tuple, upstream_grad_values

    active = {arg.name for arg in op.active_args()}
    by_grad = {arg.grad_name: arg.name for arg in op.active_args()}
    leaves: dict[str, torch.Tensor] = {}
    positional = []
    for arg in op.args:
        value = values.get(arg.name, getattr(arg, "default", None))
        if arg.name in active and torch.is_tensor(value):
            value = value.detach().clone().requires_grad_(True)
            leaves[arg.name] = value
        positional.append(value)

    outputs = as_output_tuple(op, forward(*positional))
    upstream = upstream_grad_values(op, values)
    douts = upstream if isinstance(upstream, tuple) else (upstream,)
    grads = torch.autograd.grad(outputs, list(leaves.values()), douts)

    collected = {
        name: tensor.detach().clone()
        for name, tensor in zip(op.output_names, outputs)
    }
    by_name = dict(zip(leaves.keys(), grads))
    for grad_name in op.grad_names():
        collected[grad_name] = by_name[by_grad[grad_name]].detach().clone()
    return collected


def reference_results(op, values):
    from evograd.opdecl.oracle import resolve_forward

    return _results(op, resolve_forward(op), values)


def production_results(op, values):
    from evograd.opdecl.oracle import resolve_runtime_forward

    return _results(op, resolve_runtime_forward(op), values)


# ── what a disagreement looks like ───────────────────────────────────────────


def reduction_dims(op, result_name: str) -> dict[str, Any]:
    """Which declared dims a gradient sums over, and how long that sum is.

    A parameter gradient is a contraction: ``dgate_weight`` is ``[I, H]`` while
    the activation carrying it is ``[B, T, H]``, so ``B`` and ``T`` are summed
    away. Naming them is what makes a tolerance dimension-aware rather than a
    constant somebody picked -- BF16 rounding accumulates with the length of
    that sum, and the correctness grid's sum is 8 terms where the model's is
    4096.
    """
    by_grad = {arg.grad_name: arg for arg in op.active_args()}
    arg = by_grad.get(result_name)
    if arg is None:  # an output, not a gradient
        return {"is_gradient": False, "reduced_dims": [], "reduction_length": 1}
    present = _shape_dims(getattr(arg, "shape", None))
    everything = set()
    for other in op.args:
        everything |= set(_shape_dims(getattr(other, "shape", None)))
    for out in op.outputs:
        everything |= set(_shape_dims(getattr(out, "shape", None)))
    reduced = sorted(everything - set(present))
    return {
        "is_gradient": True,
        "arg_shape": getattr(arg, "shape", None),
        "reduced_dims": reduced,
    }


def _shape_dims(shape: str | None) -> list[str]:
    if not shape:
        return []
    inner = shape.strip()[1:-1].strip()
    return [part.strip() for part in inner.split(",")] if inner else []


def reduction_length(op, result_name: str, dims: dict[str, int]) -> int:
    """The number of terms the gradient's contraction sums, for this workload."""
    record = reduction_dims(op, result_name)
    if not record["is_gradient"]:
        return 1
    length = 1
    for name in record["reduced_dims"]:
        length *= int(dims.get(name, 1))
    return max(length, 1)


def compare(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> dict[str, Any]:
    """Everything worth knowing about one disagreement, in one place."""
    a = actual.detach().to("cpu", torch.float32)
    b = expected.detach().to("cpu", torch.float32)
    diff = (a - b).abs()
    bound = atol + rtol * b.abs()
    failures = int((diff > bound).sum())
    ref_absmax = float(b.abs().max())
    # The smallest s for which allclose(atol=s*atol, rtol=s*rtol) accepts this
    # result. 1.0 means the declared pair is exactly enough; below 1.0 is
    # margin; above 1.0 is the factor it is short by.
    required_scale = float((diff / bound).max()) if atol > 0 or rtol > 0 else float("inf")
    nonzero = b.abs() > 0
    max_rel = float((diff[nonzero] / b.abs()[nonzero]).max()) if bool(nonzero.any()) else 0.0
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "declared_atol": atol,
        "declared_rtol": rtol,
        "max_abs_err": float(diff.max()),
        "max_rel_err": max_rel,
        "max_abs_err_over_ref_absmax": float(diff.max()) / ref_absmax if ref_absmax else None,
        "allclose_failures": failures,
        "elements": int(b.numel()),
        "failure_fraction": failures / b.numel() if b.numel() else 0.0,
        "required_scale_over_declared": required_scale,
        "ref_absmax": ref_absmax,
        "ref_l2_norm": float(b.norm()),
        "ref_rms": float(b.pow(2).mean().sqrt()),
        # What an atol-only widening would cost, holding rtol at its declared
        # value: the smallest atol that accepts every element.
        "required_atol_at_declared_rtol": float(
            (diff - rtol * b.abs()).clamp(min=0.0).max()
        ),
    }


def case(
    op,
    label: str,
    workload,
    values: dict[str, Any],
    *,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """One workload, every result, plus the noise floor under it."""
    reference = reference_results(op, values)
    production = production_results(op, values)

    results: dict[str, Any] = {}
    for name in (*op.output_names, *op.grad_names()):
        atol, rtol = op.tolerance_for(workload, name)
        record = compare(production[name], reference[name], atol=atol, rtol=rtol)
        record.update(reduction_dims(op, name))
        record["reduction_length"] = reduction_length(op, name, workload.dims)
        results[name] = record

    # Run-to-run noise of the production spelling. Non-deterministic reductions
    # move on their own, and a tolerance below that movement gates on nothing.
    noise = {name: 0.0 for name in results}
    for _ in range(max(repeats - 1, 0)):
        again = production_results(op, values)
        for name in results:
            a = again[name].detach().to("cpu", torch.float32)
            b = production[name].detach().to("cpu", torch.float32)
            noise[name] = max(noise[name], float((a - b).abs().max()))

    for name, record in results.items():
        scale = record["required_scale_over_declared"]
        # How much room the declared gate has over what a correct
        # implementation measurably needs. Below 1.0 is a failing gate.
        record["margin_over_requirement"] = (1.0 / scale) if scale > 0 else None

    return {
        "label": label,
        "workload_id": _workload_id(workload),
        "dims": dict(workload.dims),
        "dtype": workload.dtype,
        "repeats": repeats,
        "results": results,
        "production_noise_max_abs": noise,
        "ok": all(r["allclose_failures"] == 0 for r in results.values()),
        "worst_required_scale": max(
            r["required_scale_over_declared"] for r in results.values()
        ),
    }


# ── the populations ──────────────────────────────────────────────────────────


def _workload_id(workload) -> str:
    """A short, stable name for one measured configuration."""
    provenance = getattr(workload, "provenance", None)
    component = getattr(provenance, "component", None) if provenance else None
    dims = ",".join(f"{k}={v}" for k, v in sorted(workload.dims.items()))
    return f"{component or 'workload'}[{dims}]:{workload.dtype}"


def _observed_workloads(op):
    try:
        return op.benchmark_workloads(suite="qwen3_0_6b_observed")
    except Exception:
        return ()


def inventory_task(
    task: str,
    *,
    device: str = "cuda",
    repeats: int = DEFAULT_REPEATS,
    include_harvested: bool = True,
    artifact: Path | None = None,
) -> dict[str, Any]:
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(task)
    cases: list[dict[str, Any]] = []
    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            case(op, f"correctness {dict(workload.dims)}", workload, values,
                 repeats=repeats)
        )
    for workload in _observed_workloads(op):
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            case(op, f"observed {dict(workload.dims)}", workload, values,
                 repeats=repeats)
        )
    harvested = None
    if include_harvested and artifact is not None and artifact.is_file():
        harvested = _harvested_case(op, artifact, device=device, repeats=repeats)
        if harvested is not None:
            cases.append(harvested)

    return {
        "task": task,
        "device": device,
        "declared": {
            "tolerances": {k: list(v) for k, v in op.tolerances.items()},
            "multipliers": {
                k: list(v) for k, v in op.tolerance_multipliers.items()
            },
            "tolerance_hook": (
                op.tolerance_hook.describe()
                if hasattr(op.tolerance_hook, "describe")
                else getattr(op.tolerance_hook, "__name__", None)
            ),
        },
        "provenance": {
            "input_generator": getattr(op.make_inputs, "__name__", None),
            "input_seeding": (
                "deterministic, derived from the workload dims inside the "
                "declaration's make_inputs; no global seed is read"
            ),
            "runtime_forward": op.runtime_forward,
            "forward": op.forward,
            "snapshot_hash": _snapshot_hash(),
        },
        "cases": cases,
        "failing_cases": [c["label"] for c in cases if not c.get("ok", True)],
        "worst_required_scale": max(
            (c.get("worst_required_scale", 0.0) for c in cases), default=0.0
        ),
    }


def _snapshot_hash() -> str | None:
    try:
        from evograd.bench.workloads.qwen3.harvest.snapshot import load

        return load()["snapshot_hash"]
    except Exception:
        return None


def _harvested_case(op, artifact: Path, *, device: str, repeats: int):
    """The model's own tensors for this boundary, if the capture is present.

    Synthetic inputs and the model's differ in scale and in correlation
    structure. Running both through the identical comparison is what separates
    "BF16 cannot do this reduction" from "the synthetic distribution is not the
    model's".
    """
    builders = {
        "qwen3_swiglu_mlp": _harvested_mlp,
        "qwen3_attention": _harvested_attention,
        "qwen3_qkv_norm_rope": _harvested_qkv,
    }
    build = builders.get(op.name)
    if build is None:
        return None
    try:
        values, workload = build(op, artifact, device=device)
    except Exception as exc:  # a missing or stale capture is not a failure here
        return {
            "label": "harvested layer-14 (unavailable)",
            "error": f"{type(exc).__name__}: {exc}",
            "ok": True,
            "results": {},
            "worst_required_scale": 0.0,
        }
    record = case(op, "harvested layer-14 invocation", workload, values, repeats=repeats)
    record["source"] = str(artifact)
    return record


def _observed_workload_for(op):
    observed = _observed_workloads(op)
    return observed[0] if observed else op.benchmark[0]


def _harvested_mlp(op, artifact: Path, *, device: str):
    from evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp import derive_mlp_invocation

    payload, _ = derive_mlp_invocation(artifact, device=device)
    weights = payload["weights"]
    values = {
        "x": payload["input"].to(device),
        "gate_weight": weights["gate_weight"].to(device),
        "up_weight": weights["up_weight"].to(device),
        "down_weight": weights["down_weight"].to(device),
        "dout": payload["grad_output"].to(device),
    }
    return values, _observed_workload_for(op)


def _harvested_attention(op, artifact: Path, *, device: str):
    from evograd.bench.workloads.qwen3.levels.level2.attention import derive_attention_invocation

    payload, _ = derive_attention_invocation(artifact, device=device)
    values = {
        "q": payload["q"].to(device),
        "k": payload["k"].to(device),
        "v": payload["v"].to(device),
        "o_weight": payload["o_weight"].to(device),
        "dout": payload["grad_output"].to(device),
    }
    return values, _observed_workload_for(op)


def _harvested_qkv(op, artifact: Path, *, device: str):
    from evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope import derive_qkv_invocation

    payload, _ = derive_qkv_invocation(artifact, device=device)
    values = {
        name: tensor.to(device) if hasattr(tensor, "to") else tensor
        for name, tensor in payload["inputs"].items()
    }
    for grad_name in op.upstream_grad_names:
        values[grad_name] = payload["output_grads"][grad_name].to(device)
    return values, _observed_workload_for(op)


# ── does the disagreement scale with the reduction? ──────────────────────────


def scaling_study(
    task: str, *, device: str = "cuda", repeats: int = 2
) -> dict[str, Any]:
    """Hold the widths, sweep the token axis, and watch the requirement move.

    This is the measurement a dimension-aware tolerance stands on. If the
    required atol grows like the square root of the contraction length, the law
    is BF16 rounding accumulating over independent terms and a ``sqrt(N)`` hook
    is the honest shape for it. If it grows linearly, the errors are correlated
    and something else is wrong. If it does not grow, reduction length is not
    the cause and the tolerance should be a constant.
    """
    from evograd.opdecl.activity import Workload
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(task)
    base = _observed_workload_for(op)
    token_dim = "T"
    if token_dim not in base.dims:
        return {"task": task, "skipped": "no T dimension"}

    points = []
    for tokens in (64, 128, 256, 512, 1024, 2048):
        dims = dict(base.dims)
        dims[token_dim] = tokens
        workload = Workload(dims=dims, dtype=base.dtype, provenance=base.provenance)
        try:
            values = make_case_inputs(op, workload, device=device)
            record = case(op, f"T={tokens}", workload, values, repeats=repeats)
        except Exception as exc:
            points.append({"tokens": tokens, "error": f"{type(exc).__name__}: {exc}"})
            continue
        points.append(
            {
                "tokens": tokens,
                "per_result": {
                    name: {
                        "required_atol_at_declared_rtol": r["required_atol_at_declared_rtol"],
                        "max_abs_err": r["max_abs_err"],
                        "ref_absmax": r["ref_absmax"],
                        "reduction_length": r["reduction_length"],
                    }
                    for name, r in record["results"].items()
                },
            }
        )
    return {"task": task, "device": device, "sweep_dim": token_dim, "points": points,
            "exponents": _fit_exponents(points)}


def _fit_exponents(points) -> dict[str, float | None]:
    """log-log slope of required atol against reduction length, per result."""
    usable = [p for p in points if "per_result" in p]
    if len(usable) < 2:
        return {}
    names = usable[0]["per_result"].keys()
    out: dict[str, float | None] = {}
    for name in names:
        xs, ys = [], []
        for point in usable:
            record = point["per_result"][name]
            n = record["reduction_length"]
            y = record["required_atol_at_declared_rtol"]
            if n > 1 and y > 0:
                xs.append(math.log(n))
                ys.append(math.log(y))
        if len(xs) < 2:
            out[name] = None
            continue
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        out[name] = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
            if denom
            else None
        )
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def summarize(report: dict[str, Any]) -> str:
    lines = []
    for task_report in report["tasks"]:
        lines.append(f"=== {task_report['task']} ===")
        lines.append(f"  declared: {task_report['declared']}")
        for record in task_report["cases"]:
            if record.get("error"):
                lines.append(f"  {record['label']}: {record['error']}")
                continue
            verdict = "PASS" if record["ok"] else "FAIL"
            lines.append(
                f"  [{verdict}] {record['label']} [{record['dtype']}]  "
                f"worst required scale {record['worst_required_scale']:.3g}"
            )
            for name, r in record["results"].items():
                mark = " " if r["allclose_failures"] == 0 else "*"
                lines.append(
                    f"   {mark}{name:<16} scale={r['required_scale_over_declared']:.3g} "
                    f"fail={r['allclose_failures']}/{r['elements']} "
                    f"({r['failure_fraction']:.2e}) "
                    f"max_abs={r['max_abs_err']:.3e} "
                    f"need_atol={r['required_atol_at_declared_rtol']:.3e} "
                    f"(declared {r['declared_atol']:.3e}) "
                    f"|ref|max={r['ref_absmax']:.3e} rms={r['ref_rms']:.3e} "
                    f"N={r['reduction_length']} over {r['reduced_dims']} "
                    f"noise={record['production_noise_max_abs'][name]:.3e}"
                )
        lines.append("")
    for study in report.get("scaling", []):
        if "points" not in study:
            continue
        lines.append(f"=== scaling {study['task']} (sweep {study['sweep_dim']}) ===")
        for name, slope in study["exponents"].items():
            lines.append(
                f"  {name:<16} d log(required atol) / d log(N) = "
                + ("n/a" if slope is None else f"{slope:.3f}")
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "scaling"):
        sp = sub.add_parser(name)
        sp.add_argument("--task", action="append", default=None, choices=TASKS)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
        sp.add_argument("--report", type=Path, default=None)
        sp.add_argument(
            "--artifact",
            type=Path,
            default=Path("results/qwen3-level4/layer14.pt"),
            help="harvested capture to take the model's own tensors from",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    tasks = tuple(args.task) if args.task else TASKS

    report: dict[str, Any] = {
        "schema_version": "evograd-qwen3-l2-calibration/1",
        "command": args.command,
        "device": args.device,
        "repeats": args.repeats,
        "compared": (
            "op.forward (declared float32-accumulated oracle) vs "
            "op.runtime_forward (the exact Transformers spelling); identical "
            "inputs and identical upstream gradients, both differentiated by "
            "torch.autograd"
        ),
        "tasks": [],
    }
    if args.command == "inventory":
        report["tasks"] = [
            inventory_task(
                task, device=args.device, repeats=args.repeats,
                artifact=args.artifact,
            )
            for task in tasks
        ]
    else:
        report["tasks"] = []
        report["scaling"] = [
            scaling_study(task, device=args.device, repeats=max(args.repeats // 2, 2))
            for task in tasks
        ]

    print(summarize(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
