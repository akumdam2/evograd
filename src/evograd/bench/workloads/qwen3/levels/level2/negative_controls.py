"""Does the calibrated gate still reject a wrong kernel?

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.negative_controls \
        --report results/qwen3-level4/l2-negative-controls.json

A tolerance that accepts everything is not a tolerance. Widening one to admit
BF16's real behaviour at 4096-token reductions is only defensible if the widened
gate still catches the errors an implementation actually makes, so every
tolerance family that moved gets an injected fault and has to fail.

Two kinds of fault, because they fail differently:

* **scaled** -- every element of one result multiplied by ``1 + eps``. This is
  the shape of a systematic error: a missing float32 accumulation, a wrong
  scale factor, a dtype cast in the wrong place. The gate's ``rtol`` is what
  catches it, and ``rtol`` was deliberately left untouched by the calibration.
* **dropped-term** -- one slice removed from a reduction, which is what an
  off-by-one bound or a mis-sized tile does. Its size is set by the reduction,
  not by the value, so it is exactly the error a reduction-scaled ``atol`` is
  most at risk of hiding. That is why it is here.

The controls run against the *production* results, so a fault is measured
against a spelling the gate currently accepts -- the question asked is "would
this defect have been caught", not "is BF16 different from float32".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

TASKS = ("qwen3_qkv_norm_rope", "qwen3_attention", "qwen3_swiglu_mlp")

#: Relative perturbations to try, smallest first. The reported number is the
#: smallest one the gate catches -- its detection floor for that result.
SCALE_FAULTS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25)


def _gate(op, workload, name, actual, expected) -> bool:
    """True when the declared gate accepts, i.e. the fault went undetected."""
    atol, rtol = op.tolerance_for(workload, name)
    return bool(
        torch.allclose(
            actual.detach().float(), expected.detach().float(), atol=atol, rtol=rtol
        )
    )


def scale_fault_floor(op, workload, name, clean, reference) -> dict[str, Any]:
    """The smallest uniform relative error on ``name`` that the gate rejects."""
    for eps in SCALE_FAULTS:
        if not _gate(op, workload, name, clean * (1.0 + eps), reference):
            return {"fault": "scaled", "detected_at": eps, "tried": list(SCALE_FAULTS)}
    return {"fault": "scaled", "detected_at": None, "tried": list(SCALE_FAULTS)}


def dropped_term_fault(op, workload, name, values, reference) -> dict[str, Any]:
    """Recompute one gradient with a token removed, and see if the gate notices.

    Not a synthetic perturbation: the whole forward and backward run again on a
    batch one token shorter, so the fault has the correlation structure a real
    off-by-one produces rather than the uniform shape of a scaled error.
    """
    from evograd.bench.workloads.qwen3.levels.level2.calibrate import production_results

    token_dim_arg = next(
        (
            arg
            for arg in op.active_args()
            if "T" in _dims_of(arg) and "B" in _dims_of(arg)
        ),
        None,
    )
    if token_dim_arg is None:
        return {"fault": "dropped_token", "applicable": False}

    truncated = dict(values)
    activation = values[token_dim_arg.name]
    axis = _dims_of(token_dim_arg).index("T")
    index = [slice(None)] * activation.dim()
    index[axis] = slice(0, activation.shape[axis] - 1)
    # Zero the last token instead of reshaping: the reduction then sums one
    # fewer contributing term while every declared shape stays exactly as the
    # contract requires, so the only thing that changed is the sum.
    damaged = activation.detach().clone()
    last = [slice(None)] * activation.dim()
    last[axis] = slice(activation.shape[axis] - 1, activation.shape[axis])
    damaged[tuple(last)] = 0.0
    truncated[token_dim_arg.name] = damaged

    faulty = production_results(op, truncated)
    caught = {}
    for result_name in (*op.output_names, *op.grad_names()):
        caught[result_name] = not _gate(
            op, workload, result_name, faulty[result_name], reference[result_name]
        )
    return {
        "fault": "dropped_token",
        "applicable": True,
        "zeroed": f"{token_dim_arg.name}[..., T={activation.shape[axis] - 1}, ...]",
        "detected": caught,
    }


def _dims_of(arg) -> list[str]:
    shape = getattr(arg, "shape", None)
    if not shape:
        return []
    inner = shape.strip()[1:-1].strip()
    return [p.strip() for p in inner.split(",")] if inner else []


def run_task(task: str, *, device: str = "cuda") -> dict[str, Any]:
    from evograd.bench.workloads.qwen3.levels.level2.calibrate import (
        production_results,
        reference_results,
    )
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(task)
    workloads = list(op.benchmark_workloads(suite="qwen3_0_6b_observed"))
    out: list[dict[str, Any]] = []
    for workload in workloads:
        values = make_case_inputs(op, workload, device=device)
        reference = reference_results(op, values)
        clean = production_results(op, values)
        accepted_clean = {
            name: _gate(op, workload, name, clean[name], reference[name])
            for name in (*op.output_names, *op.grad_names())
        }
        scaled = {
            name: scale_fault_floor(op, workload, name, clean[name], reference[name])
            for name in (*op.output_names, *op.grad_names())
        }
        dropped = dropped_term_fault(op, workload, None, values, reference)
        out.append(
            {
                "dims": dict(workload.dims),
                "dtype": workload.dtype,
                "clean_accepted": accepted_clean,
                "clean_all_accepted": all(accepted_clean.values()),
                "scaled_fault": scaled,
                "dropped_token_fault": dropped,
                "tolerances": {
                    name: list(op.tolerance_for(workload, name))
                    for name in (*op.output_names, *op.grad_names())
                },
            }
        )
    return {"task": task, "cases": out}


def summarize(report: dict[str, Any]) -> str:
    lines = []
    for task in report["tasks"]:
        lines.append(f"=== {task['task']} ===")
        for record in task["cases"]:
            lines.append(
                f"  {record['dims']} [{record['dtype']}]  "
                f"clean accepted: {record['clean_all_accepted']}"
            )
            dropped = record["dropped_token_fault"].get("detected", {})
            for name, entry in record["scaled_fault"].items():
                atol, rtol = record["tolerances"][name]
                floor = entry["detected_at"]
                lines.append(
                    f"    {name:<16} atol={atol:.3e} rtol={rtol:.3e}  "
                    f"scaled fault caught at {'>25%' if floor is None else f'{floor:.1%}'}  "
                    f"dropped-token caught: {dropped.get(name, 'n/a')}"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.negative_controls",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", action="append", default=None, choices=TASKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    report = {
        "schema_version": "evograd-qwen3-l2-negative-control/1",
        "device": args.device,
        "scale_faults_tried": list(SCALE_FAULTS),
        "tasks": [run_task(t, device=args.device) for t in (args.task or TASKS)],
    }
    print(summarize(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
