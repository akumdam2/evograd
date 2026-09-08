"""Measuring the tolerance a Qwen3 Level-1 primitive actually needs.

Generic in shape -- it calibrates any task that declares a
``runtime_forward`` -- and Qwen-specific in what it is pointed at. It reports
what the observed cases require; it does not decide the declared tolerance,
which stays in the declaration where a reader can see it.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import torch

from evograd.benchmark import get_task
from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import Level1Error
from evograd.evaluation.workloads.qwen3_0_6b.level3.replay import required_tolerance
def _active_names(op) -> tuple[str, ...]:
    from evograd.opdecl.activity import Active

    return tuple(a.name for a in op.args if isinstance(a, Active))


def _pair_pass(op, forward, values):
    from evograd.opdecl.inputs import as_output_tuple, upstream_grad_values

    active = _active_names(op)
    leaves = {name: values[name].detach().clone().requires_grad_(True) for name in active}
    args = [leaves.get(a.name, values.get(a.name, getattr(a, "default", None))) for a in op.args]
    outputs = as_output_tuple(op, forward(*args))
    douts = upstream_grad_values(op, values)
    torch.autograd.backward(outputs, douts if isinstance(douts, tuple) else (douts,))
    results = {out.name: o.detach().clone() for out, o in zip(op.outputs, outputs)}
    results.update({f"d{name}": leaves[name].grad.detach().clone() for name in active})
    return results


def _result_names(op) -> tuple[str, ...]:
    return op.output_names + tuple(f"d{name}" for name in _active_names(op))


def run_calibration(op_name: str, *, device: str = "cuda", repeats: int = 3) -> dict[str, Any]:
    """Measure what a correct implementation needs, per result.

    The disagreement between the declared oracle and ``runtime_forward`` -- the
    spelling the model runs -- is the smallest error any correct implementation
    can have with the oracle, so a gate that rejects it rejects correct code.
    """
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
    from evograd.benchmark import get_task

    op = get_task(op_name)
    if not op.runtime_forward:
        raise Level1Error(f"{op_name} has no runtime_forward to calibrate against")
    reference_fn = resolve_forward(op)
    production_fn = resolve_runtime_forward(op)
    names = _result_names(op)

    workloads = list(op.correctness) + list(op.benchmark_workloads("qwen3_0_6b_observed"))
    cases = []
    for workload in workloads:
        values = make_case_inputs(op, workload, device=device)
        multipliers = {
            name: tuple(
                t / b if b else 1.0
                for t, b in zip(op.tolerance_for(workload, name), op.tolerance_for(workload))
            )
            for name in names
        }
        reference = _pair_pass(op, reference_fn, values)
        production = _pair_pass(op, production_fn, values)
        results = {
            name: required_tolerance(production[name], reference[name], multipliers[name])
            for name in names
        }
        noise = {name: 0.0 for name in names}
        for _ in range(max(repeats - 1, 0)):
            again = _pair_pass(op, production_fn, values)
            for name in names:
                noise[name] = max(
                    noise[name],
                    required_tolerance(again[name], production[name], multipliers[name])[
                        "required_t"
                    ],
                )
        cases.append(
            {
                "label": f"{workload.dims}",
                "dtype": workload.dtype,
                "observed": workload.provenance is not None
                and workload.provenance.model == "qwen3_0_6b",
                "results": results,
                "production_noise_required_t": noise,
            }
        )

    def worst(subset):
        return {
            name: max((c["results"][name]["required_t"] for c in subset), default=0.0)
            for name in names
        }

    return {
        "schema_version": "evograd-qwen3-level1-tolerance/1",
        "task": op_name,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|))"
        ),
        "compared": "declared forward vs runtime_forward",
        "cases": cases,
        "worst_required_t": {
            "overall": worst(cases),
            "bfloat16": worst([c for c in cases if c["dtype"] == "bfloat16"]),
            "float32": worst([c for c in cases if c["dtype"] == "float32"]),
            "observed_only": worst([c for c in cases if c["observed"]]),
        },
        "declared_tolerances": op.tolerances,
    }


def summarize_calibration(report: dict[str, Any]) -> str:
    lines = [
        f"tolerance calibration for {report['task']}",
        f"  metric: {report['metric']}",
        f"  comparing: {report['compared']}",
        "",
    ]
    for case in report["cases"]:
        tag = " (observed)" if case["observed"] else ""
        lines.append(f"  {case['label']}  [{case['dtype']}]{tag}")
        for name, record in case["results"].items():
            lines.append(
                f"    {name:<6} required_t {record['required_t']:.3e}   "
                f"rel_vs_scale {record['max_rel_err_vs_scale']:.3e}   "
                f"noise {case['production_noise_required_t'][name]:.3e}"
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
