"""Two-process, declaration-driven Nsight Compute profiling."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from evograd.opdecl.activity import OpDecl, Workload

DEFAULT_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)

_FRIENDLY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": (
        "long_scoreboard_stall_ratio"
    ),
    "smsp__sass_inst_executed_op_local_ld.sum": "local_loads",
    "smsp__sass_inst_executed_op_local_st.sum": "local_stores",
}


@dataclass(frozen=True)
class ProfileResult:
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    kernels: tuple[dict, ...] = ()
    report_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "metrics": self.metrics,
            "kernels": list(self.kernels),
            "report_path": self.report_path,
            "error": self.error,
        }


def representative_workload(op: OpDecl) -> Workload:
    if not op.benchmark:
        raise ValueError(f"{op.name}: no benchmark workloads to profile")
    return max(
        op.benchmark,
        key=lambda workload: (
            sum(workload.dims.values()),
            max(workload.dims.values()),
        ),
    )


def _script(
    op: OpDecl,
    candidate: Path,
    workload: Workload,
    inputs_path: Path,
    *,
    warmup: int | None,
) -> str:
    if warmup is not None:
        values = "values = make_case_inputs(op, workload, device='cuda')"
        invocation = (
            f"for _ in range({warmup}):\n"
            "    y, saved = forward(*args)\n"
            "    backward(dout, saved, **kwargs)\n"
            "torch.cuda.synchronize()\n"
            f"torch.save(values, {str(inputs_path)!r})"
        )
    else:
        values = f"values = torch.load({str(inputs_path)!r}, map_location='cuda')"
        invocation = (
            "y, saved = forward(*args)\n"
            "backward(dout, saved, **kwargs)\n"
            "torch.cuda.synchronize()"
        )
    return f"""import importlib.util
import torch
from dataclasses import replace

from evograd.opdecl.activity import Workload
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import upstream_grad_values
from evograd.opdecl.inputs import make_case_inputs
from evograd.ops import get_op, load_op

op = load_op({op.declaration!r}) if {bool(op.declaration)!r} else get_op({op.name!r})
op = replace(op, forward={op.forward!r})
workload = Workload(dims={workload.dims!r}, dtype={workload.dtype!r})
spec = importlib.util.spec_from_file_location("evograd_ncu_candidate", {str(candidate)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forward, backward = lookup_pair(op, module)
{values}
dout = upstream_grad_values(op, values)
args = [values.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
kwargs = backward_inactive_kwargs(op, backward, values)
{invocation}
"""


def _number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text in ("n/a", "N/A"):
        return None
    scale = 1.0
    suffixes = {"K": 1e3, "M": 1e6, "G": 1e9}
    if text[-1:] in suffixes:
        scale = suffixes[text[-1]]
        text = text[:-1]
    text = text.rstrip("%")
    try:
        return float(text) * scale
    except ValueError:
        return None


def _parse_csv(output: str) -> tuple[dict[str, float], tuple[dict, ...]]:
    lines = [line for line in output.splitlines() if line.lstrip().startswith('"')]
    if not lines:
        return {}, ()
    header_index = next(
        (index for index, line in enumerate(lines) if '"Metric Name"' in line),
        None,
    )
    if header_index is None:
        return {}, ()
    reader = csv.DictReader(lines[header_index:])
    kernels = []
    values: dict[str, list[float]] = {}
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        value = _number(row.get("Metric Value") or "")
        if not name or value is None:
            continue
        values.setdefault(name, []).append(value)
        kernels.append(
            {
                "kernel": row.get("Kernel Name") or row.get("Kernel") or "",
                "metric": name,
                "value": value,
                "unit": row.get("Metric Unit") or "",
            }
        )
    aggregate = {}
    for raw_name, observations in values.items():
        friendly = _FRIENDLY.get(raw_name, raw_name)
        aggregate[friendly] = sum(observations) / len(observations)
    aggregate["register_spilling"] = float(
        aggregate.get("local_loads", 0.0) > 0
        or aggregate.get("local_stores", 0.0) > 0
    )
    return aggregate, tuple(kernels)


def run_ncu_profile(
    op: OpDecl,
    candidate: Path,
    *,
    workload: Workload | None = None,
    warmup: int = 5,
    timeout: int = 120,
    ncu_bin: str = "ncu",
) -> ProfileResult:
    workload = workload or representative_workload(op)
    binary = shutil.which(ncu_bin)
    if binary is None:
        return ProfileResult(ok=False, error=f"{ncu_bin!r} was not found on PATH")
    candidate = candidate.resolve()
    subprocess_env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    subprocess_env["PYTHONPATH"] = source_root + (
        os.pathsep + subprocess_env["PYTHONPATH"]
        if subprocess_env.get("PYTHONPATH")
        else ""
    )
    with tempfile.TemporaryDirectory(prefix="evograd_ncu_") as raw_tmp:
        tmp = Path(raw_tmp)
        inputs = tmp / "inputs.pt"
        warmup_script = tmp / "warmup.py"
        warmup_script.write_text(
            _script(op, candidate, workload, inputs, warmup=warmup),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(warmup_script)],
                text=True,
                capture_output=True,
                timeout=timeout,
                env=subprocess_env,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"warmup timed out after {timeout}s")
        if completed.returncode != 0 or not inputs.is_file():
            return ProfileResult(
                ok=False,
                error=f"warmup failed: {completed.stderr[-2000:]}",
            )

        profiled_script = tmp / "profiled.py"
        profiled_script.write_text(
            _script(op, candidate, workload, inputs, warmup=None),
            encoding="utf-8",
        )
        report_base = tmp / "profile"
        command = [
            binary,
            "--csv",
            "--page",
            "details",
            "--metrics",
            ",".join(DEFAULT_METRICS),
            "--target-processes",
            "all",
            "--force-overwrite",
            "--export",
            str(report_base),
            sys.executable,
            str(profiled_script),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=subprocess_env,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"ncu timed out after {timeout}s")
        if completed.returncode != 0:
            return ProfileResult(
                ok=False,
                error=f"ncu failed: {(completed.stderr or completed.stdout)[-2000:]}",
            )
        metrics, kernels = _parse_csv(completed.stdout + "\n" + completed.stderr)
        report = report_base.with_suffix(".ncu-rep")
        parse_output = completed.stdout + "\n" + completed.stderr
        if not metrics and report.is_file():
            try:
                imported = subprocess.run(
                    [binary, "--import", str(report), "--csv", "--page", "details"],
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    env=subprocess_env,
                )
            except subprocess.TimeoutExpired:
                imported = None
            if imported is not None and imported.returncode == 0:
                parse_output = imported.stdout + "\n" + imported.stderr
                metrics, kernels = _parse_csv(parse_output)
        persistent = None
        if report.is_file():
            fd, name = tempfile.mkstemp(prefix="evograd_", suffix=".ncu-rep")
            os.close(fd)
            shutil.copy2(report, name)
            persistent = name
        if not metrics:
            return ProfileResult(
                ok=False,
                report_path=persistent,
                error=(
                    "ncu completed but no requested metrics could be parsed; "
                    f"output tail: {parse_output[-8000:]}"
                ),
            )
        return ProfileResult(
            ok=True,
            metrics=metrics,
            kernels=kernels,
            report_path=persistent,
        )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--declaration", default=None)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    from evograd.ops import get_op, load_op

    op = load_op(args.declaration) if args.declaration else get_op(args.op)
    if op.name != args.op:
        parser.error(f"declaration name {op.name!r} does not match --op {args.op!r}")
    result = run_ncu_profile(
        op, args.candidate, timeout=args.timeout
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1
