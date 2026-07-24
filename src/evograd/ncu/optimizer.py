"""Profiling-grounded diagnosis and code generation."""

from __future__ import annotations

import json
import subprocess

from evograd.ncu.roofline import RooflineResult, analyze, format_roofline
from evograd.pipelines.shared.llm_client import generate_with_openai_compatible_api
from evograd.pipelines.shared.runner import strip_code_fence

_DIAGNOSE_SYSTEM = """You are a GPU performance expert diagnosing a Triton
forward/backward program from Nsight Compute metrics. Cite the supplied numbers,
separate evidence from inference, and recommend only concrete Triton-level
changes. Return a JSON array of bottlenecks, root causes, and fixes."""

_GENERATE_SYSTEM = """You are a Triton performance engineer. Rewrite the
complete program using only fixes supported by the NCU diagnosis. Preserve the
forward/backward signatures, saved-state contract, numerical semantics, dtypes,
and all shapes. Return the complete Python source in one code block."""


def _gpu_summary() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,multiprocessor_count",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except Exception:
        pass
    return "GPU details unavailable; use the NCU metrics as authoritative."


def optimize_from_profile(
    code: str,
    metrics: dict[str, float],
    benchmark_metrics: dict,
    *,
    kernel_metrics: tuple[dict, ...] = (),
    model: str,
    api_base: str,
    api_key: str | None,
    timeout: int = 360,
    skip_at_roofline_pct: float = 95.0,
) -> tuple[str | None, dict]:
    """Return ``(generated_code, pass_record)``; code is None when skipped."""
    roofline: RooflineResult = analyze(metrics, threshold=skip_at_roofline_pct)
    record = {
        "roofline": {
            "bottleneck": roofline.bottleneck,
            "efficiency_pct": roofline.efficiency_pct,
            "at_roofline": roofline.at_roofline,
            "compute_pct": roofline.compute_pct,
            "memory_pct": roofline.memory_pct,
            "notes": list(roofline.notes),
        },
        "ncu_metrics": metrics,
        "kernel_metrics": list(kernel_metrics),
    }
    if roofline.at_roofline:
        record["outcome"] = (
            f"skipped: {roofline.efficiency_pct:.1f}% SOL meets "
            f"{skip_at_roofline_pct:.1f}% threshold"
        )
        return None, record

    diagnosis_prompt = f"""GPU:
{_gpu_summary()}

Deterministic roofline triage:
{format_roofline(roofline)}

NCU aggregate metrics:
{json.dumps(metrics, indent=2, sort_keys=True)}

Per-kernel raw metrics:
{json.dumps(list(kernel_metrics), indent=2, sort_keys=True)}

Measured evaluator metrics:
{json.dumps(benchmark_metrics, indent=2, sort_keys=True)}

Program:
```python
{code}
```

Return JSON only. Examine compute/memory SOL, occupancy, waves, registers,
spilling, and long-scoreboard stalls. Do not recommend a rewrite unsupported
by those signals."""
    diagnosis = generate_with_openai_compatible_api(
        prompt=diagnosis_prompt,
        system_message=_DIAGNOSE_SYSTEM,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=8000,
        temperature=0.2,
        timeout=timeout,
    )
    record["diagnosis"] = diagnosis

    generation_prompt = f"""GPU:
{_gpu_summary()}

Roofline:
{format_roofline(roofline)}

Current evaluator metrics:
{json.dumps(benchmark_metrics, indent=2, sort_keys=True)}

Diagnosis:
{diagnosis}

Current complete program:
```python
{code}
```

Apply the highest-confidence fixes. The new program is retained only if the
same correctness and performance evaluator gives it a strictly higher score."""
    response = generate_with_openai_compatible_api(
        prompt=generation_prompt,
        system_message=_GENERATE_SYSTEM,
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=20000,
        temperature=0.2,
        timeout=timeout,
    )
    generated = strip_code_fence(response)
    record["generation_response"] = response
    if generated.strip() == code.strip():
        record["outcome"] = "generated code was unchanged"
        return None, record
    return generated, record
