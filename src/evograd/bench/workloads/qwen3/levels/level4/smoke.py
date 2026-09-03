"""Run the canonical Qwen3 training step once and describe what happened.

Structured so a later observation stage reuses the parts rather than the whole:
``build_model``/``make_inputs``/``training_step`` come from :mod:`.model` and can
be driven under a hook, a profiler or a tracer; everything this module adds is
verification and reporting around them.

The verification is deliberately after-the-fact. Asking for BF16 SDPA proves
nothing -- what the report carries is what the built model says it is, what the
gradients turned out to be, and, for every trainable parameter, whether one
arrived at all and whether it is finite.
"""

from __future__ import annotations

import platform
import subprocess
import time
import traceback
from typing import Any

import torch

from .model import (
    build_model,
    check_effective_settings,
    effective_settings,
    make_inputs,
    require_transformers,
    training_step,
)
from .report import STATUS_FAILED, STATUS_OK, SmokeReport
from .spec import CANONICAL, WorkloadSpec


def _nvidia_smi_driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    line = out.stdout.strip().splitlines()
    return line[0].strip() if out.returncode == 0 and line else None


def environment_info(spec: WorkloadSpec) -> dict[str, Any]:
    """Versions and hardware. Collected without importing Transformers, so a
    report can still be written when the optional dependency is what failed."""
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": None,
        "gpu_capability": None,
        "gpu_total_memory_bytes": None,
        "cuda_driver_version": None,
    }
    if spec.device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info["gpu_name"] = props.name
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        info["gpu_total_memory_bytes"] = props.total_memory
        # The installed driver, distinct from the runtime `cuda_version` above:
        # a mismatch between them explains a whole class of launch failures.
        # torch exposes no accessor for it (2.11 dropped
        # `_C._cuda_getDriverVersion`), so ask the tool that knows. A machine
        # without `nvidia-smi` on PATH simply reports None.
        info["cuda_driver_version"] = _nvidia_smi_driver_version()
    return info


def workload_info(spec: WorkloadSpec) -> dict[str, Any]:
    return {
        "workload_id": spec.workload_id,
        "workload_hash": spec.workload_hash,
        "canonical": spec.is_canonical,
        "canonical_workload_id": CANONICAL.workload_id,
        "model_name": spec.model_name,
        "config": spec.arch,
        "config_hash": spec.config_hash,
        "batch_size": spec.batch_size,
        "seq_len": spec.seq_len,
        "token_count": spec.token_count,
        "dtype": spec.dtype,
        "device": spec.device,
        "requested_attn_implementation": spec.attn_implementation,
        "use_cache": spec.use_cache,
        "gradient_checkpointing": spec.gradient_checkpointing,
        "training": spec.training,
        "seed": spec.seed,
    }


def gradient_coverage(model) -> dict[str, Any]:
    """Which trainable parameters received a gradient, and whether it is finite.

    Reported per parameter name rather than as a count alone: a step where 309
    of 310 tensors got a gradient is a specific bug in a specific module, and the
    count on its own does not say which.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    with_grad = [(name, p) for name, p in trainable if p.grad is not None]
    missing = [name for name, p in trainable if p.grad is None]
    non_finite = [
        name for name, p in with_grad if not bool(torch.isfinite(p.grad).all().item())
    ]
    return {
        "trainable_params": len(trainable),
        "trainable_elements": sum(p.numel() for _, p in trainable),
        "params_with_grad": len(with_grad),
        "missing_grad_params": missing,
        "non_finite_grad_params": non_finite,
        "grads_all_finite": not non_finite and not missing,
    }


def run_smoke(spec: WorkloadSpec | None = None) -> SmokeReport:
    """Execute one training step and return the report. Never raises for a
    run-time failure -- the failure is the report's payload."""
    spec = (spec or CANONICAL).validate()
    report = SmokeReport(workload=workload_info(spec), environment=environment_info(spec))

    try:
        require_transformers()
        if spec.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "the canonical workload runs on CUDA and no CUDA device is visible; "
                "allocate a GPU node, or pass --device cpu for a (non-canonical) "
                "debug run"
            )
        if spec.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        model = build_model(spec)
        input_ids, labels = make_inputs(spec)

        effective = effective_settings(model, spec)
        effective["input_ids_dtype"] = str(input_ids.dtype)
        effective["input_ids_device"] = str(input_ids.device).split(":")[0]
        effective["input_ids_shape"] = list(input_ids.shape)
        # A checksum makes "deterministic synthetic inputs" verifiable across runs
        # without shipping the tensor.
        effective["input_ids_checksum"] = int(input_ids.sum().item())
        effective["labels_match_input_ids"] = bool(torch.equal(input_ids, labels))
        report.effective = effective

        problems = check_effective_settings(effective, spec)
        if problems:
            raise RuntimeError(
                "the built model does not match the requested workload: "
                + "; ".join(problems)
            )

        outputs = training_step(model, input_ids, labels)
        loss = outputs.loss
        if spec.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        report.effective["loss_dtype"] = str(loss.dtype)
        report.effective["logits_dtype"] = str(outputs.logits.dtype)
        report.effective["returned_past_key_values"] = outputs.past_key_values is not None

        loss_value = float(loss.detach().float().item())
        coverage = gradient_coverage(model)
        report.result = {
            "loss": loss_value,
            "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
            **coverage,
        }
        report.diagnostics = {
            "note": "diagnostic only -- a single unwarmed step, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if spec.device.startswith("cuda") else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if spec.device.startswith("cuda") else None
            ),
        }
        if report.result["loss_is_finite"] is False:
            raise RuntimeError(f"loss is not finite: {loss_value}")
        if coverage["missing_grad_params"]:
            raise RuntimeError(
                f"{len(coverage['missing_grad_params'])} trainable parameters received "
                f"no gradient, first: {coverage['missing_grad_params'][:5]}"
            )
        if coverage["non_finite_grad_params"]:
            raise RuntimeError(
                f"{len(coverage['non_finite_grad_params'])} parameter gradients are not "
                f"finite, first: {coverage['non_finite_grad_params'][:5]}"
            )
        report.status = STATUS_OK
    except Exception as exc:  # the failure is the result, not an interruption
        report.status = STATUS_FAILED
        report.failure = f"{type(exc).__name__}: {exc}"
        report.diagnostics.setdefault("traceback", traceback.format_exc())
    return report
