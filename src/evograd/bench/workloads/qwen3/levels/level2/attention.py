"""Derive and verify ``qwen3_attention`` from the canonical Layer-14 artifact.

    # describe the derived invocation (metadata only -- no tensors are written)
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.attention derive \
        --source results/qwen3-level4/layer14.pt \
        --metadata-out results/qwen3-level4/layer14-attention.json

    # check both spellings against what Transformers computed
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.attention verify \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-attention-verify.json

    # measure what tolerance a correct implementation actually needs
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.attention calibrate \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/qwen3_attention-tolerance.json

Like the MLP task, nothing here writes a second tensor file. ``layer14.pt``
already holds every number this boundary needs; the invocation is re-derived by
replaying that artifact, which takes about a second, and only JSON is written.

The boundary is captured at two points inside the replay: the
``scaled_dot_product_attention`` call, which is where q, k and v are visible in
the exact layout the model presented them, and the ``o_proj`` module, which is
where the boundary's output and its upstream gradient are. The gradients of q, k
and v come from tensor hooks on those three tensors -- each is consumed only by
this attention call, so its gradient *is* the boundary's gradient.

Layout is preserved deliberately. q, k and v are non-contiguous head-major views,
and a copy that quietly made them contiguous would describe a different call, so
the CPU copies are allocated with ``empty_strided`` at the observed strides.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from ...levels.level3.artifact import ArtifactError, content_hash_over, identity_hash_over, load_canonical
from ...levels.level3.replay import required_tolerance
from ...harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-qwen3-attention-task/1"
REPORT_SCHEMA = "evograd-qwen3-attention-verify/1"

TASK_NAME = "qwen3_attention"

CONTENT_KEYS = ("q", "k", "v", "o_weight", "out", "grad_output", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "sdpa_config_id",
    "o_proj_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

RESULT_NAMES = ("out", "dq", "dk", "dv", "do_weight")


class AttentionExtractionError(RuntimeError):
    """The attention invocation could not be derived, or does not check out."""


# --------------------------------------------------------------------------
# layout-preserving capture
# --------------------------------------------------------------------------


def preserve_layout_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """A CPU copy with the source's exact strides.

    ``.contiguous()`` here would be a silent change of contract: q, k and v
    reach SDPA as head-major transposes, and a contiguous copy is a different
    memory access pattern with different performance and different kernel
    dispatch.
    """
    source = tensor.detach()
    out = torch.empty_strided(
        source.shape, source.stride(), dtype=source.dtype, device="cpu"
    )
    out.copy_(source)
    return out


class _AttentionCapture:
    def __init__(self) -> None:
        self.q = self.k = self.v = None
        self.out = None
        self.grad_output = None
        self.grads: dict[str, torch.Tensor] = {}
        self.sdpa_calls = 0
        self.proj_calls = 0
        self.attrs: dict[str, Any] = {}
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [n for n in ("q", "k", "v", "out", "grad_output") if getattr(self, n) is None]
        missing += [n for n in ("dq", "dk", "dv") if n not in self.grads]
        if missing:
            raise AttentionExtractionError(
                f"attention capture is incomplete, missing {missing}"
            )


@contextmanager
def capture_attention(layer: torch.nn.Module) -> Iterator[_AttentionCapture]:
    """Watch the SDPA call and the output projection inside one decoder layer."""
    capture = _AttentionCapture()
    o_proj = layer.get_submodule("self_attn.o_proj")
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa_wrapper(query, key, value, *args, **kwargs):
        capture.sdpa_calls += 1
        if capture.sdpa_calls > 1:
            raise AttentionExtractionError(
                "SDPA ran more than once inside one extraction; the task is "
                "defined as a single invocation"
            )
        for name, tensor in (("q", query), ("k", key), ("v", value)):
            setattr(capture, name, preserve_layout_cpu(tensor))
            if tensor.requires_grad:
                capture.handles.append(
                    tensor.register_hook(
                        lambda grad, name=name: capture.grads.__setitem__(
                            f"d{name}", preserve_layout_cpu(grad)
                        )
                    )
                )
        capture.attrs = {
            "dropout_p": float(kwargs.get("dropout_p", 0.0)),
            "is_causal": bool(kwargs.get("is_causal", False)),
            "scale": kwargs.get("scale"),
            "enable_gqa": bool(kwargs.get("enable_gqa", False)),
            "attn_mask_provided": kwargs.get("attn_mask", None) is not None,
        }
        return original(query, key, value, *args, **kwargs)

    def post_hook(module, args, kwargs, output):
        capture.proj_calls += 1
        tensor = output[0] if isinstance(output, tuple) else output
        capture.out = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_output", preserve_layout_cpu(grad))
                )
            )
        return None

    handle = o_proj.register_forward_hook(post_hook, with_kwargs=True)
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper
    try:
        yield capture
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        handle.remove()
        for h in capture.handles:
            h.remove()
        capture.handles.clear()


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def derive_attention_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay Layer 14 and capture the attention boundary inside it."""
    from ...levels.level3.replay import prepare_layer

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AttentionExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_attention(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    o_proj = layer.get_submodule("self_attn.o_proj")
    if o_proj.weight.grad is None:
        raise AttentionExtractionError("o_proj.weight has no gradient after backward")
    o_weight = preserve_layout_cpu(o_proj.weight)
    capture.grads["do_weight"] = preserve_layout_cpu(o_proj.weight.grad)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.self_attn"
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "sdpa_config_id": harvest["config_id"],
            "o_proj_config_id": harvest["supporting"]["output_projection"]["config_id"],
            "frequency": harvest["frequency"],
            "layer_index": identity["layer_index"],
            "module_path": module_path,
            "source_content_hash": payload["content_hash"],
            "source_artifact_hash": payload["artifact_hash"],
            "provenance_kind": "derived_from_verified_replay",
        },
        "provenance_chain": [
            f"canonical workload {identity['workload_id']}",
            f"harvest manifest {identity['manifest_hash']}",
            f"{identity['module_path']} (event ordinal {identity['event_ordinal']})",
            f"Layer-14 artifact {payload['artifact_hash']}",
            f"scaled_dot_product_attention + o_proj at {module_path}",
            TASK_NAME,
        ],
        "attrs": capture.attrs,
        "q": capture.q,
        "k": capture.k,
        "v": capture.v,
        "o_weight": o_weight,
        "out": capture.out,
        "grad_output": capture.grad_output,
        "grads": dict(capture.grads),
    }
    task_payload["content_hash"] = content_hash_over(task_payload, CONTENT_KEYS)
    task_payload["derivation_hash"] = identity_hash_over(task_payload, IDENTITY_KEYS)

    def meta(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "contiguous": bool(tensor.is_contiguous()),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": task_payload["identity"],
        "provenance_chain": task_payload["provenance_chain"],
        "content_hash": task_payload["content_hash"],
        "derivation_hash": task_payload["derivation_hash"],
        "derived_from": str(source),
        "tensors_written": False,
        "snapshot_hash": snapshot["snapshot_hash"],
        "sdpa_attrs": capture.attrs,
        "signature": {
            name: meta(task_payload[name])
            for name in ("q", "k", "v", "o_weight", "out", "grad_output")
        },
        "gradients": {name: meta(tensor) for name, tensor in sorted(capture.grads.items())},
        "diagnostics": {
            "note": "diagnostic only -- one derivation pass, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
        },
    }
    return task_payload, metadata


def check_provenance(payload: dict[str, Any], *, snapshot_path: Path | None = None) -> list[str]:
    """Every link, against the tracked snapshot. Empty means clean."""
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("sdpa_config_id", harvest["config_id"]),
        ("o_proj_config_id", harvest["supporting"]["output_projection"]["config_id"]),
        ("frequency", harvest["frequency"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "derived_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = f"{snapshot['representative_layer']['module_path']}.self_attn"
    if identity.get("module_path") != expected_path:
        problems.append(
            f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}"
        )
    # The SDPA configuration is only the same configuration if its scalars match.
    for field, expected in sorted(harvest["attrs"].items()):
        got = payload["attrs"].get(field)
        if field == "scale":
            if got is None or abs(float(got) - float(expected)) > 1e-12:
                problems.append(f"attrs.scale: {got!r} != {expected!r}")
        elif got != expected:
            problems.append(f"attrs.{field}: {got!r} != {expected!r}")
    for index, name in enumerate(("q", "k", "v")):
        observed = harvest["input_shapes"][index]
        tensor = payload[name]
        if list(tensor.shape) != observed["shape"]:
            problems.append(f"{name} shape {list(tensor.shape)} != harvested {observed['shape']}")
        if list(tensor.stride()) != observed["stride"]:
            problems.append(
                f"{name} stride {list(tensor.stride())} != harvested {observed['stride']}"
            )
    expected_weight = harvest["supporting"]["output_projection"]["params"]["weight"]["shape"]
    if list(payload["o_weight"].shape) != expected_weight:
        problems.append(
            f"o_weight shape {list(payload['o_weight'].shape)} != harvested {expected_weight}"
        )
    return problems


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?"""
    from evograd.ops import get_op

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    declared = get_op(TASK_NAME).benchmark[0]
    q, k, _ = (entry["shape"] for entry in harvest["input_shapes"])
    hidden, fan_in = harvest["supporting"]["output_projection"]["params"]["weight"]["shape"]
    problems: list[str] = []
    for dim, expected in (
        ("B", q[0]),
        ("HQ", q[1]),
        ("T", q[2]),
        ("D", q[3]),
        ("HK", k[1]),
        ("QO", fan_in),
        ("H", hidden),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}")
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}")
    return problems


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
) -> dict[str, Any]:
    """Check both spellings against what Transformers computed.

    Two comparisons with two separately justified tolerances, as for the MLP.
    The production spelling is *the same call* the model made, so it is held to
    the Level-3 replay tolerances -- one BF16 unit roundoff forward, one epsilon
    on gradients. The declared dense reference computes the softmax in float32
    over a materialized score matrix, which is deliberately a different
    computation, and is held to the operator's declared tolerance.
    """
    from evograd.ops import get_op
    from evograd.ops.level2.qwen3_attention.forward_ref import (
        qwen3_attention_forward_production,
        qwen3_attention_forward_ref,
    )

    from ...levels.level3.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
        declared_gate,
        validate_noise_repeats,
    )

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AttentionExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    shape_problems = declaration_problems(snapshot_path)

    op = get_op(TASK_NAME)
    case = op.benchmark[0]
    base = op.tolerances[case.dtype][0]
    declared_tol = {
        "out": op.tolerance_for(case),
        **{name: op.tolerance_for(case, name) for name in ("dq", "dk", "dv", "do_weight")},
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
    production = _pair_pass(qwen3_attention_forward_production, **tensors)
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
        dense = _pair_pass(qwen3_attention_forward_ref, **tensors)
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
            _pair_pass(qwen3_attention_forward_production, **tensors) for _ in range(noise_repeats)
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
        "schema_version": REPORT_SCHEMA,
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


def _calibration_case(label, dtype, tensors, multipliers, repeats: int = 3):
    from evograd.ops.level2.qwen3_attention.forward_ref import (
        qwen3_attention_forward_production,
        qwen3_attention_forward_ref,
    )

    reference = _pair_pass(qwen3_attention_forward_ref, **tensors)
    production = _pair_pass(qwen3_attention_forward_production, **tensors)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(qwen3_attention_forward_production, **tensors)
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
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(TASK_NAME)
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
            )
        )

    canonical = None
    if not skip_canonical:
        payload, _ = derive_attention_invocation(
            source, device=device, snapshot_path=snapshot_path
        )
        canonical = _calibration_case(
            "canonical layer-14 invocation",
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
        )
        cases.append(canonical)

    def worst(subset):
        return {
            name: max((c["results"][name]["required_t"] for c in subset), default=0.0)
            for name in RESULT_NAMES
        }

    return {
        "schema_version": "evograd-qwen3-attention-tolerance/1",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.attention",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(sp):
        sp.add_argument("--source", type=Path, default=Path("results/qwen3-level4/layer14.pt"))
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
                args.source, device=args.device, skip_canonical=args.skip_canonical
            )
            print(summarize_calibration(report))
            _write(args.report, report)
            return 0

        from ...levels.level3.replay import validate_noise_repeats

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
        )
    except (AttentionExtractionError, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
