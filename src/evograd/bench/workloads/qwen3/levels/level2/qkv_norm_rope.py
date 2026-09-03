"""Derive and verify ``qwen3_qkv_norm_rope`` from the canonical Layer-14 artifact.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope derive \
        --source results/qwen3-level4/layer14.pt \
        --metadata-out results/qwen3-level4/layer14-qkv.json

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope verify \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-qkv-verify.json

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope calibrate \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/qwen3_qkv_norm_rope-tolerance.json

``layer14.pt`` stays the only tensor artifact. This boundary's inputs and
outputs already live inside it, so they are re-derived by replaying the layer
and hooking two points, and only JSON is written.

The two hooks bracket the boundary exactly:

* ``self_attn``'s forward pre-hook sees ``hidden_states`` -- the normalized
  residual stream that is this task's ``x`` -- and a tensor hook on it yields
  ``dx``. It is consumed by nothing else in the layer, so its gradient *is* this
  boundary's input gradient.
* the ``scaled_dot_product_attention`` call sees ``q``, ``k`` and ``v`` in the
  exact layout the boundary must produce, and tensor hooks on them yield
  ``dq``, ``dk``, ``dv`` -- which are this boundary's *upstream* gradients, the
  ones the next task hands back.

``cos`` and ``sin`` come from the artifact's ``position_embeddings`` argument.
The five weights and their gradients come from the layer's own modules; each is
used once per step, so a gradient read after one backward belongs to this
boundary alone.
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
from .attention import preserve_layout_cpu
from ...levels.level3.replay import declared_gate, required_tolerance
from ...harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-qwen3-qkv-task/1"
REPORT_SCHEMA = "evograd-qwen3-qkv-verify/1"

TASK_NAME = "qwen3_qkv_norm_rope"

CONTENT_KEYS = ("inputs", "outputs", "output_grads", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "rope_config_id",
    "q_proj_config_id",
    "kv_proj_config_id",
    "q_norm_config_id",
    "k_norm_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

OUTPUT_NAMES = ("q", "k", "v")
GRAD_NAMES = (
    "dx",
    "dq_weight",
    "dk_weight",
    "dv_weight",
    "dq_norm_weight",
    "dk_norm_weight",
)

#: HF parameter path -> the declaration's argument name.
WEIGHT_MAP = {
    "q_proj.weight": "q_weight",
    "k_proj.weight": "k_weight",
    "v_proj.weight": "v_weight",
    "q_norm.weight": "q_norm_weight",
    "k_norm.weight": "k_norm_weight",
}


class QkvExtractionError(RuntimeError):
    """The QKV invocation could not be derived, or does not check out."""


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


class _QkvCapture:
    def __init__(self) -> None:
        self.x = None
        self.outputs: dict[str, torch.Tensor] = {}
        self.output_grads: dict[str, torch.Tensor] = {}
        self.dx = None
        self.attn_calls = 0
        self.sdpa_calls = 0
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [n for n in ("x", "dx") if getattr(self, n) is None]
        missing += [n for n in OUTPUT_NAMES if n not in self.outputs]
        missing += [f"d{n}" for n in OUTPUT_NAMES if f"d{n}" not in self.output_grads]
        if missing:
            raise QkvExtractionError(f"QKV capture is incomplete, missing {missing}")


@contextmanager
def capture_qkv(layer: torch.nn.Module) -> Iterator[_QkvCapture]:
    """Bracket the boundary: the attention input, and the SDPA inputs."""
    capture = _QkvCapture()
    attention = layer.get_submodule("self_attn")
    original = torch.nn.functional.scaled_dot_product_attention

    def pre_hook(module, args, kwargs):
        capture.attn_calls += 1
        if capture.attn_calls > 1:
            raise QkvExtractionError(
                "the attention module ran more than once inside one derivation; "
                "the task is defined as a single invocation"
            )
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(hidden):  # pragma: no cover - defensive
            raise QkvExtractionError("could not find the attention module's input")
        capture.x = preserve_layout_cpu(hidden)
        if hidden.requires_grad:
            capture.handles.append(
                hidden.register_hook(
                    lambda grad: setattr(capture, "dx", preserve_layout_cpu(grad))
                )
            )
        return None

    def sdpa_wrapper(query, key, value, *args, **kwargs):
        capture.sdpa_calls += 1
        if capture.sdpa_calls > 1:
            raise QkvExtractionError("SDPA ran more than once inside one derivation")
        for name, tensor in zip(OUTPUT_NAMES, (query, key, value)):
            capture.outputs[name] = preserve_layout_cpu(tensor)
            if tensor.requires_grad:
                capture.handles.append(
                    tensor.register_hook(
                        lambda grad, name=name: capture.output_grads.__setitem__(
                            f"d{name}", preserve_layout_cpu(grad)
                        )
                    )
                )
        return original(query, key, value, *args, **kwargs)

    handle = attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper
    try:
        yield capture
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        handle.remove()
        for h in capture.handles:
            h.remove()
        capture.handles.clear()


def derive_qkv_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ...levels.level3.replay import prepare_layer

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QkvExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    cos, sin = kwargs["position_embeddings"]
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_qkv(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    attention = layer.get_submodule("self_attn")
    weights: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {"dx": capture.dx}
    for hf_name, declared in WEIGHT_MAP.items():
        param = attention.get_parameter(hf_name)
        weights[declared] = preserve_layout_cpu(param)
        if param.grad is None:
            raise QkvExtractionError(f"self_attn.{hf_name} has no gradient after backward")
        grads[f"d{declared}"] = preserve_layout_cpu(param.grad)
    missing = [name for name in GRAD_NAMES if name not in grads]
    if missing:  # pragma: no cover - defensive
        raise QkvExtractionError(f"missing input gradients {missing}")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.self_attn"
    supporting = harvest["supporting"]
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "rope_config_id": harvest["config_id"],
            "q_proj_config_id": supporting["q_projection"]["config_id"],
            "kv_proj_config_id": supporting["kv_projection"]["config_id"],
            "q_norm_config_id": supporting["q_norm"]["config_id"],
            "k_norm_config_id": supporting["k_norm"]["config_id"],
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
            f"q/k/v projections, Q/K RMSNorm and RoPE at {module_path}",
            TASK_NAME,
        ],
        "attrs": {
            "eps": supporting["q_norm"]["attrs"]["eps"],
            "unsqueeze_dim": harvest["attrs"]["unsqueeze_dim"],
            "num_attention_heads": supporting["enclosing_attention"]["attrs"][
                "num_attention_heads"
            ],
            "num_key_value_heads": supporting["enclosing_attention"]["attrs"][
                "num_key_value_heads"
            ],
            "head_dim": supporting["enclosing_attention"]["attrs"]["head_dim"],
        },
        "inputs": {
            "x": capture.x,
            **weights,
            "cos": preserve_layout_cpu(cos),
            "sin": preserve_layout_cpu(sin),
        },
        "outputs": {name: capture.outputs[name] for name in OUTPUT_NAMES},
        "output_grads": {f"d{name}": capture.output_grads[f"d{name}"] for name in OUTPUT_NAMES},
        "grads": {name: grads[name] for name in GRAD_NAMES},
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
        "attrs": task_payload["attrs"],
        "signature": {
            "inputs": {k: meta(v) for k, v in task_payload["inputs"].items()},
            "outputs": {k: meta(v) for k, v in task_payload["outputs"].items()},
            "output_grads": {k: meta(v) for k, v in task_payload["output_grads"].items()},
            "grads": {k: meta(v) for k, v in task_payload["grads"].items()},
        },
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
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    supporting = harvest["supporting"]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("rope_config_id", harvest["config_id"]),
        ("q_proj_config_id", supporting["q_projection"]["config_id"]),
        ("kv_proj_config_id", supporting["kv_projection"]["config_id"]),
        ("q_norm_config_id", supporting["q_norm"]["config_id"]),
        ("k_norm_config_id", supporting["k_norm"]["config_id"]),
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
    if payload["attrs"]["eps"] != supporting["q_norm"]["attrs"]["eps"]:
        problems.append("attrs.eps disagrees with the harvested q_norm")
    # The observed output layout, output by output.
    for index, name in enumerate(("q", "k")):
        observed = harvest["output_shapes"][index]
        tensor = payload["outputs"][name]
        if list(tensor.shape) != observed["shape"]:
            problems.append(f"{name} shape {list(tensor.shape)} != harvested {observed['shape']}")
        if list(tensor.stride()) != observed["stride"]:
            problems.append(f"{name} stride {list(tensor.stride())} != harvested {observed['stride']}")
    v_observed = supporting["consumer"]["input_shapes"][2]
    v = payload["outputs"]["v"]
    if list(v.shape) != v_observed["shape"] or list(v.stride()) != v_observed["stride"]:
        problems.append(
            f"v layout {list(v.shape)}/{list(v.stride())} != harvested "
            f"{v_observed['shape']}/{v_observed['stride']}"
        )
    for declared, record in (
        ("q_weight", supporting["q_projection"]),
        ("k_weight", supporting["kv_projection"]),
        ("v_weight", supporting["kv_projection"]),
        ("q_norm_weight", supporting["q_norm"]),
        ("k_norm_weight", supporting["k_norm"]),
    ):
        expected_shape = record["params"]["weight"]["shape"]
        got = list(payload["inputs"][declared].shape)
        if got != expected_shape:
            problems.append(f"{declared} shape {got} != harvested {expected_shape}")
    return problems


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?"""
    from evograd.ops import get_op

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    supporting = harvest["supporting"]
    declared = get_op(TASK_NAME).benchmark[0]
    batch, heads, tokens, head_dim = harvest["output_shapes"][0]["shape"]
    kv_heads = harvest["output_shapes"][1]["shape"][1]
    hidden = supporting["q_projection"]["input_shapes"][0]["shape"][-1]
    problems: list[str] = []
    for dim, expected in (
        ("B", batch),
        ("T", tokens),
        ("H", hidden),
        ("HQ", heads),
        ("HK", kv_heads),
        ("D", head_dim),
        ("QO", supporting["q_projection"]["params"]["weight"]["shape"][0]),
        ("KVO", supporting["kv_projection"]["params"]["weight"]["shape"][0]),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}")
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}")
    return problems


# --------------------------------------------------------------------------
# verification and calibration
# --------------------------------------------------------------------------

ARG_ORDER = (
    "x",
    "q_weight",
    "k_weight",
    "v_weight",
    "q_norm_weight",
    "k_norm_weight",
    "cos",
    "sin",
)
ACTIVE_ARGS = ARG_ORDER[:6]


def _pair_pass(forward, inputs: dict[str, torch.Tensor], output_grads, eps: float):
    """Forward and backward through one spelling, keeping the observed layout."""
    leaves = {name: inputs[name].detach().clone().requires_grad_(True) for name in ACTIVE_ARGS}
    outputs = forward(
        *(leaves.get(name, inputs.get(name)) for name in ARG_ORDER), eps=eps
    )
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
) -> dict[str, Any]:
    """Check both spellings against what Transformers computed.

    The production spelling is the same computation the model ran, so it is held
    to the Level-3 replay tolerances. The declared reference keeps the RMSNorm
    scale and the rotation in float32 and is therefore deliberately a different
    computation, held to the operator's own declared gate.
    """
    from evograd.ops import get_op
    from evograd.ops.level2.qwen3_qkv_norm_rope.forward_ref import (
        qwen3_qkv_norm_rope_forward_production,
        qwen3_qkv_norm_rope_forward_ref,
    )

    from ...levels.level3.replay import (
        FORWARD_TOL,
        GRADIENT_TOL,
        _max_noise,
        _noise,
        compare_tensors,
        validate_noise_repeats,
    )

    noise_repeats = validate_noise_repeats(noise_repeats)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QkvExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    declaration_issues = declaration_problems(snapshot_path)

    op = get_op(TASK_NAME)
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
    eps = float(payload["attrs"]["eps"])
    names = OUTPUT_NAMES + GRAD_NAMES

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    production = _pair_pass(
        qwen3_qkv_norm_rope_forward_production, inputs, output_grads, eps
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

    reference = _pair_pass(qwen3_qkv_norm_rope_forward_ref, inputs, output_grads, eps)
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
            _pair_pass(qwen3_qkv_norm_rope_forward_production, inputs, output_grads, eps)
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
        "schema_version": REPORT_SCHEMA,
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


def _calibration_case(label, dtype, inputs, output_grads, eps, multipliers, repeats: int = 3):
    from evograd.ops.level2.qwen3_qkv_norm_rope.forward_ref import (
        qwen3_qkv_norm_rope_forward_production,
        qwen3_qkv_norm_rope_forward_ref,
    )

    names = OUTPUT_NAMES + GRAD_NAMES
    reference = _pair_pass(qwen3_qkv_norm_rope_forward_ref, inputs, output_grads, eps)
    production = _pair_pass(qwen3_qkv_norm_rope_forward_production, inputs, output_grads, eps)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in names
    }
    noise = {name: 0.0 for name in names}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(qwen3_qkv_norm_rope_forward_production, inputs, output_grads, eps)
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
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(TASK_NAME)
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
                float(values["eps"]),
                multipliers,
            )
        )

    if not skip_canonical:
        payload, _ = derive_qkv_invocation(source, device=device, snapshot_path=snapshot_path)
        cases.append(
            _calibration_case(
                "canonical layer-14 invocation",
                str(payload["outputs"]["q"].dtype).replace("torch.", ""),
                {k: v.to(device) for k, v in payload["inputs"].items()},
                [payload["output_grads"][f"d{n}"].to(device) for n in OUTPUT_NAMES],
                float(payload["attrs"]["eps"]),
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
        "schema_version": "evograd-qwen3-qkv-tolerance/1",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.qkv_norm_rope",
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
        payload, _ = derive_qkv_invocation(args.source, device=args.device)
        report = run_verify(
            payload, device=args.device, noise_repeats=args.noise_repeats
        )
    except (QkvExtractionError, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
