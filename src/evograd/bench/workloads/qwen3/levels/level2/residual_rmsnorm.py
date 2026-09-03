"""Derive and verify ``fused_add_rms_norm`` from the canonical Layer-14 artifact.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.residual_rmsnorm derive \
        --source results/qwen3-level4/layer14.pt \
        --metadata-out results/qwen3-level4/layer14-residual.json

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.residual_rmsnorm verify \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-residual-verify.json

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.residual_rmsnorm calibrate \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/fused_add_rms_norm-tolerance.json

``layer14.pt`` stays the only tensor artifact; the boundary is re-derived by
replaying the layer and hooking it, and only JSON is written.

**The representative boundary inside layer 14** is the attention residual add
and the norm that follows it::

    residual   = the decoder layer's input
    x          = the self_attn branch's output
    summed     = residual + x
    normalized = post_attention_layernorm(summed)

**What is and is not directly comparable.** Three of the four gradients are
observable in the layer graph as themselves:

* ``dx`` -- ``x`` is consumed only by the residual add, so its gradient in the
  layer *is* this boundary's ``dx``.
* ``dtotal`` -- ``summed``'s own gradient, which the contract says equals
  ``dsummed + RMSNormBackward(dnormalized)``. Comparing it checks the
  combination rule directly.
* ``dweight`` -- ``post_attention_layernorm.weight`` is used once per step.

``dresidual`` is **not** directly comparable, and this module does not pretend
otherwise. In the real layer the decoder input feeds two consumers -- the
residual add *and* ``input_layernorm`` -- so its ``.grad`` is the sum of this
boundary's ``dresidual`` and a gradient from a path outside the boundary. The
contract's claim about it (``dresidual == dx``) is proved instead against an
isolated autograd reference, and the report records that it was proved that way.

The two upstream gradients are both observed rather than invented:
``dnormalized`` is the gradient arriving at the norm's output, and ``dsummed``
is the layer's own ``grad_output`` -- because the layer ends with
``summed + mlp(...)``, so the gradient reaching ``summed`` from its second
consumer is exactly the gradient of the layer's output.
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

SCHEMA_VERSION = "evograd-qwen3-residual-task/1"
REPORT_SCHEMA = "evograd-qwen3-residual-verify/1"

TASK_NAME = "fused_add_rms_norm"

CONTENT_KEYS = ("inputs", "outputs", "output_grads", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "rms_norm_config_id",
    "fusion_sites",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

OUTPUT_NAMES = ("out", "summed")
#: Only these are observable as themselves in the layer graph; ``dr`` is not.
DIRECT_GRADS = ("dx", "dweight")


class ResidualExtractionError(RuntimeError):
    """The boundary could not be derived, or does not check out."""


class _ResidualCapture:
    def __init__(self) -> None:
        self.residual = None
        self.x = None
        self.summed = None
        self.normalized = None
        self.dnormalized = None
        self.dsummed_total = None
        self.dx = None
        self.norm_calls = 0

    def require_complete(self) -> None:
        missing = [
            name
            for name in (
                "residual",
                "x",
                "summed",
                "normalized",
                "dnormalized",
                "dsummed_total",
                "dx",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ResidualExtractionError(f"residual capture is incomplete, missing {missing}")


@contextmanager
def capture_residual(layer: torch.nn.Module) -> Iterator[_ResidualCapture]:
    """Hook the attention branch's output and the norm that follows the add."""
    capture = _ResidualCapture()
    attention = layer.get_submodule("self_attn")
    norm = layer.get_submodule("post_attention_layernorm")
    handles: list[Any] = []

    def attention_post(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.x = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "dx", preserve_layout_cpu(grad))
                )
            )
        return None

    def norm_pre(module, args, kwargs):
        capture.norm_calls += 1
        if capture.norm_calls > 1:
            raise ResidualExtractionError(
                "post_attention_layernorm ran more than once inside one "
                "derivation; the task is defined as a single invocation"
            )
        tensor = args[0] if args else kwargs["hidden_states"]
        capture.summed = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            # `summed`'s own gradient: the contract's `dtotal`.
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(
                        capture, "dsummed_total", preserve_layout_cpu(grad)
                    )
                )
            )
        return None

    def norm_post(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.normalized = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(
                        capture, "dnormalized", preserve_layout_cpu(grad)
                    )
                )
            )
        return None

    handles.append(attention.register_forward_hook(attention_post, with_kwargs=True))
    handles.append(norm.register_forward_pre_hook(norm_pre, with_kwargs=True))
    handles.append(norm.register_forward_hook(norm_post, with_kwargs=True))
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def derive_residual_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ...levels.level3.replay import prepare_layer

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ResidualExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_residual(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    # The decoder layer's input is this boundary's `residual`.
    capture.residual = preserve_layout_cpu(leaf)
    capture.require_complete()

    norm = layer.get_submodule("post_attention_layernorm")
    if norm.weight.grad is None:
        raise ResidualExtractionError("post_attention_layernorm.weight has no gradient")
    weight = preserve_layout_cpu(norm.weight)
    dweight = preserve_layout_cpu(norm.weight.grad)
    eps = float(norm.variance_epsilon)

    # `summed + mlp(...)` is the layer's output, so the gradient reaching
    # `summed` from its second consumer *is* the layer's upstream gradient.
    dsummed = preserve_layout_cpu(grad_output)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.post_attention_layernorm"
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "rms_norm_config_id": harvest["config_id"],
            "fusion_sites": harvest["fusion_sites"]["total"],
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
            f"attention residual add + {module_path}",
            TASK_NAME,
        ],
        "attrs": {
            "eps": eps,
            "rows": int(capture.summed.numel() // capture.summed.shape[-1]),
            "cols": int(capture.summed.shape[-1]),
            "fusion_sites_per_step": harvest["fusion_sites"]["total"],
            "directly_verified_invocations": harvest["fusion_sites"][
                "directly_verified_invocations"
            ],
        },
        "inputs": {"x": capture.x, "r": capture.residual, "weight": weight},
        "outputs": {"out": capture.normalized, "summed": capture.summed},
        "output_grads": {"dout": capture.dnormalized, "dsummed": dsummed},
        "grads": {
            "dx": capture.dx,
            "dweight": dweight,
            # `summed`'s own gradient, which the contract says is
            # `dsummed + RMSNormBackward(dnormalized)` -- i.e. `dtotal`.
            "dtotal": capture.dsummed_total,
        },
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
        "not_directly_comparable": {
            "dr": (
                "the decoder layer's input feeds both the residual add and "
                "input_layernorm, so its gradient in the layer is this "
                "boundary's dresidual plus a path outside the boundary; the "
                "contract's dresidual == dx is proved against an isolated "
                "autograd reference instead"
            )
        },
        "signature": {
            group: {k: meta(v) for k, v in task_payload[group].items()}
            for group in ("inputs", "outputs", "output_grads", "grads")
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
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("rms_norm_config_id", harvest["config_id"]),
        ("fusion_sites", harvest["fusion_sites"]["total"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "derived_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = (
        f"{snapshot['representative_layer']['module_path']}.post_attention_layernorm"
    )
    if identity.get("module_path") != expected_path:
        problems.append(
            f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}"
        )
    if payload["attrs"]["eps"] != harvest["attrs"]["eps"]:
        problems.append("attrs.eps disagrees with the harvested RMSNorm")
    if payload["attrs"]["cols"] != harvest["attrs"]["normalized_size"]:
        problems.append(
            f"cols {payload['attrs']['cols']} != harvested normalized_size "
            f"{harvest['attrs']['normalized_size']}"
        )
    observed = harvest["output_shapes"][0]["shape"]
    rows = payload["attrs"]["rows"]
    if rows != observed[0] * observed[1]:
        problems.append(f"rows {rows} != harvested tokens {observed[0]} x {observed[1]}")
    return problems


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Qwen suite in the declaration still match the snapshot?"""
    from evograd.ops import get_op
    from evograd.ops.level2.fused_add_rms_norm import QWEN3_FUSION_SITES

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    op = get_op(TASK_NAME)
    cases = op.benchmark_workloads("qwen3_0_6b_observed")
    problems: list[str] = []
    if len(cases) != 1:
        problems.append(f"expected one observed Qwen case, found {len(cases)}")
        return problems
    case = cases[0]
    observed = harvest["output_shapes"][0]["shape"]
    for dim, expected in (
        ("rows", observed[0] * observed[1]),
        ("cols", harvest["attrs"]["normalized_size"]),
    ):
        if case.dims.get(dim) != expected:
            problems.append(f"declared {dim}={case.dims.get(dim)} != harvested {expected}")
    if case.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {case.dtype!r} != harvested {harvest['dtype']!r}")
    if QWEN3_FUSION_SITES["total"] != harvest["fusion_sites"]["total"]:
        problems.append(
            f"declared fusion sites {QWEN3_FUSION_SITES['total']} != snapshot "
            f"{harvest['fusion_sites']['total']}"
        )
    return problems


# --------------------------------------------------------------------------
# verification and calibration
# --------------------------------------------------------------------------


def transformers_spelling(x, residual, weight, eps=1e-6):
    """The exact ``Qwen3RMSNorm`` spelling, after the residual add.

    ``Qwen3RMSNorm`` computes the variance in float32 but applies the learned
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
    from evograd.ops import get_op
    from evograd.ops.level2.fused_add_rms_norm.forward_ref import (
        fused_add_rms_norm_forward_ref,
        fused_add_rms_norm_runtime_ref,
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
        raise ResidualExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)
    declaration_issues = declaration_problems(snapshot_path)

    op = get_op(TASK_NAME)
    case = op.benchmark_workloads("qwen3_0_6b_observed")[0]
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

    runtime = _pair_pass(fused_add_rms_norm_runtime_ref, *args)
    reference = _pair_pass(fused_add_rms_norm_forward_ref, *args)

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
    from evograd.ops.level2.fused_add_rms_norm.forward_ref import (
        fused_add_rms_norm_forward_ref,
        fused_add_rms_norm_runtime_ref,
    )

    reference = _pair_pass(fused_add_rms_norm_forward_ref, *args)
    production = _pair_pass(fused_add_rms_norm_runtime_ref, *args)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(fused_add_rms_norm_runtime_ref, *args)
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
    from evograd.ops import get_op

    op = get_op(TASK_NAME)
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
        case = op.benchmark_workloads("qwen3_0_6b_observed")[0]
        multipliers = {
            name: tuple(
                t / b if b else 1.0
                for t, b in zip(op.tolerance_for(case, name), op.tolerance_for(case))
            )
            for name in RESULT_NAMES
        }
        cases.append(
            _calibration_case(
                "canonical layer-14 invocation",
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
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.residual_rmsnorm",
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

        from ...levels.level3.replay import validate_noise_repeats

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
