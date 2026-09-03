"""Extract and verify ``qwen3_swiglu_mlp`` from the verified Layer-14 replay.

    # describe the derived invocation (metadata only -- no tensors are written)
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp derive \
        --source results/qwen3-level4/layer14.pt \
        --metadata-out results/qwen3-level4/layer14-mlp.json

    # check the Level-2 declaration's reference against what the model computed
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp verify \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-mlp-verify.json

    # measure what tolerance a correct BF16 implementation actually needs
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp calibrate \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/qwen3_swiglu_mlp-tolerance.json

The source is deliberately the *replay*, not the full model. The Layer-14
artifact has already been shown to reproduce the full model bitwise, so a
capture taken from replaying it inherits that guarantee while costing one layer
instead of 596M parameters -- and, more usefully, the extraction is then
reproducible on any machine that has the artifact, with no 17 GiB training step
in between.

**Nothing here writes a second tensor file.** The MLP's input, output, upstream
gradient and all three weights and their gradients already live inside
``layer14.pt``; a derived ``.pt`` would be 68 MiB of the same numbers under a
different name, and the moment one of the two is regenerated they disagree
silently. The invocation is re-derived by replaying the layer artifact whenever
it is needed, which takes about a second, and only JSON metadata and reports are
written.

The provenance chain is carried explicitly and checked, link by link:

    canonical workload
      -> harvest manifest
        -> model.layers.14
          -> Layer-14 artifact (content and identity hashes)
            -> Qwen3MLP invocation
              -> qwen3_swiglu_mlp

Verification asks two questions, with two different tolerances, because they are
two different questions.

**Is the extraction wired correctly?** Compared against ``qwen3_swiglu_mlp_forward_hf``
-- the BF16 spelling ``Qwen3MLP`` actually executes -- this is the same
computation, so it is held to the Level-3 replay tolerances: one BF16 unit
roundoff forward, one epsilon on gradients. If the weight mapping, a transpose,
or the upstream gradient were wrong, this is what would catch it.

**Does the declared contract agree?** The declaration accumulates the gate/up
product in float32 and ``Qwen3MLP`` does not, so these are *deliberately not the
same computation* and a rounding-level tolerance would be the wrong instrument.
It is held to the operator's own declared BF16 tolerance -- the one the benchmark
harness gates candidates with -- because that is exactly the question being
asked: is the reference a valid answer for this operator.

Holding the second comparison to the first's tolerance would be a category
error, and loosening the first to accommodate the second would silently weaken a
check that is currently exact.
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

from ...levels.level3.artifact import (
    ArtifactError,
    content_hash_over,
    identity_hash_over,
    load_canonical,
    tensor_meta,
    to_cpu,
    to_device,
)
from ...levels.level3.replay import required_tolerance
from ...harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-qwen3-mlp-task/1"
REPORT_SCHEMA = "evograd-qwen3-mlp-verify/1"

TASK_NAME = "qwen3_swiglu_mlp"

CONTENT_KEYS = ("input", "output", "grad_output", "grad_input", "weights", "weight_grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "harvest_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

#: HF parameter name -> the declaration's argument name. The declaration takes
#: the three matrices separately because Qwen3 stores them separately.
WEIGHT_MAP = {
    "gate_proj.weight": "gate_weight",
    "up_proj.weight": "up_weight",
    "down_proj.weight": "down_weight",
}


class MlpExtractionError(RuntimeError):
    """The MLP invocation could not be extracted, or does not check out."""


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


class _MlpCapture:
    def __init__(self) -> None:
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.grad_input: torch.Tensor | None = None
        self.grad_output: torch.Tensor | None = None
        self.calls = 0
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [
            name
            for name in ("input", "output", "grad_input", "grad_output")
            if getattr(self, name) is None
        ]
        if missing:
            raise MlpExtractionError(f"MLP capture is incomplete, missing {missing}")


@contextmanager
def capture_mlp(layer: torch.nn.Module) -> Iterator[_MlpCapture]:
    """Watch the decoder layer's MLP submodule for the duration of the block."""
    mlp = layer.get_submodule("mlp")
    capture = _MlpCapture()

    def pre_hook(module, args, kwargs):
        capture.calls += 1
        if capture.calls > 1:
            raise MlpExtractionError(
                "the MLP ran more than once inside one extraction; the task "
                "artifact is defined as a single invocation"
            )
        tensor = args[0]
        capture.input = to_cpu(tensor, where="$input")
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_input", grad.detach().clone().to("cpu"))
                )
            )
        return None

    def post_hook(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.output = to_cpu(tensor, where="$output")
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_output", grad.detach().clone().to("cpu"))
                )
            )
        return None

    handles = [
        mlp.register_forward_pre_hook(pre_hook, with_kwargs=True),
        mlp.register_forward_hook(post_hook, with_kwargs=True),
    ]
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()
        for handle in capture.handles:
            handle.remove()
        capture.handles.clear()


def derive_mlp_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay Layer 14 and capture the MLP invocation inside it.

    Returns the tensors in memory and a JSON-safe description of them. Nothing
    is written: ``layer14.pt`` is the authoritative tensor store, and this is a
    view of part of it.
    """
    from ...levels.level3.replay import prepare_layer

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    # Every identity check, with no way to opt out.
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise MlpExtractionError(
            "the canonical extraction runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug extraction"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)

    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_mlp(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    mlp = layer.get_submodule("mlp")
    weights: dict[str, torch.Tensor] = {}
    weight_grads: dict[str, torch.Tensor] = {}
    for hf_name, declared in WEIGHT_MAP.items():
        param = mlp.get_parameter(hf_name)
        weights[declared] = param.detach().clone().to("cpu")
        if param.grad is None:
            raise MlpExtractionError(f"mlp.{hf_name} has no gradient after backward")
        weight_grads[declared] = param.grad.detach().clone().to("cpu")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.mlp"
    if module_path not in harvest["module_paths"]:
        raise MlpExtractionError(
            f"{module_path} is not among the harvested module paths for {TASK_NAME}; "
            "the snapshot and the artifact describe different runs"
        )

    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "harvest_config_id": harvest["config_id"],
            "frequency": harvest["frequency"],
            "layer_index": identity["layer_index"],
            "module_path": module_path,
            "source_content_hash": payload["content_hash"],
            "source_artifact_hash": payload["artifact_hash"],
            "provenance_kind": "captured_from_verified_replay",
        },
        "provenance_chain": [
            f"canonical workload {identity['workload_id']}",
            f"harvest manifest {identity['manifest_hash']}",
            f"{identity['module_path']} (event ordinal {identity['event_ordinal']})",
            f"Layer-14 artifact {payload['artifact_hash']}",
            f"Qwen3MLP invocation at {module_path}",
            TASK_NAME,
        ],
        "arch": {
            "hidden_size": harvest["attrs"]["hidden_size"],
            "intermediate_size": harvest["attrs"]["intermediate_size"],
            "hidden_act": harvest["attrs"]["hidden_act"],
            "dtype": dtype,
        },
        "input": capture.input,
        "output": capture.output,
        "grad_output": capture.grad_output,
        "grad_input": capture.grad_input,
        "weights": weights,
        "weight_grads": weight_grads,
    }
    # A fingerprint of the derived numbers, so two derivations on two machines
    # can be compared even though neither writes a file.
    task_payload["content_hash"] = content_hash_over(task_payload, CONTENT_KEYS)
    task_payload["derivation_hash"] = identity_hash_over(task_payload, IDENTITY_KEYS)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": task_payload["identity"],
        "provenance_chain": task_payload["provenance_chain"],
        "content_hash": task_payload["content_hash"],
        "derivation_hash": task_payload["derivation_hash"],
        "derived_from": str(source),
        "tensors_written": False,
        "arch": task_payload["arch"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "signature": {
            "input": tensor_meta(capture.input),
            "output": tensor_meta(capture.output),
            "grad_output": tensor_meta(capture.grad_output),
            "grad_input": tensor_meta(capture.grad_input),
            "weights": {k: tensor_meta(v) for k, v in sorted(weights.items())},
            "weight_grads": {k: tensor_meta(v) for k, v in sorted(weight_grads.items())},
        },
        "diagnostics": {
            "note": "diagnostic only -- one extraction pass, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
        },
    }
    return task_payload, metadata


def load_or_derive(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The only way to obtain the invocation: derive it from the layer artifact.

    There is no ``.pt`` to load. Keeping one would mean two files holding the
    same numbers, and nothing forcing them to stay equal.
    """
    return derive_mlp_invocation(source, device=device, snapshot_path=snapshot_path)


def check_provenance(payload: dict[str, Any], *, snapshot_path: Path | None = None) -> list[str]:
    """Every link in the chain, against the tracked snapshot. Empty means clean."""
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("harvest_config_id", harvest["config_id"]),
        ("frequency", harvest["frequency"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "captured_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = f"{snapshot['representative_layer']['module_path']}.mlp"
    if identity.get("module_path") != expected_path:
        problems.append(f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}")
    if identity.get("module_path") not in harvest["module_paths"]:
        problems.append(f"{identity.get('module_path')!r} is not a harvested module path")
    arch = payload["arch"]
    for field, expected in (
        ("hidden_size", harvest["attrs"]["hidden_size"]),
        ("intermediate_size", harvest["attrs"]["intermediate_size"]),
        ("hidden_act", harvest["attrs"]["hidden_act"]),
    ):
        if arch.get(field) != expected:
            problems.append(f"arch.{field}: {arch.get(field)!r} != {expected!r}")
    for name, entries in (("input", harvest["input_shapes"]), ("output", harvest["output_shapes"])):
        expected = entries[0]["shape"]
        got = list(payload[name].shape)
        if got != expected:
            problems.append(f"{name} shape {got} != harvested {expected}")
    return problems


# --------------------------------------------------------------------------
# verification of the Level-2 declaration against the capture
# --------------------------------------------------------------------------


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?

    Pure data: no tensors, no GPU. The declaration derives its benchmark dims
    from the snapshot at import time, so this can only fail if one of them was
    edited by hand -- which is exactly what it is here to catch.
    """
    from evograd.ops import get_op

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    declared = get_op(TASK_NAME).benchmark[0]
    problems: list[str] = []
    batch, seq, hidden = harvest["input_shapes"][0]["shape"]
    for dim, expected in (
        ("B", batch),
        ("T", seq),
        ("H", hidden),
        ("I", harvest["attrs"]["intermediate_size"]),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(
                f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}"
            )
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(
            f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}"
        )
    return problems


def _reference_pass(forward, payload: dict[str, Any], device: str):
    x = payload["input"].to(device).detach().clone().requires_grad_(True)
    weights = {
        name: payload["weights"][name].to(device).detach().clone().requires_grad_(True)
        for name in ("gate_weight", "up_weight", "down_weight")
    }
    out = forward(x, weights["gate_weight"], weights["up_weight"], weights["down_weight"])
    out.backward(payload["grad_output"].to(device))
    return (
        out.detach().clone(),
        x.grad.detach().clone(),
        {name: tensor.grad.detach().clone() for name, tensor in weights.items()},
    )


def _gate_reason(record: dict[str, Any]) -> str:
    if "required_t" in record:
        return (
            f"required_t={record['required_t']:.3e} > declared base "
            f"{record['declared_base']:.3e} (atol={record['atol']}, rtol={record['rtol']})"
        )
    return (
        f"max_rel_err_vs_scale={record.get('max_rel_err_vs_scale')} > "
        f"{record.get('tolerance')}"
    )


def run_verify(
    payload: dict[str, Any],
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    from evograd.ops import get_op
    from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
        qwen3_swiglu_mlp_forward_hf,
        qwen3_swiglu_mlp_forward_ref,
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
        raise MlpExtractionError(
            "the canonical verification runs on CUDA and no CUDA device is visible"
        )
    provenance_problems = check_provenance(payload, snapshot_path=snapshot_path)

    op = get_op(TASK_NAME)
    declared = op.benchmark[0].dims
    # Two independent questions, deliberately not merged: does this artifact
    # match the snapshot it claims to come from (``check_provenance``), and does
    # the *declaration* still match the canonical snapshot. The second is about
    # the tracked snapshot whatever artifact is being verified, so a shrunken
    # debug capture does not make the declaration look wrong.
    shape_problems = declaration_problems()
    weight_shapes = {
        "gate_weight": list(payload["weights"]["gate_weight"].shape),
        "up_weight": list(payload["weights"]["up_weight"].shape),
        "down_weight": list(payload["weights"]["down_weight"].shape),
    }
    hidden = payload["arch"]["hidden_size"]
    intermediate = payload["arch"]["intermediate_size"]
    for name, expected in (
        ("gate_weight", [intermediate, hidden]),
        ("up_weight", [intermediate, hidden]),
        ("down_weight", [hidden, intermediate]),
    ):
        if weight_shapes[name] != expected:
            shape_problems.append(
                f"captured {name} {weight_shapes[name]} != {expected} implied by the "
                "captured hidden/intermediate widths"
            )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    out, grad_x, weight_grads = _reference_pass(qwen3_swiglu_mlp_forward_ref, payload, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    # The operator's own declared tolerance, per result, applied exactly as the
    # benchmark harness applies it -- `allclose(atol, rtol)`, not a
    # scale-normalized proxy. Taken from the declaration rather than chosen
    # here, so it cannot be tuned to fit this result.
    case = op.benchmark[0]
    declared_tol = {
        "output": op.tolerance_for(case),
        "grad_input": op.tolerance_for(case, "dx"),
        **{
            name: op.tolerance_for(case, f"d{name}")
            for name in ("gate_weight", "up_weight", "down_weight")
        },
    }
    base = op.tolerances[case.dtype][0]

    comparisons = {
        "output": declared_gate(out, payload["output"].to(device), declared_tol["output"], base),
        "grad_input": declared_gate(
            grad_x, payload["grad_input"].to(device), declared_tol["grad_input"], base
        ),
        "weight_grads": {
            name: declared_gate(
                weight_grads[name],
                payload["weight_grads"][name].to(device),
                declared_tol[name],
                base,
            )
            for name in sorted(weight_grads)
        },
    }

    # The BF16 spelling Transformers actually ran. Same computation, so it gets
    # the tight replay tolerances -- this is the wiring check, and it is expected
    # to be bitwise identical.
    hf_out, hf_grad_x, hf_weight_grads = _reference_pass(
        qwen3_swiglu_mlp_forward_hf, payload, device
    )
    hf_comparisons = {
        "output": compare_tensors(hf_out, payload["output"].to(device), FORWARD_TOL),
        "grad_input": compare_tensors(hf_grad_x, payload["grad_input"].to(device), GRADIENT_TOL),
        "weight_grads": {
            name: compare_tensors(
                hf_weight_grads[name], payload["weight_grads"][name].to(device), GRADIENT_TOL
            )
            for name in sorted(hf_weight_grads)
        },
    }

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "reference compared against itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [
            _reference_pass(qwen3_swiglu_mlp_forward_ref, payload, device)
            for _ in range(noise_repeats)
        ]
        noise["output"] = _max_noise(
            _noise(passes[i][0], passes[0][0]) for i in range(1, noise_repeats)
        )
        noise["grad_input"] = _max_noise(
            _noise(passes[i][1], passes[0][1]) for i in range(1, noise_repeats)
        )
        noise["weight_grads"] = {
            name: _max_noise(
                _noise(passes[i][2][name], passes[0][2][name]) for i in range(1, noise_repeats)
            )
            for name in sorted(weight_grads)
        }

    failures = [f"provenance: {p}" for p in provenance_problems]
    failures += [f"shape: {p}" for p in shape_problems]
    for label, group in (("declared reference", comparisons), ("BF16 spelling", hf_comparisons)):
        for name in ("output", "grad_input"):
            record = group[name]
            if not record["within_tolerance"]:
                failures.append(f"{label} {name}: {_gate_reason(record)}")
        for name, record in group["weight_grads"].items():
            if not record["within_tolerance"]:
                failures.append(f"{label} {name} gradient: {_gate_reason(record)}")

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": TASK_NAME,
        "identity": payload["identity"],
        "provenance_chain": payload["provenance_chain"],
        "provenance_validated": not provenance_problems,
        "snapshot_hash": load_snapshot(snapshot_path)["snapshot_hash"],
        "declared_dims": declared,
        "tolerances": {
            "metric": "max|a-b| / max|b| (reference scale)",
            "declared_reference": {
                "values": declared_tol,
                "source": (
                    f"the {TASK_NAME} declaration's bfloat16 tolerance and its "
                    "per-gradient multipliers"
                ),
                "why": (
                    "the declared contract accumulates the gate/up product in "
                    "float32 and Qwen3MLP does not, so these are deliberately "
                    "not the same computation; the question is whether the "
                    "reference is a valid answer for the operator, which is what "
                    "this tolerance was declared to answer"
                ),
            },
            "hf_spelling": {
                "forward": FORWARD_TOL,
                "gradient": GRADIENT_TOL,
                "why": (
                    "the same computation Transformers ran, so it is held to the "
                    "Level-3 replay tolerances: one BF16 unit roundoff forward, "
                    "one epsilon on gradients. This is the wiring check."
                ),
            },
        },
        "comparisons": comparisons,
        "hf_spelling_comparisons": hf_comparisons,
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
    comparisons = report["comparisons"]
    lines = [
        f"[{report['status'].upper()}] {report['task']} against the captured Qwen3MLP",
        f"  provenance validated: {report['provenance_validated']}  "
        f"snapshot {report['snapshot_hash'][:16]}...",
    ]
    for link in report["provenance_chain"]:
        lines.append(f"    -> {link}")
    lines += [
        "",
        f"  declared dims {report['declared_dims']}",
        "  declared reference (float32-accumulated), against the operator's "
        "declared BF16 tolerance:",
        f"  output      rel {_fmt(comparisons['output']['max_rel_err_vs_scale'])}  "
        f"required_t {_fmt(comparisons['output'].get('required_t'))} <= base "
        f"{comparisons['output'].get('declared_base')}",
        f"  grad x      rel {_fmt(comparisons['grad_input']['max_rel_err_vs_scale'])}  "
        f"required_t {_fmt(comparisons['grad_input'].get('required_t'))}",
    ]
    for name, record in comparisons["weight_grads"].items():
        lines.append(
            f"  grad {name:<12} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"required_t {_fmt(record.get('required_t'))}"
        )
    hf = report["hf_spelling_comparisons"]
    lines += [
        "",
        "  the BF16 spelling Transformers ran -- same computation, tight tolerance:",
        f"    output rel {_fmt(hf['output']['max_rel_err_vs_scale'])}  "
        f"bitwise {hf['output']['bitwise_identical']}",
        f"    grad x rel {_fmt(hf['grad_input']['max_rel_err_vs_scale'])}  "
        f"bitwise {hf['grad_input']['bitwise_identical']}",
    ]
    for name, record in hf["weight_grads"].items():
        lines.append(
            f"    grad {name:<12} rel {_fmt(record['max_rel_err_vs_scale'])}  "
            f"bitwise {record['bitwise_identical']}"
        )
    noise = report["noise_floor"]
    if noise.get("measured"):
        worst = max(
            (v for v in noise["weight_grads"].values() if v is not None), default=0.0
        )
        lines += [
            "",
            f"  measured noise floor ({noise['repeats']} reference runs): "
            f"output {_fmt(noise['output'])}  grad x {_fmt(noise['grad_input'])}  "
            f"weight grads {_fmt(worst)}",
        ]
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# tolerance calibration
# --------------------------------------------------------------------------

#: Results the gate covers: the output and every gradient, named as the
#: declaration names them.
RESULT_NAMES = ("out", "dx", "dgate_weight", "dup_weight", "ddown_weight")


def _pair_pass(forward, x, gate_weight, up_weight, down_weight, dout):
    """Forward and backward through one spelling, returning every result."""
    leaves = {
        "x": x.detach().clone().requires_grad_(True),
        "gate_weight": gate_weight.detach().clone().requires_grad_(True),
        "up_weight": up_weight.detach().clone().requires_grad_(True),
        "down_weight": down_weight.detach().clone().requires_grad_(True),
    }
    out = forward(**leaves)
    out.backward(dout)
    return {
        "out": out.detach().clone(),
        "dx": leaves["x"].grad.detach().clone(),
        "dgate_weight": leaves["gate_weight"].grad.detach().clone(),
        "dup_weight": leaves["up_weight"].grad.detach().clone(),
        "ddown_weight": leaves["down_weight"].grad.detach().clone(),
    }


def _calibration_case(
    label: str,
    dtype: str,
    tensors: dict[str, Any],
    multipliers: dict[str, tuple[float, float]],
    repeats: int = 3,
):
    from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
        qwen3_swiglu_mlp_forward_hf,
        qwen3_swiglu_mlp_forward_ref,
    )

    reference = _pair_pass(qwen3_swiglu_mlp_forward_ref, **tensors)
    production = _pair_pass(qwen3_swiglu_mlp_forward_hf, **tensors)
    results = {
        name: required_tolerance(
            production[name], reference[name], multipliers.get(name, (1.0, 1.0))
        )
        for name in RESULT_NAMES
    }
    # Run-to-run noise of the production spelling, so the recommendation is not
    # calibrated against a number that moves on its own.
    noise = {name: 0.0 for name in RESULT_NAMES}
    for _ in range(max(repeats - 1, 0)):
        again = _pair_pass(qwen3_swiglu_mlp_forward_hf, **tensors)
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
    """Measure what a correct BF16 implementation actually needs.

    Two populations, because they answer different halves of the question: the
    declared correctness workloads are what every test run gates on, and the
    canonical [2, 2048, 1024] invocation is the one the benchmark times and the
    only one at production width -- a tolerance calibrated on 32-wide cases
    would say nothing about a 3072-long contraction.
    """
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(TASK_NAME)
    multipliers = {
        name: tuple(op.tolerance_multipliers.get(name, (1.0, 1.0))) for name in RESULT_NAMES
    }
    cases: list[dict[str, Any]] = []

    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        cases.append(
            _calibration_case(
                f"correctness {workload.dims}",
                workload.dtype,
                {
                    "x": values["x"],
                    "gate_weight": values["gate_weight"],
                    "up_weight": values["up_weight"],
                    "down_weight": values["down_weight"],
                    "dout": values["dout"],
                },
                multipliers,
            )
        )

    canonical = None
    if not skip_canonical:
        payload, _ = derive_mlp_invocation(source, device=device, snapshot_path=snapshot_path)
        canonical = _calibration_case(
            "canonical layer-14 invocation",
            payload["arch"]["dtype"],
            {
                "x": payload["input"].to(device),
                "gate_weight": payload["weights"]["gate_weight"].to(device),
                "up_weight": payload["weights"]["up_weight"].to(device),
                "down_weight": payload["weights"]["down_weight"].to(device),
                "dout": payload["grad_output"].to(device),
            },
            multipliers,
        )
        cases.append(canonical)

    by_result = {
        name: max(case["results"][name]["required_t"] for case in cases)
        for name in RESULT_NAMES
    }
    bf16_cases = [case for case in cases if case["dtype"] == "bfloat16"]
    bf16_by_result = {
        name: max((case["results"][name]["required_t"] for case in bf16_cases), default=0.0)
        for name in RESULT_NAMES
    }
    fp32_cases = [case for case in cases if case["dtype"] == "float32"]
    fp32_by_result = {
        name: max((case["results"][name]["required_t"] for case in fp32_cases), default=0.0)
        for name in RESULT_NAMES
    }
    declared = {
        "float32": op.tolerances.get("float32"),
        "bfloat16": op.tolerances.get("bfloat16"),
        "multipliers": op.tolerance_multipliers,
    }
    return {
        "schema_version": "evograd-qwen3-mlp-tolerance/1",
        "task": TASK_NAME,
        "device": device,
        "metric": (
            "smallest base t with allclose(atol=ma*t, rtol=mr*t); "
            "t >= max(|a-b| / (ma + mr*|b|)) using the declaration's multipliers"
        ),
        "multipliers": {name: list(value) for name, value in multipliers.items()},
        "compared": "declared float32-accumulated forward vs runtime_forward (the HF BF16 spelling)",
        "cases": cases,
        "worst_required_t": {
            "overall": by_result,
            "bfloat16": bf16_by_result,
            "float32": fp32_by_result,
        },
        "canonical_worst_required_t": canonical["worst_required_t"] if canonical else None,
        "declared_tolerances": declared,
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
                f"    {name:<14} required_t {record['required_t']:.3e}   "
                f"max_abs {record['max_abs_err']:.3e}   "
                f"noise {case['production_noise_required_t'][name]:.3e}"
            )
    lines.append("")
    for group, values in report["worst_required_t"].items():
        worst = max(values.values()) if values else 0.0
        lines.append(f"  worst required_t ({group}): {worst:.3e}   per result: " + ", ".join(
            f"{name}={value:.2e}" for name, value in values.items()
        ))
    lines.append(f"  declared: {report['declared_tolerances']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(sp):
        sp.add_argument(
            "--source", type=Path, default=Path("results/qwen3-level4/layer14.pt")
        )
        sp.add_argument("--device", default="cuda")
        return sp

    derive = source_args(sub.add_parser("derive", help="describe the derived invocation"))
    derive.add_argument("--metadata-out", type=Path, default=None)

    verify = source_args(
        sub.add_parser("verify", help="check the Level-2 reference against the capture")
    )
    verify.add_argument("--report", type=Path, default=None)
    verify.add_argument("--noise-repeats", type=int, default=4)

    calibrate = source_args(
        sub.add_parser("calibrate", help="measure the tolerance a correct BF16 spelling needs")
    )
    calibrate.add_argument("--report", type=Path, default=None)
    calibrate.add_argument(
        "--skip-canonical",
        action="store_true",
        help="calibrate on the declared correctness workloads only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "derive":
        try:
            payload, metadata = derive_mlp_invocation(args.source, device=args.device)
        except (MlpExtractionError, ArtifactError) as exc:
            print(f"derivation failed: {exc}", file=sys.stderr)
            return 1
        print(f"derived {TASK_NAME} from {args.source} (no tensors written)")
        for link in payload["provenance_chain"]:
            print(f"  -> {link}")
        print(f"  content    {payload['content_hash']}")
        print(f"  derivation {payload['derivation_hash']}")
        if args.metadata_out is not None:
            Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.metadata_out).write_text(
                json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.metadata_out}")
        return 0

    from ...levels.level3.replay import validate_noise_repeats

    if args.command == "calibrate":
        try:
            report = run_calibration(
                args.source, device=args.device, skip_canonical=args.skip_canonical
            )
        except (MlpExtractionError, ArtifactError) as exc:
            print(f"calibration failed: {exc}", file=sys.stderr)
            return 1
        print(summarize_calibration(report))
        if args.report is not None:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"\nwrote {args.report}")
        return 0

    try:
        validate_noise_repeats(args.noise_repeats)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        payload, _ = derive_mlp_invocation(args.source, device=args.device)
        report = run_verify(payload, device=args.device, noise_repeats=args.noise_repeats)
    except (MlpExtractionError, ArtifactError) as exc:
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(summarize_verify(report))
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
