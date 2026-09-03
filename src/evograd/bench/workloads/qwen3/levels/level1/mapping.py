"""The Qwen3-0.6B Level-1 mapping: check it, calibrate it, verify it.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping mapping
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping calibrate \
        --op causal_gqa_attention \
        --report results/qwen3-level4/causal_gqa_attention-tolerance.json
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping verify \
        --source results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-sdpa-verify.json
    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping loss

Six generic primitives carry every configuration the canonical step runs.
Five already existed and were generalized; ``causal_gqa_attention`` is new,
because a decoder-only step runs fused causal GQA attention and this benchmark
had no generic task for it.

Two configurations are deliberately *not* mapped. There is no standalone
``softmax`` case: the model runs fused SDPA and never materializes one, so a
Qwen softmax workload would be a shape no step executes. And the observed
``silu`` record maps onto ``swiglu`` rather than a bare activation task, because
the production pointwise boundary is ``silu(gate) * up`` -- the activation never
appears without the multiply. The SiLU record survives as supporting provenance.

``layer14.pt`` stays the only tensor artifact. The canonical SDPA check re-derives
its tensors by replaying that artifact; the vocabulary-width check builds its
logits on demand and drops them.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from ...levels.level3.artifact import ArtifactError, load_canonical
from ...levels.level2.attention import preserve_layout_cpu
from ...levels.level3.replay import declared_gate, required_tolerance
from ...harvest.snapshot import load as load_snapshot

REPORT_SCHEMA = "evograd-qwen3-level1-verify/1"

#: Generic task -> the Level-2 contract its outputs feed. Checked structurally
#: by the focused tests; recorded here so the composition is stated once.
COMPOSES_INTO = {
    "linear_no_bias": ("qwen3_qkv_norm_rope", "qwen3_attention", "qwen3_swiglu_mlp"),
    "rmsnorm": ("qwen3_qkv_norm_rope", "fused_add_rms_norm"),
    "rope": ("qwen3_qkv_norm_rope",),
    "swiglu": ("qwen3_swiglu_mlp",),
    "causal_gqa_attention": ("qwen3_attention",),
    "cross_entropy": (),
}


class Level1Error(RuntimeError):
    """The Level-1 mapping could not be checked."""


# --------------------------------------------------------------------------
# the mapping table
# --------------------------------------------------------------------------


def mapping(snapshot_path: Path | None = None) -> dict[str, Any]:
    payload = load_snapshot(snapshot_path)
    return {
        "snapshot_hash": payload["snapshot_hash"],
        "workload_id": payload["workload_id"],
        "tasks": payload["level1"],
        "composes_into": COMPOSES_INTO,
    }


def summarize_mapping(report: dict[str, Any]) -> str:
    lines = [
        f"Qwen3-0.6B Level-1 mapping  (snapshot {report['snapshot_hash'][:16]}...)",
        f"  workload {report['workload_id']}",
        "",
    ]
    for task, entry in report["tasks"].items():
        feeds = ", ".join(report["composes_into"].get(task, ())) or "-"
        lines.append(
            f"  {task}  (from harvested {entry['harvested_task']}, "
            f"total frequency {entry['total_frequency']})   -> {feeds}"
        )
        for config in entry["configurations"]:
            roles = ",".join(config["roles"])
            dims = " ".join(f"{k}={v}" for k, v in config["dims"].items())
            lines.append(
                f"      {config['config_id']}  x{config['frequency']:<3} "
                f"{roles:<38} {dims}  [{config['dtype'].replace('torch.', '')}]"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# generic tolerance calibration for any task with a runtime_forward
# --------------------------------------------------------------------------


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
    from evograd.ops import get_op

    op = get_op(op_name)
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


# --------------------------------------------------------------------------
# canonical SDPA verification, derived from layer14.pt
# --------------------------------------------------------------------------


class _SdpaCapture:
    def __init__(self) -> None:
        self.q = self.k = self.v = self.o = None
        self.grads: dict[str, torch.Tensor] = {}
        self.calls = 0
        self.attrs: dict[str, Any] = {}
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [n for n in ("q", "k", "v", "o") if getattr(self, n) is None]
        missing += [n for n in ("dq", "dk", "dv", "do") if n not in self.grads]
        if missing:
            raise Level1Error(f"SDPA capture is incomplete, missing {missing}")


@contextmanager
def capture_sdpa() -> Iterator[_SdpaCapture]:
    """Watch the one SDPA call, its output, and every gradient around it."""
    capture = _SdpaCapture()
    original = torch.nn.functional.scaled_dot_product_attention

    def wrapper(query, key, value, *args, **kwargs):
        capture.calls += 1
        if capture.calls > 1:
            raise Level1Error("SDPA ran more than once inside one derivation")
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
        out = original(query, key, value, *args, **kwargs)
        capture.o = preserve_layout_cpu(out)
        if out.requires_grad:
            capture.handles.append(
                out.register_hook(
                    lambda grad: capture.grads.__setitem__("do", preserve_layout_cpu(grad))
                )
            )
        return out

    torch.nn.functional.scaled_dot_product_attention = wrapper
    try:
        yield capture
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        for handle in capture.handles:
            handle.remove()
        capture.handles.clear()


def derive_sdpa_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> _SdpaCapture:
    from ...levels.level3.replay import prepare_layer

    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Level1Error("the canonical derivation runs on CUDA and none is visible")
    layer, args, kwargs, grad_output, _dtype = prepare_layer(payload, device)
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_sdpa() as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()
    capture.identity = artifact.identity
    capture.source_hashes = {
        "content_hash": payload["content_hash"],
        "artifact_hash": payload["artifact_hash"],
    }
    return capture


def run_verify(
    source: Path,
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Check the new Level-1 task against the SDPA the model actually ran."""
    from evograd.ops import get_op
    from evograd.ops.level1.causal_gqa_attention.forward_ref import (
        causal_gqa_attention_forward_production,
        causal_gqa_attention_forward_ref,
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
    capture = derive_sdpa_invocation(source, device=device, snapshot_path=snapshot_path)
    snapshot = load_snapshot(snapshot_path)
    observed = snapshot["level1"]["causal_gqa_attention"]["configurations"][0]

    op = get_op("causal_gqa_attention")
    case = op.benchmark_workloads("qwen3_0_6b_observed")[0]
    base = op.tolerances[case.dtype][0]
    names = ("o", "dq", "dk", "dv")
    declared_tol = {name: op.tolerance_for(case, name) for name in names}

    tensors = {n: getattr(capture, n).to(device) for n in ("q", "k", "v")}
    do = capture.grads["do"].to(device)
    captured = {"o": capture.o.to(device), **{n: capture.grads[n].to(device) for n in ("dq", "dk", "dv")}}

    def run(forward):
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in tensors.items()}
        out = forward(leaves["q"], leaves["k"], leaves["v"])
        out.backward(do)
        return {
            "o": out.detach().clone(),
            **{f"d{n}": leaves[n].grad.detach().clone() for n in ("q", "k", "v")},
        }

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    production = run(causal_gqa_attention_forward_production)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    production_checks = {
        name: compare_tensors(
            production[name], captured[name], FORWARD_TOL if name == "o" else GRADIENT_TOL
        )
        for name in names
    }
    dense = run(causal_gqa_attention_forward_ref)
    dense_checks = {
        name: declared_gate(dense[name], captured[name], declared_tol[name], base)
        for name in names
    }
    del dense

    noise: dict[str, Any] = {"repeats": noise_repeats, "note": "production spelling vs itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [run(causal_gqa_attention_forward_production) for _ in range(noise_repeats)]
        noise["results"] = {
            name: _max_noise(
                _noise(passes[i][name], passes[0][name]) for i in range(1, noise_repeats)
            )
            for name in names
        }

    problems = []
    for field, expected in (
        ("is_causal", True),
        ("enable_gqa", True),
        ("dropout_p", 0.0),
        ("attn_mask_provided", False),
    ):
        if capture.attrs.get(field) != expected:
            problems.append(f"sdpa attrs.{field}: {capture.attrs.get(field)!r} != {expected!r}")
    if abs(float(capture.attrs["scale"]) - 1.0 / math.sqrt(case.dims["D"])) > 1e-12:
        problems.append(f"sdpa scale {capture.attrs['scale']} != 1/sqrt({case.dims['D']})")
    for name, entry in zip(("q", "k", "v"), observed["inputs"]):
        tensor = getattr(capture, name)
        if list(tensor.shape) != entry["shape"] or list(tensor.stride()) != entry["stride"]:
            problems.append(
                f"{name} layout {list(tensor.shape)}/{list(tensor.stride())} != harvested "
                f"{entry['shape']}/{entry['stride']}"
            )

    failures = [f"provenance: {p}" for p in problems]
    for label, group in (("production", production_checks), ("dense reference", dense_checks)):
        for name, record in group.items():
            if not record["within_tolerance"]:
                failures.append(
                    f"{label} {name}: max_rel_err_vs_scale="
                    f"{record.get('max_rel_err_vs_scale')}, required_t="
                    f"{record.get('required_t')}"
                )

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": "causal_gqa_attention",
        "identity": capture.identity,
        "source_hashes": capture.source_hashes,
        "snapshot_hash": snapshot["snapshot_hash"],
        "harvested_config_id": observed["config_id"],
        "frequency": observed["frequency"],
        "declared_dims": expected_dims,
        "canonical_declared_dims": case.dims,
        "sdpa_attrs": capture.attrs,
        "tolerances": {
            "production": {"forward": FORWARD_TOL, "gradient": GRADIENT_TOL},
            "dense_reference": {n: list(v) for n, v in declared_tol.items()},
            "declared_base": base,
        },
        "comparisons": {"production": production_checks, "dense_reference": dense_checks},
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
        f"[{report['status'].upper()}] {report['task']} against the captured SDPA call",
        f"  harvested configuration {report['harvested_config_id']} x{report['frequency']}",
        f"  snapshot {report['snapshot_hash'][:16]}...  "
        f"layer artifact {report['source_hashes']['artifact_hash'][:16]}...",
        f"  dims {report['declared_dims']}",
        f"  sdpa {report['sdpa_attrs']}",
        "",
    ]
    for label, group in report["comparisons"].items():
        lines.append(f"  {label}:")
        for name, record in group.items():
            lines.append(
                f"    {name:<4} rel {_fmt(record['max_rel_err_vs_scale'])}  "
                f"bitwise {record['bitwise_identical']}  stride_ok {record['stride_match']}  "
                f"required_t {_fmt(record.get('required_t'))}"
            )
    noise = report["noise_floor"]
    if noise.get("measured"):
        lines += ["", f"  measured noise floor ({noise['repeats']} runs):"]
        lines.append("    " + "  ".join(f"{n}={_fmt(v)}" for n, v in noise["results"].items()))
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# vocabulary-width behaviour, built on demand and dropped
# --------------------------------------------------------------------------


def run_loss_check(*, device: str = "cuda", snapshot_path: Path | None = None) -> dict[str, Any]:
    """Run the observed cross entropy at its real shape without persisting it.

    ``[4096, 151936]`` float32 logits are 2.5 GiB; a stored artifact of them
    would be larger than every other result in this directory put together, and
    they are reproducible from the seed. So they are built here, checked, and
    dropped.

    What is checked is behaviour, not a stored tensor: at the canonical shape a
    freshly initialised model's loss sits at ``ln(vocab)``, and the Level-4 smoke
    run recorded 12.1439 against ``ln(151936) = 11.93``.

    This is a **sanity check, not the equivalence proof**. It would pass for any
    implementation that got the scale right and the gradient wrong, because it
    never looks at a gradient and never touches the model's own tensors. The
    proof is :func:`run_cross_entropy_check`, which intercepts the model's
    actual call and compares loss and ``dlogits`` on the live tensors.
    """
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import oracle
    from evograd.ops import get_op

    op = get_op("cross_entropy")
    case = op.benchmark_workloads("qwen3_0_6b_observed")[0]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Level1Error("the canonical loss check runs on CUDA and none is visible")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    values = make_case_inputs(op, case, device=device)
    loss, grads = oracle(op, values)
    rows, cols = case.dims["rows"], case.dims["cols"]

    uniform = torch.zeros((rows, cols), device=device, dtype=torch.float32)
    uniform_loss = torch.nn.functional.cross_entropy(
        uniform, values["target"], ignore_index=-100, reduction="mean"
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    report = {
        "schema_version": "evograd-qwen3-level1-loss/1",
        "task": "cross_entropy",
        "dims": case.dims,
        "dtype": case.dtype,
        "logits_dtype": str(values["logits"].dtype),
        "target_dtype": str(values["target"].dtype),
        "loss": float(loss),
        "loss_is_finite": bool(torch.isfinite(loss)),
        "loss_is_scalar": list(loss.shape) == [],
        "gradient_names": sorted(grads),
        "dlogits_shape": list(grads["dlogits"].shape),
        "dlogits_all_finite": bool(torch.isfinite(grads["dlogits"]).all()),
        "uniform_logit_loss": float(uniform_loss),
        "ln_vocab": math.log(cols),
        "canonical_level4_loss": 12.14388656616211,
        "tensors_written": False,
        "diagnostics": {
            "note": "diagnostic only -- one unwarmed pass, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": peak,
        },
    }
    del values, grads, uniform
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def run_cross_entropy_check(*, device: str = "cuda", spec=None) -> dict[str, Any]:
    """Compare the Level-1 contract against the model's own cross entropy.

    The canonical Level-4 step is run on demand and its call to
    ``fixed_cross_entropy`` is intercepted. The declared Level-1 reference is
    then evaluated **on the same live tensors** -- the exact flattened float32
    logits and int64 labels Transformers passed -- and against the same upstream
    scalar, so what is compared is two implementations of one contract rather
    than two independently generated workloads.

    Nothing is persisted. The ``[4096, 151936]`` logits and their gradient are
    2.5 GiB each; they are compared inside the hooks that see them and dropped
    with the step. A loss magnitude near ``ln(vocab)`` is kept elsewhere as a
    sanity check, but it is not the equivalence proof -- it would pass for any
    implementation that got the scale right and the gradient wrong.
    """
    from evograd.ops.level1.cross_entropy.forward_ref import cross_entropy_forward_ref

    from ...levels.level4.model import build_model, make_inputs, require_transformers, training_step
    from ...levels.level3.replay import compare_tensors
    from ...levels.level4.spec import CANONICAL

    require_transformers()
    import transformers.loss.loss_utils as loss_utils

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Level1Error("the canonical cross-entropy check runs on CUDA and none is visible")
    # `spec` exists so a test can drive the same interception with a two-layer
    # model on CPU; the canonical run is the default and the only one whose
    # numbers are reported.
    if spec is None:
        spec = CANONICAL if device.startswith("cuda") else CANONICAL.replace(device=device)

    captured: dict[str, Any] = {"calls": 0}
    original = loss_utils.fixed_cross_entropy

    def _meta(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "contiguous": bool(tensor.is_contiguous()),
            "device": tensor.device.type,
        }

    def wrapper(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
        captured["calls"] += 1
        loss = original(source, target, num_items_in_batch, ignore_index, **kwargs)
        if captured["calls"] > 1:
            return loss
        captured["attrs"] = {
            "ignore_index": int(ignore_index),
            "reduction": "sum" if num_items_in_batch is not None else "mean",
            "num_items_in_batch": num_items_in_batch,
        }
        captured["logits"] = _meta(source)
        captured["target"] = _meta(target)
        captured["model_loss_meta"] = _meta(loss)
        with torch.no_grad():
            reference_loss = cross_entropy_forward_ref(source, target)
        captured["model_loss"] = float(loss.detach())
        captured["reference_loss"] = float(reference_loss.detach())
        captured["reference_loss_meta"] = _meta(reference_loss)
        del reference_loss

        def on_loss_grad(grad):
            # The upstream scalar this call receives. Captured rather than
            # assumed to be 1.0: the whole point is to feed the reference the
            # same one.
            captured["upstream"] = float(grad.detach())
            captured["upstream_meta"] = _meta(grad)
            return None

        def on_logits_grad(grad):
            # `source` and `target` are still alive here, so the reference can
            # be differentiated against exactly what the model differentiated.
            # A backward hook runs with grad mode off, so the reference would
            # otherwise be built without a graph to differentiate.
            with torch.enable_grad():
                leaf = source.detach().clone().requires_grad_(True)
                reference = cross_entropy_forward_ref(leaf, target)
                upstream = torch.full_like(reference, captured.get("upstream", 1.0))
                (reference_grad,) = torch.autograd.grad(
                    reference, leaf, grad_outputs=upstream
                )
            captured["dlogits"] = compare_tensors(reference_grad, grad, 0.0)
            captured["model_dlogits_meta"] = _meta(grad)
            captured["reference_dlogits_meta"] = _meta(reference_grad)
            del leaf, reference, upstream, reference_grad
            return None

        if loss.requires_grad:
            loss.register_hook(on_loss_grad)
        if source.requires_grad:
            source.register_hook(on_logits_grad)
        return loss

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_utils.fixed_cross_entropy = wrapper
    try:
        model = build_model(spec)
        input_ids, labels = make_inputs(spec)
        outputs = training_step(model, input_ids, labels)
        step_loss = float(outputs.loss.detach())
    finally:
        loss_utils.fixed_cross_entropy = original
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    del model, outputs, input_ids, labels
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    for key in ("logits", "target", "dlogits", "upstream"):
        if key not in captured:
            raise Level1Error(f"the cross-entropy interception never saw {key!r}")

    op = get_op_cross_entropy()
    case = op.benchmark_workloads("qwen3_0_6b_observed")[0]
    declared = {"rows": captured["logits"]["shape"][0], "cols": captured["logits"]["shape"][1]}
    # A shrunken debug run compares against its own shape, not the canonical one.
    expected_dims = case.dims if spec.is_canonical else declared

    dlogits = captured["dlogits"]
    loss_abs = abs(captured["reference_loss"] - captured["model_loss"])
    failures = []
    if declared != expected_dims:
        failures.append(f"observed dims {declared} != declared {expected_dims}")
    if captured["logits"]["dtype"] != "torch.float32":
        failures.append(f"logits dtype {captured['logits']['dtype']} != torch.float32")
    if captured["target"]["dtype"] != "torch.int64":
        failures.append(f"target dtype {captured['target']['dtype']} != torch.int64")
    if captured["attrs"]["ignore_index"] != -100 or captured["attrs"]["reduction"] != "mean":
        failures.append(f"attrs {captured['attrs']} != ignore_index -100 / mean")
    if loss_abs != 0.0:
        failures.append(f"loss differs by {loss_abs}")
    for field in ("shape_match", "dtype_match", "stride_match", "actual_all_finite",
                  "expected_all_finite"):
        if not dlogits.get(field):
            failures.append(f"dlogits {field} is False")
    if not dlogits.get("bitwise_identical"):
        failures.append(
            f"dlogits not bitwise identical (max_abs_err {dlogits.get('max_abs_err')})"
        )

    return {
        "schema_version": "evograd-qwen3-level1-cross-entropy/1",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "task": "cross_entropy",
        "workload_id": spec.workload_id,
        "canonical": spec.is_canonical,
        "declared_dims": expected_dims,
        "canonical_declared_dims": case.dims,
        "observed_dims": declared,
        "attrs": captured["attrs"],
        "signature": {
            "logits": captured["logits"],
            "target": captured["target"],
            "model_loss": captured["model_loss_meta"],
            "reference_loss": captured["reference_loss_meta"],
            "upstream": captured["upstream_meta"],
            "model_dlogits": captured["model_dlogits_meta"],
            "reference_dlogits": captured["reference_dlogits_meta"],
        },
        "loss": {
            "model": captured["model_loss"],
            "reference": captured["reference_loss"],
            "abs_error": loss_abs,
            "step_loss": step_loss,
            "upstream_scalar": captured["upstream"],
        },
        "dlogits": dlogits,
        "tensors_written": False,
        "diagnostics": {
            "note": "diagnostic only -- one canonical step, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": peak,
        },
    }


def get_op_cross_entropy():
    from evograd.ops import get_op

    return get_op("cross_entropy")


def summarize_cross_entropy(report: dict[str, Any]) -> str:
    signature = report["signature"]
    loss = report["loss"]
    dlogits = report["dlogits"]
    lines = [
        f"[{report['status'].upper()}] cross_entropy against the model's own call",
        f"  workload {report['workload_id']}  canonical={report['canonical']}",
        f"  logits {signature['logits']['shape']} {signature['logits']['dtype']} "
        f"stride {signature['logits']['stride']}",
        f"  target {signature['target']['shape']} {signature['target']['dtype']}   "
        f"attrs {report['attrs']}",
        "",
        f"  loss   model {loss['model']!r}   reference {loss['reference']!r}   "
        f"abs error {loss['abs_error']:.3e}",
        f"         upstream scalar {loss['upstream_scalar']}   "
        f"step loss {loss['step_loss']!r}",
        f"  dlogits {dlogits['actual']['shape']} {dlogits['actual']['dtype']}  "
        f"stride_ok {dlogits['stride_match']}  bitwise {dlogits['bitwise_identical']}  "
        f"max_abs_err {dlogits['max_abs_err']:.3e}  finite "
        f"{dlogits['actual_all_finite'] and dlogits['expected_all_finite']}",
        "",
        f"  no tensors written ({report['tensors_written']}); peak "
        f"{(report['diagnostics']['peak_allocated_bytes'] or 0) / 2**30:.2f} GiB",
    ]
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


def summarize_loss(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"cross_entropy at the observed shape {report['dims']} "
            f"[{report['dtype']}]  (no tensors written)",
            f"  logits {report['logits_dtype']}  target {report['target_dtype']}  "
            f"loss scalar: {report['loss_is_scalar']}  finite: {report['loss_is_finite']}",
            f"  loss {report['loss']:.6f}   uniform-logit loss "
            f"{report['uniform_logit_loss']:.6f}   ln(vocab) {report['ln_vocab']:.6f}",
            f"  canonical Level-4 loss for reference: {report['canonical_level4_loss']}",
            f"  dlogits {report['dlogits_shape']} all finite: "
            f"{report['dlogits_all_finite']}",
        ]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level1.mapping",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    table = sub.add_parser("mapping", help="print the six-task mapping table")
    table.add_argument("--report", type=Path, default=None)

    calibrate = sub.add_parser("calibrate", help="measure a task's required tolerance")
    calibrate.add_argument("--op", default="causal_gqa_attention")
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--report", type=Path, default=None)

    verify = sub.add_parser("verify", help="check causal_gqa_attention against the model")
    verify.add_argument("--source", type=Path, default=Path("results/qwen3-level4/layer14.pt"))
    verify.add_argument("--device", default="cuda")
    verify.add_argument("--noise-repeats", type=int, default=4)
    verify.add_argument("--report", type=Path, default=None)

    loss = sub.add_parser(
        "loss", help="ln(vocab) sanity check at the observed shape (not the proof)"
    )
    loss.add_argument("--device", default="cuda")
    loss.add_argument("--report", type=Path, default=None)

    ce = sub.add_parser(
        "cross-entropy", help="compare the Level-1 contract against the model's own call"
    )
    ce.add_argument("--device", default="cuda")
    ce.add_argument("--report", type=Path, default=None)
    return parser


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mapping":
            report = mapping()
            print(summarize_mapping(report))
            _write(args.report, report)
            return 0
        if args.command == "calibrate":
            report = run_calibration(args.op, device=args.device)
            print(summarize_calibration(report))
            _write(args.report, report)
            return 0
        if args.command == "loss":
            report = run_loss_check(device=args.device)
            print(summarize_loss(report))
            _write(args.report, report)
            return 0
        if args.command == "cross-entropy":
            report = run_cross_entropy_check(device=args.device)
            print(summarize_cross_entropy(report))
            _write(args.report, report)
            return 0 if report["status"] == "pass" else 1
        report = run_verify(
            args.source, device=args.device, noise_repeats=args.noise_repeats
        )
    except (Level1Error, ArtifactError) as exc:
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(summarize_verify(report))
    _write(args.report, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
