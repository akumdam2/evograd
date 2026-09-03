"""Replay one captured decoder layer, alone, and check it against the full model.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.replay \
        --artifact results/qwen3-level4/layer14.pt \
        --report results/qwen3-level4/layer14-replay.json

This process constructs exactly one ``Qwen3DecoderLayer``. It never builds
``Qwen3Model`` or ``Qwen3ForCausalLM``, and the report proves it by scanning the
live heap for instances of those classes rather than asserting it in prose.

The layer is given the captured weights and the captured arguments, run forward,
and then backward with the *real* upstream gradient the full-model backward
delivered to it. Output, input gradient and every parameter gradient are compared
against what the full model produced.

**Tolerances are stated, not tuned.** BF16 carries a 7-bit explicit mantissa, so
the spacing between representable numbers near 1.0 -- machine epsilon -- is
``2^-7``, and the unit roundoff, half that spacing, is ``2^-8``. The forward path
is deterministic -- same weights, same inputs, same kernels -- and is held to one
unit roundoff. Backward is not:
SDPA's backward accumulates with atomics, so a rerun of the *same* layer differs
from itself. That noise is measured here, by replaying several times and
comparing the replays to each other, and reported next to the replay-versus-
capture error so the reader can see which is larger. The tolerance is a fixed
constant either way; if the error exceeds it, the report says FAIL.

Errors are reported relative to the *scale of the reference tensor*
(``max|a-b| / max|b|``) rather than elementwise. An elementwise relative error is
meaningless where the reference is near zero, and a gradient tensor is full of
near-zero entries; the scale-normalized form asks the question that matters --
how large is the disagreement compared to the signal. Note what this does *not*
say: a small-magnitude tensor is not near BF16 underflow -- the smallest normal
BF16 is about 1.2e-38 -- and BF16's relative precision is the same at every
exponent. The reason to prefer the scale-normalized metric is that entries far
below a tensor's own maximum contribute nothing to what the tensor is used for,
not that they are badly represented.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .artifact import ArtifactError, LayerArtifact, tensor_meta, to_device
from ...levels.level4.model import DTYPES, require_transformers

#: Bumped from /1 with the tolerance-key rename and the stricter
#: ``within_tolerance``: a /1 report used ``bf16_ulp`` for what is really the
#: unit roundoff, and could pass a comparison with a mismatched stride.
REPORT_SCHEMA = "evograd-qwen3-layer-replay-report/2"

#: BF16 machine epsilon: the spacing between representable values near 1.0. The
#: format has a 7-bit explicit mantissa, so consecutive values near 1.0 differ by
#: ``2^-7``.
BF16_EPS = 2.0**-7
#: Unit roundoff: the largest relative error a single correct rounding can
#: introduce, half the spacing above.
BF16_UNIT_ROUNDOFF = 2.0**-8

#: The forward path is deterministic -- same weights, same inputs, same kernels
#: -- so a correct replay should differ by at most a single rounding, if at all.
FORWARD_TOL = BF16_UNIT_ROUNDOFF
#: Backward reorders reductions inside SDPA, so a correct replay can land one
#: representable step away rather than one rounding.
GRADIENT_TOL = BF16_EPS

#: Elementwise relative error is only meaningful where the reference is not
#: dwarfed by the tensor's own scale.
ELEMENTWISE_FLOOR_FRACTION = 1e-2


class ReplayError(RuntimeError):
    """The replay could not be performed, or disagreed with the capture."""


def validate_noise_repeats(value: Any) -> int:
    """0 to skip the measurement, or at least 2 to make one.

    A single replay has nothing to be compared against, so ``1`` would silently
    report no noise floor while looking like it measured one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"noise_repeats must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"noise_repeats must not be negative, got {value}")
    if value == 1:
        raise ValueError(
            "noise_repeats=1 cannot measure anything: one replay has nothing to "
            "be compared with. Pass 0 to skip the measurement, or >= 2 to make it."
        )
    return value


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def build_config(arch: dict[str, Any]):
    require_transformers()
    from transformers import Qwen3Config

    return Qwen3Config(**arch, use_cache=False, attn_implementation="sdpa")


def build_single_layer(arch: dict[str, Any], layer_index: int, *, device: str, dtype: str):
    """One ``Qwen3DecoderLayer``. Nothing above it is constructed."""
    require_transformers()
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    config = build_config(arch)
    torch_dtype = DTYPES[dtype]
    layer = Qwen3DecoderLayer(config, layer_index)
    # A decoder layer owns no floating-point buffers -- unlike the full model,
    # whose rotary `inv_freq` must stay float32 -- so casting the whole module is
    # safe here. Checked rather than assumed.
    float_buffers = [name for name, buf in layer.named_buffers() if buf.is_floating_point()]
    if float_buffers:  # pragma: no cover - version dependent
        raise ReplayError(
            f"this Qwen3DecoderLayer has floating-point buffers {float_buffers}; "
            "casting the module would change their precision"
        )
    layer = layer.to(device=device, dtype=torch_dtype)
    layer.train()
    layer.gradient_checkpointing = False
    return layer


def live_model_instances() -> dict[str, int]:
    """Count live Qwen3 objects on the heap.

    The standalone claim is worth only as much as its evidence, and "we did not
    call the constructor" is weaker evidence than "no such object exists".
    """
    require_transformers()
    from transformers.models.qwen3 import modeling_qwen3 as modeling

    classes = {
        name: getattr(modeling, name)
        for name in ("Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention")
        if hasattr(modeling, name)
    }
    counts = {name: 0 for name in classes}
    gc.collect()
    for obj in gc.get_objects():
        try:
            for name, cls in classes.items():
                if type(obj) is cls:
                    counts[name] += 1
                    break
        except ReferenceError:  # pragma: no cover - weakref proxies
            continue
    return counts


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, tol: float) -> dict[str, Any]:
    """One structured comparison. Differences are computed in float32 so the
    measurement itself does not round."""
    result: dict[str, Any] = {
        "shape_match": list(actual.shape) == list(expected.shape),
        "dtype_match": actual.dtype == expected.dtype,
        "stride_match": list(actual.stride()) == list(expected.stride()),
        "actual": tensor_meta(actual),
        "expected": tensor_meta(expected),
        "tolerance": tol,
    }
    if not result["shape_match"]:
        result.update(within_tolerance=False, reason="shape mismatch")
        return result

    a = actual.detach().to("cpu", torch.float32)
    b = expected.detach().to("cpu", torch.float32)
    diff = (a - b).abs()
    ref_absmax = float(b.abs().max())
    max_abs = float(diff.max())
    result["bitwise_identical"] = bool(
        actual.dtype == expected.dtype
        and torch.equal(actual.detach().to("cpu"), expected.detach().to("cpu"))
    )
    result["ref_absmax"] = ref_absmax
    result["max_abs_err"] = max_abs
    # A zero reference has no scale to normalize by. If the replay is also zero
    # the tensors agree exactly; if it is not, the disagreement is total, and
    # dividing by zero-scale to get 0.0 -- which is what a naive guard does --
    # would report a perfect match for the worst possible result.
    if ref_absmax > 0:
        result["max_rel_err_vs_scale"] = max_abs / ref_absmax
        result["zero_reference_mismatch"] = False
    elif max_abs == 0:
        result["max_rel_err_vs_scale"] = 0.0
        result["zero_reference_mismatch"] = False
    else:
        # Deliberately null rather than infinity: the ratio is undefined, and
        # `Infinity` is not valid JSON for anything but Python's own parser.
        result["max_rel_err_vs_scale"] = None
        result["zero_reference_mismatch"] = True
    # Elementwise relative error, but only where the reference is not negligible
    # against the tensor's own scale.
    floor = ref_absmax * ELEMENTWISE_FLOOR_FRACTION
    mask = b.abs() > floor
    if bool(mask.any()):
        result["max_rel_err_elementwise"] = float((diff[mask] / b[mask].abs()).max())
        result["elementwise_compared"] = int(mask.sum())
    else:  # an all-zero reference, or one whose entries are all at the floor
        result["max_rel_err_elementwise"] = None
        result["elementwise_compared"] = 0
    result["actual_all_finite"] = bool(torch.isfinite(a).all())
    result["expected_all_finite"] = bool(torch.isfinite(b).all())
    relative = result["max_rel_err_vs_scale"]
    result["within_tolerance"] = bool(
        result["shape_match"]
        and result["dtype_match"]
        and result["stride_match"]
        and result["actual_all_finite"]
        # A non-finite *reference* means the capture itself is unusable; a replay
        # that reproduced a NaN would otherwise be scored as agreement.
        and result["expected_all_finite"]
        and relative is not None
        and relative <= tol
    )
    return result


def _why(record: dict[str, Any]) -> str:
    """Say which of the verdict's conditions actually failed."""
    reasons: list[str] = []
    for key, label in (
        ("shape_match", "shape"),
        ("dtype_match", "dtype"),
        ("stride_match", "stride"),
        ("actual_all_finite", "replay is finite"),
        ("expected_all_finite", "capture is finite"),
    ):
        if not record.get(key, True):
            reasons.append(label)
    relative = record.get("max_rel_err_vs_scale")
    if record.get("zero_reference_mismatch"):
        reasons.append(
            f"the captured tensor is all zeros but the replay is not "
            f"(max_abs_err={record.get('max_abs_err')})"
        )
    elif relative is not None and relative > record.get("tolerance", 0.0):
        reasons.append(
            f"max_rel_err_vs_scale={relative:.3e} > {record['tolerance']:.3e}"
        )
    return "; ".join(reasons) or "unspecified"


def declared_gate(actual, expected, tolerance, base: float) -> dict[str, Any]:
    """Compare under a declaration's own ``allclose`` gate.

    :func:`compare_tensors` supplies the structural checks and the descriptive
    numbers; the numerical verdict is the exact test the benchmark harness runs,
    so a result that would fail the harness cannot pass here. ``required_t`` is
    the smallest base tolerance that would have accepted it, directly comparable
    with a calibration report.

    Shared by every task derived from the layer artifact, so the three cannot
    drift into gating three different ways.
    """
    atol, rtol = tolerance
    record = compare_tensors(actual, expected, rtol)
    record.update(atol=atol, rtol=rtol, declared_base=base)
    if not record["shape_match"]:
        return record
    a = actual.detach().to("cpu", torch.float32)
    b = expected.detach().to("cpu", torch.float32)
    diff = (a - b).abs()
    ma, mr = atol / base, rtol / base
    record["required_t"] = float((diff / (ma + mr * b.abs())).max())
    record["allclose"] = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    record["within_tolerance"] = bool(
        record["shape_match"]
        and record["dtype_match"]
        and record["stride_match"]
        and record["actual_all_finite"]
        and record["expected_all_finite"]
        and record["allclose"]
    )
    return record


def required_tolerance(actual, expected, multiplier=(1.0, 1.0)) -> dict[str, Any]:
    """The smallest declaration-level base tolerance that would accept a result.

    The gate is ``|a-b| <= ma*atol + mr*rtol*|b|``; written with one base ``t``
    in both slots that is ``t >= max(|a-b| / (ma + mr*|b|))``. Measured against
    the multiplier the declaration actually applies, so a multiplier that is
    doing real work shows a smaller base requirement rather than hiding behind a
    looser global tolerance.
    """
    a = actual.detach().to("cpu", torch.float32)
    b = expected.detach().to("cpu", torch.float32)
    diff = (a - b).abs()
    scale = float(b.abs().max())
    ma, mr = multiplier
    return {
        "required_t": float((diff / (ma + mr * b.abs())).max()),
        "required_t_unweighted": float((diff / (1.0 + b.abs())).max()),
        "max_abs_err": float(diff.max()),
        "ref_absmax": scale,
        "max_rel_err_vs_scale": float(diff.max()) / scale if scale > 0 else None,
        "minimal_atol_multiplier": {
            f"{t:.0e}": max(1.0, float(((diff - t * b.abs()).clamp(min=0.0) / t).max()))
            for t in (2e-3, 5e-3, 1e-2, 2e-2)
        },
    }


def _noise(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Scale-normalized difference between two replays, or ``None`` if the
    reference has no scale to normalize by and the two still differ."""
    x = a.detach().to("cpu", torch.float32)
    y = b.detach().to("cpu", torch.float32)
    scale = float(y.abs().max())
    diff = float((x - y).abs().max())
    if scale > 0:
        return diff / scale
    return 0.0 if diff == 0 else None


def _max_noise(values) -> float | None:
    """``None`` propagates: one undefined comparison makes the floor unknown."""
    values = list(values)
    if any(value is None for value in values):
        return None
    return max(values) if values else 0.0


# --------------------------------------------------------------------------
# the replay
# --------------------------------------------------------------------------


def _one_pass(layer, hidden_states: torch.Tensor, kwargs: dict, grad_output: torch.Tensor):
    layer.zero_grad(set_to_none=True)
    leaf = hidden_states.detach().clone().requires_grad_(True)
    output = layer(leaf, **kwargs)
    tensor = output[0] if isinstance(output, tuple) else output
    tensor.backward(grad_output)
    grads = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.grad is not None}
    return tensor.detach().clone(), leaf.grad.detach().clone(), grads


def prepare_layer(payload: dict[str, Any], device: str):
    """Rebuild the captured layer and its arguments on ``device``.

    Shared by every task derived from a layer artifact, so the MLP and the
    attention block cannot drift into loading the same artifact two different
    ways.
    """
    identity = payload["identity"]
    dtype = str(payload["output"].dtype).replace("torch.", "")
    layer = build_single_layer(payload["arch"], identity["layer_index"], device=device, dtype=dtype)
    layer.load_state_dict({k: v.to(device) for k, v in payload["state_dict"].items()}, strict=True)
    args = to_device(payload["args"], device)
    kwargs = to_device(payload["kwargs"], device)
    grad_output = to_device(payload["grad_output"], device)
    return layer, args, kwargs, grad_output, dtype


def verify_layout(cpu_value: Any, device_value: Any, path: str = "$") -> list[str]:
    """Layout and dtype must survive the move to the device."""
    problems: list[str] = []
    if torch.is_tensor(cpu_value):
        if not torch.is_tensor(device_value):  # pragma: no cover - defensive
            return [f"{path}: expected a tensor"]
        if cpu_value.dtype != device_value.dtype:
            problems.append(f"{path}: dtype {device_value.dtype} != captured {cpu_value.dtype}")
        if tuple(cpu_value.shape) != tuple(device_value.shape):
            problems.append(f"{path}: shape {tuple(device_value.shape)} != {tuple(cpu_value.shape)}")
        if tuple(cpu_value.stride()) != tuple(device_value.stride()):
            problems.append(
                f"{path}: stride {tuple(device_value.stride())} != captured "
                f"{tuple(cpu_value.stride())}"
            )
        return problems
    if isinstance(cpu_value, (tuple, list)):
        for index, item in enumerate(cpu_value):
            problems += verify_layout(item, device_value[index], f"{path}[{index}]")
        return problems
    if isinstance(cpu_value, dict):
        for key, item in cpu_value.items():
            problems += verify_layout(item, device_value[key], f"{path}.{key}")
        return problems
    if cpu_value != device_value:  # pragma: no cover - scalars pass through
        problems.append(f"{path}: {device_value!r} != captured {cpu_value!r}")
    return problems


def run_replay(
    artifact: LayerArtifact,
    *,
    device: str = "cuda",
    noise_repeats: int = 4,
) -> dict[str, Any]:
    require_transformers()
    noise_repeats = validate_noise_repeats(noise_repeats)
    payload = artifact.payload
    identity = artifact.identity
    # Both hashes, unconditionally. A caller cannot opt out of either: content
    # alone would accept a relabelled file, identity alone a corrupted one.
    hashes = artifact.verify()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ReplayError(
            "the canonical replay runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug replay"
        )

    dtype = str(payload["output"].dtype).replace("torch.", "")
    layer = build_single_layer(
        payload["arch"], identity["layer_index"], device=device, dtype=dtype
    )
    layer.load_state_dict({k: v.to(device) for k, v in payload["state_dict"].items()}, strict=True)

    args = to_device(payload["args"], device)
    kwargs = to_device(payload["kwargs"], device)
    grad_output = to_device(payload["grad_output"], device)
    layout_problems = verify_layout(payload["args"], args, "$args")
    layout_problems += verify_layout(payload["kwargs"], kwargs, "$kwargs")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output, grad_input, param_grads = _one_pass(layer, args[0], kwargs, grad_output)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None

    # --- comparison against the full model -----------------------------
    expected_output = payload["output"].to(device)
    expected_grad_input = payload["grad_input"].to(device)
    expected_param_grads = payload["param_grads"]

    comparisons = {
        "output": compare_tensors(output, expected_output, FORWARD_TOL),
        "grad_input": compare_tensors(grad_input, expected_grad_input, GRADIENT_TOL),
    }
    per_param: dict[str, Any] = {}
    missing = [name for name in expected_param_grads if name not in param_grads]
    unexpected = [name for name in param_grads if name not in expected_param_grads]
    non_finite = []
    for name, expected in sorted(expected_param_grads.items()):
        if name not in param_grads:
            continue
        record = compare_tensors(param_grads[name], expected.to(device), GRADIENT_TOL)
        if not record.get("actual_all_finite", True):
            non_finite.append(name)
        per_param[name] = record
    comparisons["param_grads"] = per_param

    # --- noise floor ---------------------------------------------------
    noise = {"repeats": noise_repeats, "note": "replay compared against itself"}
    if noise_repeats == 0:
        noise["measured"] = False
    else:
        noise["measured"] = True
        passes = [_one_pass(layer, args[0], kwargs, grad_output) for _ in range(noise_repeats)]
        noise["output"] = _max_noise(
            _noise(passes[i][0], passes[0][0]) for i in range(1, noise_repeats)
        )
        noise["grad_input"] = _max_noise(
            _noise(passes[i][1], passes[0][1]) for i in range(1, noise_repeats)
        )
        worst_name, worst_value = None, -1.0
        for name in passes[0][2]:
            value = _max_noise(
                _noise(passes[i][2][name], passes[0][2][name]) for i in range(1, noise_repeats)
            )
            if value is None:
                worst_name, worst_value = name, None
                break
            if value >= worst_value:
                worst_name, worst_value = name, value
        noise["param_grads_max"] = worst_value
        noise["param_grads_worst"] = worst_name

    # --- verdict -------------------------------------------------------
    failures: list[str] = []
    if layout_problems:
        failures += [f"layout: {p}" for p in layout_problems]
    if missing:
        failures.append(f"missing parameter gradients after replay: {missing}")
    if unexpected:
        failures.append(f"parameter gradients not present in the capture: {unexpected}")
    if non_finite:
        failures.append(f"non-finite replay gradients: {non_finite}")
    for name in ("output", "grad_input"):
        record = comparisons[name]
        if not record["within_tolerance"]:
            failures.append(f"{name}: {_why(record)}")
    for name, record in per_param.items():
        if not record["within_tolerance"]:
            failures.append(f"param_grad {name}: {_why(record)}")

    worst_param = max(
        per_param.items(),
        # ``None`` means the ratio is undefined, which is worse than any number.
        key=lambda kv: (
            kv[1].get("max_rel_err_vs_scale") is None,
            kv[1].get("max_rel_err_vs_scale") or 0.0,
        ),
        default=(None, {}),
    )
    instances = live_model_instances()
    if instances.get("Qwen3ForCausalLM") or instances.get("Qwen3Model"):
        failures.append(
            f"the replay process holds full-model objects: {instances}; the "
            "standalone claim would be false"
        )

    parameters = list(layer.named_parameters())
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "identity": identity,
        "artifact": hashes,
        "construction": {
            "module_class": type(layer).__name__,
            "constructed_full_model": False,
            "live_instances": instances,
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(p.numel() for _, p in parameters),
            "parameter_bytes": sum(p.numel() * p.element_size() for _, p in parameters),
            "device": device,
            "dtype": dtype,
        },
        "tolerances": {
            "bf16_eps": BF16_EPS,
            "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
            "forward": FORWARD_TOL,
            "forward_meaning": "one BF16 unit roundoff (2^-8)",
            "gradient": GRADIENT_TOL,
            "gradient_meaning": "one BF16 epsilon, the spacing near 1.0 (2^-7)",
            "metric": "max|a-b| / max|b| (reference scale)",
            "elementwise_floor_fraction": ELEMENTWISE_FLOOR_FRACTION,
        },
        "comparisons": comparisons,
        "summary": {
            "output_max_rel_err_vs_scale": comparisons["output"].get("max_rel_err_vs_scale"),
            "output_bitwise_identical": comparisons["output"].get("bitwise_identical"),
            "grad_input_max_rel_err_vs_scale": comparisons["grad_input"].get(
                "max_rel_err_vs_scale"
            ),
            "param_grads_compared": len(per_param),
            "param_grads_worst": worst_param[0],
            "param_grads_worst_max_rel_err_vs_scale": worst_param[1].get(
                "max_rel_err_vs_scale"
            ),
            "param_grads_all_within_tolerance": all(
                r["within_tolerance"] for r in per_param.values()
            ),
            "missing_param_grads": missing,
            "non_finite_param_grads": non_finite,
        },
        "noise_floor": noise,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
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
    """``None`` is a real outcome here -- an undefined ratio -- not a gap."""
    return "undefined" if value is None else f"{value:.3e}"


def summarize(report: dict[str, Any]) -> str:
    summary = report["summary"]
    construction = report["construction"]
    noise = report["noise_floor"]
    lines = [
        f"[{report['status'].upper()}] replay of {report['identity']['module_path']} "
        f"(layer {report['identity']['layer_index']})",
        f"  artifact content {report['artifact']['content_hash'][:16]}...",
        f"  built {construction['module_class']} only: live instances "
        f"{construction['live_instances']}",
        f"  {construction['parameter_tensors']} parameter tensors, "
        f"{construction['parameter_elements']:,} elements, "
        f"{construction['parameter_bytes'] / 2**20:.1f} MiB",
        "",
        f"  tolerance forward {report['tolerances']['forward']:.6g} "
        f"({report['tolerances']['forward_meaning']}), "
        f"gradient {report['tolerances']['gradient']:.6g} "
        f"({report['tolerances']['gradient_meaning']})",
        f"  metric: {report['tolerances']['metric']}",
        f"  output        rel {_fmt(summary['output_max_rel_err_vs_scale'])}   "
        f"bitwise identical: {summary['output_bitwise_identical']}",
        f"  grad_input    rel {_fmt(summary['grad_input_max_rel_err_vs_scale'])}",
        f"  param grads   {summary['param_grads_compared']} compared, worst "
        f"{summary['param_grads_worst']} rel "
        f"{_fmt(summary['param_grads_worst_max_rel_err_vs_scale'])}",
    ]
    if noise.get("measured"):
        lines += [
            "",
            f"  measured noise floor (replay vs replay, {noise['repeats']} runs):",
            f"    output {_fmt(noise['output'])}   grad_input {_fmt(noise['grad_input'])}   "
            f"param grads {_fmt(noise['param_grads_max'])} ({noise['param_grads_worst']})",
        ]
    else:
        lines += ["", "  noise floor not measured (--noise-repeats 0)"]
    diagnostics = report["diagnostics"]
    lines += [
        "",
        f"  forward+backward {diagnostics['forward_backward_wall_time_s'] * 1e3:.1f} ms, "
        f"peak {(diagnostics['peak_allocated_bytes'] or 0) / 2**20:.1f} MiB "
        "(diagnostic only)",
    ]
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level3.replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--noise-repeats",
        type=int,
        default=4,
        help="replays used to measure the noise floor: 0 to skip, or at least 2",
    )
    parser.add_argument("--expect-workload-id", default=None)
    parser.add_argument("--expect-manifest-hash", default=None)
    parser.add_argument("--expect-layer", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_noise_repeats(args.noise_repeats)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        artifact = LayerArtifact.load(args.artifact)
        artifact.verify_identity(
            workload_id=args.expect_workload_id,
            manifest_hash=args.expect_manifest_hash,
            layer_index=args.expect_layer,
        )
        report = run_replay(artifact, device=args.device, noise_repeats=args.noise_repeats)
    except (ReplayError, ArtifactError) as exc:
        print(f"replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(summarize(report))
    if args.report is not None:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
