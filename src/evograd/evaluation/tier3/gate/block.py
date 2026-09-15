"""The block-scope correctness gate, and the policy it is held to.

Architecture-neutral: it receives the reference and candidate results of one
block (every declared output, input gradient and parameter gradient), the
invocation they came from and a frozen policy, and returns a verdict. It does
not know what the block is, which sites it has, or how many parameters it
should have -- the reference result says what exists, and the adapter's
declaration says what was expected.

The policy is derived from **trusted controls only**, before any candidate is
judged, and for the exact patch set the candidate has: the native block's own
repeatability (the hardware floor), the structural-identity control (bitwise
forward, backward bounded independently by native repeatability) and the
bound-pair control (the declared operator through ``bind``, the route an
evolved kernel takes). A compiled trusted control can be added by request.
Thresholds are ``max(observed maximum, floor) * margin`` per (kind, role)
group, exactly as the model gate derives its envelope
(:func:`evograd.evaluation.tier3.gate.numerics.derive_envelope`), and a
candidate's own error never contributes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping

import torch

from evograd.evaluation.tier3 import block as _block
from evograd.evaluation.tier3.gate import numerics

POLICY_SCHEMA = "evograd-t3-block-policy/2"
GATE_NAME = "evograd-t3-block-gate/1"

#: Default trusted controls a policy is derived from.
DEFAULT_CONTROLS = ("structural_identity", "bound_pair_identity")

#: Result-name prefixes. ``role_of`` strips none of them, so every output and
#: input gradient forms its own group; parameter gradients group by the role
#: patterns the model gate already uses (``q_proj``, ``down_proj``, ...).
OUT = "out:"
GRAD_IN = "grad_in:"
GRAD = "grad:"


def patch_key(kernels) -> str:
    return "+".join(sorted(kernels.patched)) if kernels.patched else "native"


def _entry_problems(entry: Mapping[str, Any]) -> list[str]:
    """An invalid calibration is never an admissible numerical policy."""
    problems = []
    controls = entry.get("controls") or {}
    if entry.get("controls_ok") is not True:
        problems.append("calibration controls did not pass")
    if not {"native_repeatability", "structural_identity"}.issubset(controls):
        problems.append("native and structural controls are required")
    if any(c.get("ok") is not True for c in controls.values()):
        problems.append("a calibration control failed")
    envelopes = entry.get("envelopes") or {}
    if not envelopes:
        problems.append("no calibrated envelopes")
    for group, envelope in envelopes.items():
        for metric in numerics.GATED_METRICS:
            value = envelope.get("threshold", {}).get(metric)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0):
                problems.append(f"invalid threshold {group}/{metric}")
    return problems


# ── samples ──────────────────────────────────────────────────────────────────


def _samples(reference: _block.BlockResult, candidate: _block.BlockResult) -> tuple[list[dict], list[dict]]:
    """Every comparable tensor pair as a streaming sample, plus structural problems.

    Tensors are moved to the CPU one at a time; nothing is retained.
    """
    samples: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    def one(name: str, actual, expected, kind: str) -> None:
        if expected is None and actual is None:
            return
        if expected is None or actual is None:
            problems.append({"name": name, "reason": "present in one result only",
                             "candidate": actual is not None, "reference": expected is not None})
            return
        a = actual.detach().to("cpu")
        b = expected.detach().to("cpu")
        if a.shape != b.shape or a.dtype != b.dtype:
            problems.append({"name": name, "reason": "shape or dtype differs",
                             "candidate": _block.tensor_meta(a), "reference": _block.tensor_meta(b)})
            return
        if a.stride() != b.stride():
            problems.append({"name": name, "reason": "stride differs",
                             "candidate": list(a.stride()), "reference": list(b.stride())})
        stats = numerics.compare_tensor(name, a, b, kind=kind)
        samples.append(stats.to_dict())
        del a, b

    for path in reference.outputs:
        one(OUT + path, candidate.outputs.get(path), reference.outputs[path], numerics.KIND_OUTPUT)
    for path in reference.input_grads:
        one(GRAD_IN + path, candidate.input_grads.get(path), reference.input_grads[path],
            numerics.KIND_GRADIENT)
    for name in reference.param_grads:
        one(GRAD + name, candidate.param_grads.get(name), reference.param_grads[name],
            numerics.KIND_GRADIENT)
    return samples, problems


def _worst(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    top = max(samples, key=lambda s: s["rel_l2"])
    return {k: top[k] for k in ("name", "group", "rel_l2", "max_abs_over_rms", "bitwise")}


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, float]] = {}
    for s in samples:
        g = by_group.setdefault(s["group"], {"rel_l2": 0.0, "max_abs_over_rms": 0.0, "tensors": 0})
        g["rel_l2"] = max(g["rel_l2"], s["rel_l2"])
        g["max_abs_over_rms"] = max(g["max_abs_over_rms"], s["max_abs_over_rms"])
        g["tensors"] += 1
    return {"tensors": len(samples), "bitwise": all(s["bitwise"] for s in samples),
            "worst": _worst(samples), "by_group": by_group}


# ── the candidate verdict ────────────────────────────────────────────────────


def check_candidate(reference: _block.BlockResult, candidate: _block.BlockResult, *,
                    invocation: _block.BlockInvocation, parameters: Mapping[str, torch.Tensor],
                    kernels, policy: dict[str, Any] | None, adapter=None) -> dict[str, Any]:
    """Every declared output, input gradient and parameter gradient, judged.

    In order: structure (the block returned what it declared, with the shapes,
    dtypes and strides the reference has), presence (every gradient the
    reference produced is produced; none the reference lacks appears), aliases
    (the parameter list is the reference's, same names, same order),
    finiteness, input mutation, metadata equality, then the numerical envelope
    from the policy entry for this exact patch set. No global norm stands in
    for a per-tensor verdict; the per-group summary is reported beside them.
    """
    verdict: dict[str, Any] = {"gate": GATE_NAME, "ok": False}

    def fail(stage: str, reason: str) -> dict[str, Any]:
        verdict["failed_at"] = stage
        verdict["reason"] = reason
        return verdict

    verdict["structure"] = {"ok": candidate.structure == reference.structure,
                            "candidate": candidate.structure, "reference": reference.structure}
    if not verdict["structure"]["ok"]:
        return fail("structure", "the block's return structure differs from the native block's")

    expected_names = list(reference.param_grads)
    actual_names = list(parameters)
    verdict["aliases"] = {"ok": actual_names == expected_names,
                          "parameters": len(actual_names),
                          "missing": sorted(set(expected_names) - set(actual_names)),
                          "unexpected": sorted(set(actual_names) - set(expected_names)),
                          "order_preserved": actual_names == expected_names}
    if not verdict["aliases"]["ok"]:
        return fail("aliases", "the installed block does not carry the native parameter list")

    present = [n for n, g in candidate.param_grads.items() if g is not None]
    expected_present = [n for n, g in reference.param_grads.items() if g is not None]
    missing = sorted(set(expected_present) - set(present))
    unexpected = sorted(set(present) - set(expected_present))
    missing_inputs = [p for p, g in reference.input_grads.items()
                      if g is not None and candidate.input_grads.get(p) is None]
    verdict["gradient_presence"] = {"ok": not missing and not unexpected and not missing_inputs,
                                    "expected": len(expected_present), "present": len(present),
                                    "missing": missing, "unexpected": unexpected,
                                    "missing_input_gradients": missing_inputs}
    if not verdict["gradient_presence"]["ok"]:
        return fail("gradient_presence",
                    f"missing gradients {missing + missing_inputs}, unexpected {unexpected}")

    mutated = list(getattr(candidate, "mutated_inputs", []) or [])
    verdict["input_mutation"] = {"ok": not mutated, "mutated": mutated}
    if mutated:
        return fail("input_mutation", f"the block wrote into its inputs: {mutated}")

    meta_ok = candidate.metadata == reference.metadata
    verdict["metadata_outputs"] = {"ok": meta_ok, "candidate": _jsonable(candidate.metadata),
                                   "reference": _jsonable(reference.metadata)}
    if not meta_ok:
        return fail("metadata_outputs", "a non-tensor output differs from the native block's")

    samples, problems = _samples(reference, candidate)
    verdict["structure"]["problems"] = problems
    if problems:
        return fail("structure", f"{problems[0]['name']}: {problems[0]['reason']}")
    non_finite = [s["name"] for s in samples if not s["finite"]]
    verdict["finiteness"] = {"ok": not non_finite, "non_finite": non_finite}
    if non_finite:
        return fail("finiteness", f"non-finite results: {non_finite[:4]}")

    verdict["comparison"] = _summary(samples)
    verdict["per_tensor"] = {s["name"]: {k: s[k] for k in ("group", "rel_l2", "max_abs_over_rms",
                                                            "bitwise", "cosine", "max_abs_err")}
                             for s in samples}

    key = patch_key(kernels)
    entry = ((policy or {}).get("entries") or {}).get(key)
    verdict["policy"] = {"patch_key": key, "available": entry is not None,
                         "policy_hash": (policy or {}).get("policy_hash")}
    if entry is None:
        return fail("no_policy", f"the frozen policy has no entry for patch set {key!r}; "
                                 "calibrate it before judging this provider")
    problems = _entry_problems(entry)
    if policy.get("schema") != POLICY_SCHEMA:
        problems.append("policy schema requires recalibration")
    if policy.get("policy_hash") != policy_hash(policy):
        problems.append("policy_hash does not match the policy's content")
    if adapter is not None and policy.get("case_hash") != adapter.case.case_hash:
        problems.append("policy case does not match the block")
    if problems:
        return fail("invalid_policy", "; ".join(problems))
    envelopes = {g: numerics.GroupEnvelope(**e) for g, e in entry["envelopes"].items()}
    checked = numerics.check_against(envelopes, samples)
    verdict["envelope"] = {"ok": checked["ok"], "checked": checked["checked"],
                           "exceeded": checked["exceeded"][:16],
                           "thresholds": {g: e.threshold for g, e in envelopes.items()}}
    if not checked["ok"]:
        first = checked["exceeded"][0]
        if "metric" in first and "value" in first:
            reason = (f"{first['name']} {first['metric']} {first['value']:.3g} exceeds "
                      f"{first['threshold']:.3g} ({first['ratio']:.1f}x)")
        else:
            reason = f"{first['name']}: {first.get('reason')}"
        return fail("envelope", reason)
    verdict["ok"] = True
    return verdict


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor": _block.tensor_meta(value)}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _calibration_problems(reference, result, samples, problems):
    """Check controls before their errors are allowed into any envelope."""
    problems = list(problems)
    if result.structure != reference.structure:
        problems.append({"reason": "return structure differs"})
    for field in ("outputs", "input_grads", "param_grads"):
        if list(getattr(result, field)) != list(getattr(reference, field)):
            problems.append({"reason": f"{field} names/order differ"})
    if result.mutated_inputs:
        problems.append({"reason": "control mutated its inputs"})
    if result.metadata != reference.metadata:
        problems.append({"reason": "control metadata differs"})
    for sample in samples:
        if (not sample["finite"] or not sample["reference_finite"]
                or any(not math.isfinite(sample[m]) for m in numerics.GATED_METRICS)):
            problems.append({"name": sample["name"], "reason": "non-finite control"})
    return problems


# ── the reference verdict ────────────────────────────────────────────────────


def check_reference(adapter, built, invocation, result: _block.BlockResult, *, reset,
                    noise_repeats: int, capture: dict[str, Any] | None,
                    run: Callable[[], _block.BlockResult]) -> dict[str, Any]:
    """The native block: finite, repeatable, and -- when captured -- the model's.

    Repeatability compares ``noise_repeats`` further independent repetitions
    with the first, streaming, and keeps the worst per group; it is the
    hardware floor every policy starts from. Capture agreement uses the replay
    tolerances the Level-3 replay is already held to (one unit roundoff on the
    forward, one epsilon on gradients, scale-normalized).
    """
    verdict: dict[str, Any] = {"gate": "native_reference", "ok": False}
    first = result.cpu()
    non_finite = [OUT + p for p, t in first.outputs.items() if not bool(torch.isfinite(t).all())]
    non_finite += [GRAD_IN + p for p, t in first.input_grads.items()
                   if t is not None and not bool(torch.isfinite(t).all())]
    non_finite += [GRAD + n for n, t in first.param_grads.items()
                   if t is not None and not bool(torch.isfinite(t).all())]
    verdict["finiteness"] = {"ok": not non_finite, "non_finite": non_finite}
    missing = [n for n, t in first.param_grads.items() if t is None]
    verdict["gradient_presence"] = {"ok": not missing, "expected": len(first.param_grads),
                                    "present": len(first.param_grads) - len(missing), "missing": missing}
    mutated = list(getattr(result, "mutated_inputs", []) or [])
    verdict["input_mutation"] = {"ok": not mutated, "mutated": mutated}

    repeat_samples: list[dict[str, Any]] = []
    for _ in range(max(0, noise_repeats)):
        again = run()
        samples, problems = _samples(first, again)
        problems = _calibration_problems(first, again, samples, problems)
        # Bound native drift by the existing layer replay contract, independently
        # of the subsequently fitted noise envelope. Noise cannot justify itself.
        from evograd.evaluation.workloads.common.replay import FORWARD_TOL, GRADIENT_TOL
        for sample in samples:
            tolerance = FORWARD_TOL if sample["kind"] == numerics.KIND_OUTPUT else GRADIENT_TOL
            if sample["max_abs_err"] > tolerance * max(sample["ref_absmax"], 1e-30):
                problems.append({"name": sample["name"], "reason": "native drift exceeds replay tolerance"})
        del again
        if problems:
            verdict["repeatability"] = {"ok": False, "problems": problems}
            break
        repeat_samples.extend(samples)
    if "repeatability" not in verdict:
        verdict["repeatability"] = {"ok": True, "repeats": noise_repeats,
                                    **_summary(repeat_samples)}
    verdict["repeatability_samples"] = repeat_samples

    if capture is not None:
        from evograd.evaluation.workloads.common.replay import (
            FORWARD_TOL, GRADIENT_TOL, compare_tensors)

        records: dict[str, Any] = {}
        ok = True
        for path, tensor in first.outputs.items():
            want = capture["outputs"].get(path)
            if want is None:
                records[OUT + path] = {"within_tolerance": False, "reason": "not captured"}
                ok = False
                continue
            rec = compare_tensors(tensor, want, FORWARD_TOL)
            records[OUT + path] = _trim(rec)
            ok = ok and rec["within_tolerance"]
        for path, tensor in first.input_grads.items():
            want = capture["input_grads"].get(path)
            if tensor is None or want is None:
                records[GRAD_IN + path] = {"within_tolerance": False, "reason": "missing"}
                ok = False
                continue
            rec = compare_tensors(tensor, want, GRADIENT_TOL)
            records[GRAD_IN + path] = _trim(rec)
            ok = ok and rec["within_tolerance"]
        worst = None
        for name, tensor in first.param_grads.items():
            want = capture["param_grads"].get(name)
            if tensor is None or want is None:
                records[GRAD + name] = {"within_tolerance": False, "reason": "missing"}
                ok = False
                continue
            rec = compare_tensors(tensor, want, GRADIENT_TOL)
            ok = ok and rec["within_tolerance"]
            value = rec.get("max_rel_err_vs_scale")
            if worst is None or (value is not None and value > (worst[1] or -1.0)):
                worst = (name, value, _trim(rec))
        verdict["capture_agreement"] = {
            "ok": ok, "forward_tol": FORWARD_TOL, "gradient_tol": GRADIENT_TOL,
            "outputs_and_input_gradients": records,
            "param_grads": {"count": len(first.param_grads),
                            "worst": {"name": worst[0], "max_rel_err_vs_scale": worst[1],
                                      **worst[2]} if worst else None},
        }
    else:
        verdict["capture_agreement"] = {"ok": True, "skipped": True,
                                        "reason": "config-derived case: no capture to agree with"}

    for stage in ("finiteness", "gradient_presence", "input_mutation", "repeatability",
                  "capture_agreement"):
        if not verdict[stage]["ok"]:
            verdict["failed_at"] = stage
            verdict["reason"] = f"native block failed {stage}: {verdict[stage]}"[:400]
            return verdict
    verdict["ok"] = True
    return verdict


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    keep = ("within_tolerance", "bitwise_identical", "max_abs_err", "ref_absmax",
            "max_rel_err_vs_scale", "max_rel_err_elementwise", "tolerance",
            "shape_match", "dtype_match", "stride_match", "zero_reference_mismatch")
    return {k: record[k] for k in keep if k in record}


# ── the policy ───────────────────────────────────────────────────────────────


def calibrate_policy(adapter, patch_sets: Mapping[str, Any], *, ops: Mapping[str, Any],
                     device: str, controls: tuple[str, ...] = DEFAULT_CONTROLS,
                     noise_repeats: int = 3, margin: float = numerics.SAFETY_MARGIN,
                     environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the envelope for every patch set, from trusted controls only.

    ``patch_sets`` maps a patch key to any kernel set having that patch set
    (the candidate's; only its ``patched`` sites are read, never its kernels).
    For each: the native block's repeatability samples, plus each named
    control built by the adapter for exactly those sites and compared against
    the native reference. Structural forward must be bitwise; its backward
    must fit an envelope derived ONLY from native repetitions. A failed control
    aborts calibration, before its errors can increase any threshold.
    """
    if noise_repeats < 1:
        raise _block.BlockError("block calibration requires at least one independent native repeat")
    if not math.isfinite(margin) or margin < 1:
        raise _block.BlockError("calibration margin must be finite and at least one")
    if any(k.patched for k in patch_sets.values()) and "structural_identity" not in controls:
        raise _block.BlockError("block calibration requires structural_identity")
    reference, built, invocation = _block.native_reference(adapter, device=device)
    reset = _block._reset_fn(adapter, built, None)
    native_check = check_reference(
        adapter, built, invocation, reference, reset=reset, noise_repeats=noise_repeats,
        capture=adapter.capture_reference(),
        run=lambda: _block.run_vjp(built.module, invocation, built.parameters, reset=reset),
    )
    if not native_check["ok"]:
        raise _block.BlockError(f"native calibration failed: {native_check.get('reason')}")
    noise = native_check["repeatability_samples"]
    native_envelopes = numerics.derive_envelope(noise, margin=margin)
    del built, invocation

    entries: dict[str, Any] = {}
    for key, kernels in patch_sets.items():
        if not kernels.patched:
            continue
        expected = adapter.expected_invocations(kernels)
        all_samples = list(noise)
        control_reports: dict[str, Any] = {"native_repeatability": {
            "ok": True, "repeats": max(1, noise_repeats), **_summary(noise)}}
        resolved = adapter.controls(kernels, ops, controls)
        if set(resolved) != set(controls):
            raise _block.BlockError(f"{key}: requested controls {controls}, got {tuple(resolved)}")
        for label, control in resolved.items():
            cbuilt = adapter.build(device=device)
            installed = adapter.install(cbuilt, control)
            cinv = adapter.prepare(cbuilt, device=device)
            creset = _block._reset_fn(adapter, cbuilt, installed)
            local = adapter.local_checks(cbuilt, control, cinv)
            cresult = _block.run_vjp(cbuilt.module, cinv, cbuilt.parameters, reset=creset)
            counts = installed.counters.snapshot()
            samples, problems = _samples(reference, cresult)
            problems = _calibration_problems(reference, cresult, samples, problems)
            del cresult, cbuilt, cinv
            report = {"ok": not problems and counts == expected, "invocations": counts,
                      "patched": list(control.patched), "problems": problems, **_summary(samples)}
            report["local_check"] = local
            if local is not None:
                report["ok"] = report["ok"] and local.get("ok") is True
            if label == "structural_identity":
                forward_exact = all(s["bitwise"] for s in samples if s["kind"] == numerics.KIND_OUTPUT)
                backward = numerics.check_against(native_envelopes, [
                    s for s in samples if s["kind"] == numerics.KIND_GRADIENT])
                report["forward_bitwise"] = forward_exact
                report["backward_native_envelope"] = backward
                report["bitwise_required"] = "forward_only"
                report["ok"] = report["ok"] and forward_exact and backward["ok"]
            if not report["ok"]:
                raise _block.BlockError(
                    f"{key}: calibration control {label} failed; no policy emitted: "
                    f"{json.dumps(report, default=str)[:1600]}")
            control_reports[label] = report
            all_samples.extend(samples)
        envelopes = numerics.derive_envelope(all_samples, margin=margin)
        entries[key] = {
            "patch_set": {"patched": list(kernels.patched), "expected_invocations": expected},
            "controls": control_reports,
            "controls_ok": all(r["ok"] for r in control_reports.values()),
            "samples": len(all_samples),
            "envelopes": {g: e.to_dict() for g, e in envelopes.items()},
        }
    policy = {
        "schema": POLICY_SCHEMA,
        "gate": GATE_NAME,
        "case_id": adapter.case.case_id,
        "case_hash": adapter.case.case_hash,
        "environment": environment or {},
        "environment_hash": numerics.fingerprint_hash(environment or {}),
        "margin": margin,
        "derived_from": ["native_repeatability", *controls],
        "noise_repeats": max(1, noise_repeats),
        "floors": numerics.KIND_THRESHOLD_FLOOR,
        "entries": entries,
    }
    policy["policy_hash"] = policy_hash(policy)
    return policy


def policy_hash(policy: Mapping[str, Any]) -> str:
    body = {k: v for k, v in policy.items() if k != "policy_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def check_policy_binding(policy: Mapping[str, Any], adapter, environment: Mapping[str, Any] | None) -> list[str]:
    """Why a frozen policy may not be applied to this case, if it may not."""
    problems = []
    if policy.get("schema") != POLICY_SCHEMA:
        problems.append(f"schema {policy.get('schema')!r} is not {POLICY_SCHEMA!r}")
    if policy.get("case_hash") != adapter.case.case_hash:
        problems.append(f"policy is for case {policy.get('case_id')!r} ({str(policy.get('case_hash'))[:12]}), "
                        f"not {adapter.case.case_id!r} ({adapter.case.case_hash[:12]})")
    if policy.get("policy_hash") != policy_hash(policy):
        problems.append("policy_hash does not match the policy's content")
    for key, entry in (policy.get("entries") or {}).items():
        problems.extend(f"{key}: {p}" for p in _entry_problems(entry))
    if environment is not None and policy.get("environment"):
        mismatch = numerics.environment_mismatch(policy["environment"], dict(environment))
        problems.extend(f"environment: {m}" for m in mismatch)
    return problems


__all__ = [
    "DEFAULT_CONTROLS",
    "GATE_NAME",
    "POLICY_SCHEMA",
    "calibrate_policy",
    "check_candidate",
    "check_policy_binding",
    "check_reference",
    "patch_key",
    "policy_hash",
]
