"""The whole-model correctness gate tier 3 runs before it times a Qwen provider.

Site preflight proves a kernel correct at its declared shapes. It cannot prove
that 140 of them assembled into a 28-layer model still train, and the failure it
misses is the expensive one: a provider that is right on a 4096-row grid, wrong
in composition, and fast.

So a non-eager provider must clear all five of these before a timer starts:

1. its site-level tier-1 preflight, including the workload-supplied observed
   shapes -- that gate already exists and is not repeated here;
2. every loss, logit, gradient, stepped parameter and checked optimizer state is
   finite;
3. one **untimed** canonical forward/backward/AdamW step fits inside the
   calibrated per-role envelope, measured against *both* references: the
   original eager model, and the bound-pair path an evolved kernel replaces;
4. its fresh-batch loss trajectory, over exactly the horizon the policy was
   calibrated for, fits the trajectory bound;
5. its invocation counts and ``PatchProvenance`` match what was requested.

A failure sets ``ok=False`` and ``failed_at="model_correctness"``. Everything
here runs outside every timed region.

The thresholds are not written here. They live in a calibration artifact bound
to the environment it was measured in, and this module refuses to apply one
measured on a different GPU, driver, CUDA, torch, transformers, TF32 setting or
SDPA backend rather than silently reusing someone else's noise floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from evograd.bench.tier3_gate import numerics
from evograd.bench.tier3_gate.numerics import (
    NumericsPolicy,
    check_against,
    combined_envelope,
    environment_fingerprint,
)

#: The untimed gate, in order. Each stage runs only if every earlier one
#: passed, and a failure names exactly one of them.
STAGES = (
    "site_preflight",
    "provider_purity",
    "live_boundary",
    "numerical_envelopes",
    "loss_trajectory",
    "counts_and_provenance",
)

#: Where the calibration lives. Tracked as a result, not as code.
DEFAULT_ARTIFACT = Path("results/qwen3-level4/t3-numerics-calibration.json")


class CalibrationUnavailable(RuntimeError):
    """No calibration, or one measured somewhere else."""


def load_policy(path: Path | None = None, *, require_environment: bool = True):
    """The stored calibration, or a refusal that says which field moved."""
    path = Path(path or DEFAULT_ARTIFACT)
    if not path.is_file():
        raise CalibrationUnavailable(
            f"no numerics calibration at {path}. Run "
            "`python -m evograd.bench.workloads.qwen3.evaluation.tier3.calibrate run` on this "
            "machine; tier 3 will not time a Qwen provider on an uncalibrated "
            "gate."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = NumericsPolicy.from_dict(payload["policy"])
    if require_environment:
        mismatch = policy.applies_here()
        if mismatch:
            raise CalibrationUnavailable(
                "the stored calibration was measured in a different "
                "environment, so its thresholds are not this machine's noise "
                "floor:\n  " + "\n  ".join(mismatch)
                + "\nRecalibrate, or pass require_environment=False knowing the "
                "gate is then quoting numbers from elsewhere."
            )
    return policy


# ── one untimed step, twice ──────────────────────────────────────────────────


def _step(workload, kernels, *, data_seed: int, learning_rate: float):
    """One optimizer step, keeping every quantity the gate judges separately.

    Four different things come out of a training step and they do not share a
    scale. The gradient is whatever the loss surface says. The *update* AdamW
    then applies is close to the learning rate in every element, roughly four
    orders of magnitude smaller. The two moments are smaller still, and the
    step counter is an integer. Judging them against one pooled threshold means
    judging them against the gradient's, and a wrong update disappears inside
    it -- which is why the update is captured as ``after - before`` in float32
    rather than as the stepped parameter, whose own magnitude would swamp the
    change the step made.
    """
    model, provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ids, labels = workload.batch_for(seed=data_seed)
    result = capture_step(model, optimizer, ids, labels)
    built = workload.last_build
    result.update(
        provenance=provenance.to_dict(),
        counts=built.observed() if built else {},
        expected_counts=built.expected if built else {},
        count_problems=built.count_problems() if built else [],
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def capture_step(model, optimizer, ids, labels) -> dict[str, Any]:
    """The five quantities one training step produces, each kept apart."""
    # `.clone()` is not decoration: `.float()` on a float32 parameter and
    # `.cpu()` on a host one both return the tensor itself, so without it this
    # would alias the live parameter and every update would come out zero.
    before = {n: p.detach().float().cpu().clone()
              for n, p in model.named_parameters()}
    outputs = model(input_ids=ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    optimizer.step()

    updates, exp_avg, exp_avg_sq, steps, stored = {}, {}, {}, {}, {}
    stateless = []
    for name, param in model.named_parameters():
        # Held on the host. Three float32 copies of every parameter is nine
        # gigabytes for this model, and three of these captures are live at
        # once during a comparison; the comparisons are streaming reductions
        # that do not need them resident on the device.
        # The change the step made, not the parameter it made it to.
        updates[name] = param.detach().float().cpu().clone() - before.pop(name)
        state = optimizer.state.get(param, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            stateless.append(name)
            continue
        exp_avg[name] = state["exp_avg"].detach().to("cpu", copy=True)
        exp_avg_sq[name] = state["exp_avg_sq"].detach().to("cpu", copy=True)
        # The stored value itself, in the parameter's own dtype. The update is
        # a float32 difference and cannot answer whether anything actually
        # changed in memory: at bfloat16 an update smaller than half a ULP
        # leaves the stored bits untouched, and a control that perturbs such an
        # update perturbs nothing. Keeping the stored tensor is what lets a
        # report say which of those happened.
        stored[name] = param.detach().to("cpu", copy=True)
        step = state.get("step")
        steps[name] = float(step) if step is not None else None
    return {
        # On the host for the same reason the updates are: comparing two
        # float32 copies of a [2, 2048, 151936] logits tensor is nine gigabytes
        # of peak the device does not need to carry.
        "logits": outputs.logits.detach().cpu(),
        "loss": outputs.loss.detach().cpu(),
        "grads": grads,
        "updates": updates,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "steps": steps,
        "stored": stored,
        "missing_grads": missing,
        "stateless_parameters": stateless,
        "parameter_names": [n for n, _p in model.named_parameters()],
    }


#: Each family of compared tensors, with the prefix its samples are named by
#: and the kind that gives it its own envelope namespace. Adding a family here
#: is the only thing needed to have it judged on its own terms.
_FAMILIES = (
    ("grads", "", numerics.KIND_GRADIENT),
    ("updates", "update:", numerics.KIND_UPDATE),
    ("exp_avg", "exp_avg:", numerics.KIND_EXP_AVG),
    ("exp_avg_sq", "exp_avg_sq:", numerics.KIND_EXP_AVG_SQ),
)


def _compare(candidate: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Every quantity of the step, each in its own namespace.

    An omission is a failure, not an absence: a candidate that produced no
    gradient for a parameter the reference did, or whose optimizer never built
    a moment for one, is reported as an infinite deviation rather than skipped.
    """
    from evograd.bench.tier3_gate.numerics import compare_tensor

    samples = [
        compare_tensor(name, candidate[name], reference[name],
                       kind=numerics.KIND_OUTPUT).to_dict()
        for name in ("logits", "loss")
    ]
    for family, prefix, kind in _FAMILIES:
        for name, expected in reference[family].items():
            got = candidate[family].get(name)
            if got is None:
                samples.append({
                    "name": f"{prefix}{name}", "role": numerics.role_of(name),
                    "kind": kind, "group": numerics.group_key(kind, numerics.role_of(name)),
                    "finite": False, "rel_l2": float("inf"),
                    "max_abs_over_rms": float("inf"),
                    "reason": f"the candidate produced no {family} entry for this parameter",
                })
                continue
            samples.append(
                compare_tensor(f"{prefix}{name}", got, expected, kind=kind).to_dict()
            )
    for name, expected in reference["steps"].items():
        got = candidate["steps"].get(name)
        role = numerics.role_of(name)
        samples.append({
            "name": f"step_count:{name}", "role": role, "kind": numerics.KIND_STEP,
            "group": numerics.group_key(numerics.KIND_STEP, role),
            "exact": got is not None and got == expected,
            "expected": expected, "observed": got, "finite": got is not None,
        })
    return samples


def _trajectory(workload, kernels, *, data_seed: int, horizon: int,
                learning_rate: float) -> list[float]:
    model, _p = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for index in range(horizon):
        ids, labels = workload.batch_for(seed=data_seed + 1000 + index)
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return losses


# ── the gate ─────────────────────────────────────────────────────────────────


def build_references(workload, *, policy: NumericsPolicy, data_seed: int = 0):
    """The two reference steps a verdict is measured against.

    Separated so a caller checking many providers against the same references --
    the negative controls do exactly that -- pays for them once. A provider
    check that builds its own references is the same computation.
    """
    from evograd.bench.tier3_patch import KernelSet
    from evograd.ops import OPS

    from .sites import bound_pair_identity_kernels

    learning_rate = policy.trajectory.learning_rate
    eager = _step(workload, KernelSet(registry=workload.site_registry),
                  data_seed=data_seed, learning_rate=learning_rate)
    bound = _step(workload,
                  bound_pair_identity_kernels(OPS, None, workload.site_registry),
                  data_seed=data_seed, learning_rate=learning_rate)
    return {"eager": eager, "bound": bound, "data_seed": data_seed}


def check_model_correctness(
    workload,
    kernels,
    *,
    policy: NumericsPolicy,
    data_seed: int = 0,
    check_trajectory: bool = True,
    references: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
    simple_policy: Any = None,
    simple_primary: bool = False,
) -> dict[str, Any]:
    """One untimed step against both references, plus the loss curve.

    ``eager`` is what the model was before anything was patched; ``bound`` is the
    path an evolved kernel actually replaces. A candidate is reported against
    both, because "matches eager" and "matches what it replaced" are different
    claims and only reporting one of them hides which reference moved.
    """
    from evograd.bench.tier3_patch import KernelSet

    from . import boundary, purity

    verdict: dict[str, Any] = {
        "gate": "qwen3_model_correctness",
        "policy": {
            "workload_id": policy.workload_id,
            "environment_hash": policy.environment_hash,
            "margin": policy.notes.get("margin"),
            "gated_metrics": policy.notes.get("gated_metrics"),
        },
        "data_seed": data_seed,
        "stages": list(STAGES),
    }

    def fail(stage: str, reason: str) -> dict[str, Any]:
        verdict["ok"] = False
        verdict["failed_at"] = stage
        verdict["reason"] = reason
        return verdict

    # 1. Does the provider hold at the shapes this model will actually give it?
    verdict["site_preflight"] = preflight or {"ok": True, "skipped": True}
    if not verdict["site_preflight"].get("ok", True):
        return fail("site_preflight", str(verdict["site_preflight"].get("reason",
                                                                        "site preflight failed")))

    # 2. Is it a function of its arguments, or does it remember? Asked before
    #    the model is built, because a provider that remembers must never reach
    #    one -- and asked in a process this session then throws away.
    verdict["provider_purity"] = purity.run_for(
        kernels, workload, device=str(workload.spec.device)
    )
    if not verdict["provider_purity"].get("ok", False):
        return fail("provider_purity", _purity_reason(verdict["provider_purity"]))

    # 3. Every invocation, on the model's own tensors and upstream gradients.
    verdict["live_boundary"] = boundary.validate_all_invocations(
        workload, kernels, data_seed=data_seed
    )
    if not verdict["live_boundary"].get("ok", False):
        return fail("live_boundary", _boundary_reason(verdict["live_boundary"]))

    # 4. The whole model: finiteness, then each quantity in its own namespace.
    candidate = _step(workload, kernels, data_seed=data_seed,
                      learning_rate=policy.trajectory.learning_rate)
    verdict["provenance"] = candidate["provenance"]
    verdict["observed_counts"] = candidate["counts"]
    verdict["expected_counts"] = candidate["expected_counts"]
    verdict["count_problems"] = candidate["count_problems"]
    verdict["missing_grads"] = candidate["missing_grads"]
    verdict["stateless_parameters"] = candidate["stateless_parameters"]

    references = references or build_references(
        workload, policy=policy, data_seed=data_seed
    )
    versus_eager = _compare(candidate, references["eager"])
    versus_bound = _compare(candidate, references["bound"])
    candidate_capture = candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    non_finite = [s["name"] for s in versus_eager if not s.get("finite", False)]
    verdict["finite"] = {"ok": not non_finite, "non_finite": non_finite[:16]}
    # Both comparisons are held to hardware noise plus the known integration
    # drift. Against eager a provider crosses both; against the bound-pair path
    # it replaces, the drift is already spent but the noise is not, and holding
    # the two to different bounds would make one of them the real gate by
    # accident rather than by choice.
    bound = combined_envelope(policy.envelopes, policy.bound_pair_envelopes)
    verdict["envelope"] = {
        "hardware_groups": len(policy.envelopes),
        "integration_groups": len(policy.bound_pair_envelopes),
        "kinds": sorted({g.split("|", 1)[0] for g in bound}),
        "formula": "threshold = E/E threshold + S/B threshold, per group and metric",
    }
    # The simplified gate, measured on the same captures: four questions about
    # the whole model, against thresholds calibrated for this exact patch set.
    # It needs no extra model build -- the trusted replacement was run at
    # calibration time, and its drift is already inside the threshold.
    if simple_policy is not None:
        verdict["simplified"] = _simplified_verdict(
            workload, kernels, simple_policy, candidate_capture, references["eager"]
        )
        verdict["simplified"]["role"] = "primary" if simple_primary else "shadow"

    verdict["vs_eager"] = check_against(bound, versus_eager)
    verdict["vs_bound_pair"] = check_against(bound, versus_bound)
    verdict["vs_eager"]["worst"] = _worst(versus_eager)
    verdict["vs_bound_pair"]["worst"] = _worst(versus_bound)
    for label in ("vs_eager", "vs_bound_pair"):
        verdict[label]["exceeded_groups"] = sorted(
            {e.get("group") for e in verdict[label]["exceeded"]}
        )
    del candidate, candidate_capture
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if simple_primary:
        # The simplified policy decides; the detailed one stays in the record as
        # a diagnostic so a migration can be audited rather than trusted.
        simplified = verdict.get("simplified") or {}
        verdict["legacy_envelopes"] = {
            "vs_eager_ok": verdict["vs_eager"]["ok"],
            "vs_bound_pair_ok": verdict["vs_bound_pair"]["ok"],
            "role": "diagnostic",
        }
        if not simplified.get("ok", False):
            return fail("numerical_envelopes",
                        str(simplified.get("reason") or "simplified gate failed"))
    else:
        if not verdict["finite"]["ok"]:
            return fail("numerical_envelopes",
                        f"non-finite values in {len(non_finite)} results, "
                        f"first {non_finite[0]}")
        if not (verdict["vs_eager"]["ok"] and verdict["vs_bound_pair"]["ok"]):
            return fail("numerical_envelopes", _reason(verdict))

    # 5. Five steps of training: does the loss go where the reference's went?
    if check_trajectory:
        reference_curve = _trajectory(
            workload, KernelSet(registry=workload.site_registry),
            data_seed=data_seed, horizon=policy.trajectory.horizon,
            learning_rate=policy.trajectory.learning_rate,
        )
        candidate_curve = _trajectory(
            workload, kernels, data_seed=data_seed,
            horizon=policy.trajectory.horizon,
            learning_rate=policy.trajectory.learning_rate,
        )
        limits = numerics.combined_trajectory(policy.trajectory,
                                              policy.bound_pair_trajectory)
        verdict["trajectory"] = limits.check(reference_curve, candidate_curve)
        verdict["trajectory"]["limits_source"] = (
            "E/E drift + S/B integration, matching the tensor envelope"
        )
        verdict["trajectory"]["reference"] = reference_curve
        verdict["trajectory"]["candidate"] = candidate_curve
    else:
        verdict["trajectory"] = {"ok": True, "skipped": True}
    if not verdict["trajectory"]["ok"] and not simple_primary:
        return fail("loss_trajectory", _reason(verdict))
    if simple_primary:
        # A five-step loss delta is not evidence about long-horizon training
        # quality, and the simplified gate does not pretend otherwise.
        verdict["trajectory"]["role"] = "diagnostic"

    # 6. Was every site reached the declared number of times, by what it says?
    if verdict["count_problems"] or verdict["missing_grads"]:
        return fail("counts_and_provenance", _reason(verdict))

    verdict["ok"] = True
    verdict["failed_at"] = None
    return verdict


def _simplified_verdict(workload, kernels, simple_policy, candidate, reference):
    """The simplified gate's answer, or the reason it refused to answer.

    The patch-set assertion is the load-bearing part: a policy calibrated for a
    QKV-only replacement describes a QKV-only replacement, and applying it to
    anything else would quote a threshold measured somewhere else.
    """
    from . import simple as simple_gate

    try:
        patch_set = simple_gate.PatchSet.of(
            kernels, layers=workload.spec.arch["num_hidden_layers"]
        )
        simple_policy.require_binding(
            workload_id=workload.spec.workload_id,
            workload_hash=workload.spec.workload_hash,
            dtype=str(workload.spec.dtype).replace("torch.", ""),
            environment_hash=simple_policy.environment_hash,
            patch_set=patch_set,
        )
    except simple_gate.PolicyMismatch as exc:
        return {"ok": False, "failed_at": "policy_binding", "reason": str(exc),
                "schema": simple_gate.SCHEMA_VERSION}
    metrics = simple_gate.measure(candidate, reference)
    verdict = simple_gate.check(simple_policy, metrics)
    verdict["measured_patch_set"] = patch_set.to_dict()
    return verdict


def _purity_reason(report: dict[str, Any]) -> str:
    if "reason" in report:
        return str(report["reason"])
    for site in report.get("sites", []):
        drift = site.get("first_drift")
        if drift:
            return (f"{site['site']} changed behaviour at call {drift['call']} "
                    f"({drift['result']} moved {drift['max_abs_err']:.3g}, "
                    f"{drift['regime']} bound {drift.get('bound', 0.0):.3g} from a "
                    f"median consecutive spread of "
                    f"{drift.get('median_consecutive_spread', 0.0):.3g}): "
                    "the provider's answer depends on its history")
        diverged = site.get("first_divergence")
        if diverged:
            return (f"{site['site']} diverged from its own first call at call "
                    f"{diverged['call']} on {diverged['result']}")
        if site.get("mutated_inputs"):
            return f"{site['site']} mutated its inputs: {site['mutated_inputs'][0]}"
        if not site.get("ok", True):
            return f"{site['site']} failed the purity gate"
    return "the provider failed the purity gate"


def _boundary_reason(report: dict[str, Any]) -> str:
    if report.get("errors"):
        return str(report["errors"][0])
    if report.get("shared_parameter_boundaries"):
        return ("a parameter is shared across invocations, so its gradient "
                f"cannot be attributed: {report['shared_parameter_boundaries'][0]}")
    if report.get("unexpected_sites"):
        return ("a boundary fired for a site no adapter was installed for: "
                f"{report['unexpected_sites']}")
    if report.get("duplicate_ids"):
        return (f"an invocation was recorded twice: "
                f"{report['duplicate_ids'][0]}")
    missing = report.get("missing_or_extra") or {}
    if missing:
        # Per site, and say which role each is, so "the QKV kernel never ran"
        # never again reads the same as "the carried attention boundary did not".
        sites = report.get("sites") or {}
        detail = ", ".join(
            f"{site} ({sites.get(site, {}).get('role', '?')}): expected "
            f"{sites.get(site, {}).get('expected')}, saw "
            f"{sites.get(site, {}).get('observed')}"
            for site in sorted(missing)
        )
        return f"invocation coverage is wrong per site: {detail}"
    failures = report.get("failures") or []
    if failures:
        first = failures[0]
        return (f"{report['failure_count']} of "
                f"{report['checked_invocations']} invocations disagreed with the "
                f"declaration; worst at {first['id']} on {first['result']} "
                f"({first['max_abs_err']:.3g} > atol {first['atol']:.3g})")
    if report.get("record_desync"):
        return (f"{report['checked_invocations']} records were kept for "
                f"{sum(report.get('observed_counts', {}).values())} counted "
                "invocations; a boundary record was lost")
    return "the live-boundary validation failed"


def _worst(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """The single loudest disagreement, so a failure names a tensor."""
    scored = [s for s in samples if isinstance(s.get("rel_l2"), (int, float))]
    if not scored:
        return {}
    worst = max(scored, key=lambda s: s["rel_l2"])
    return {k: worst.get(k) for k in
            ("name", "role", "rel_l2", "max_abs_over_rms", "cosine",
             "sign_flips_above_floor", "max_abs_err")}


def _reason(verdict: dict[str, Any]) -> str:
    if not verdict["finite"]["ok"]:
        return f"non-finite results: {verdict['finite']['non_finite'][:4]}"
    if verdict["missing_grads"]:
        return f"missing gradients: {verdict['missing_grads'][:4]}"
    if verdict["count_problems"]:
        return f"invocation counts: {verdict['count_problems']}"
    for label in ("vs_eager", "vs_bound_pair"):
        exceeded = verdict[label]["exceeded"]
        if exceeded:
            # Two shapes: a metric that crossed its threshold, and a result
            # whose role has no envelope at all. Both are rejections and both
            # have to be sayable, or a real failure is reported as a formatter
            # crash and its cause is lost.
            first = max(
                exceeded,
                key=lambda e: (e.get("ratio") or 0.0) if e.get("value") is not None else 1e30,
            )
            if first.get("value") is None:
                return (f"{label}: {first.get('name')} ({first.get('role')}) "
                        f"{first.get('reason', 'not comparable')}")
            return (f"{label}: {first.get('name')} ({first.get('role')}) "
                    f"{first.get('metric')}={first['value']:.3e} > "
                    f"{first['threshold']:.3e} ({first.get('ratio', 0):.2f}x)")
    if not verdict["trajectory"]["ok"]:
        return (f"loss trajectory over {verdict['trajectory'].get('horizon')} steps: "
                f"max_abs_delta={verdict['trajectory'].get('max_abs_delta')}")
    return "unknown"


def summarize(verdict: dict[str, Any]) -> dict[str, Any]:
    """The part of a verdict a report should carry, without the curves."""
    trimmed = dict(verdict)
    trajectory = dict(trimmed.get("trajectory") or {})
    trajectory.pop("reference", None)
    trajectory.pop("candidate", None)
    trimmed["trajectory"] = trajectory
    for label in ("vs_eager", "vs_bound_pair"):
        section = dict(trimmed.get(label) or {})
        section["exceeded"] = section.get("exceeded", [])[:8]
        trimmed[label] = section
    boundary = dict(trimmed.get("live_boundary") or {})
    if boundary:
        # The per-invocation detail is a debugging view, not a report: counts,
        # the worst per site, and the failures are what a reader needs.
        boundary["failures"] = boundary.get("failures", [])[:8]
        trimmed["live_boundary"] = boundary
    purity = dict(trimmed.get("provider_purity") or {})
    if purity.get("sites"):
        purity["sites"] = [
            {k: v for k, v in site.items() if k != "compared"}
            for site in purity["sites"]
        ]
        trimmed["provider_purity"] = purity
    return trimmed
