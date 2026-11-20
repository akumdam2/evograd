"""Model Gate 2: the whole model with the candidate installed, against native eager.

The candidate is installed at its site (every call of the site in the model, or
the calls of the configurations it takes) -- alone, or with the other accepted
Level-2 candidates as the combination -- and the pretrained model runs one
forward and backward on each sample (the first real training batch of each data
order), with no optimizer step. Every output the scope declares (a language
model's logits, or under the fused LM head the head's input and the loss; other
models' primary output and loss) and every parameter gradient ``z`` is compared
with native eager on the same sample, in float64::

    E2(c, r) = ||c - r||_2 / ||r||_2

The trusted control is ``torch.compile`` of the native implementation at the same
sites: a correct implementation that rounds differently. The candidate passes when,
on every sample ``x``, ::

    E2(candidate_x[z], native_x[z]) <= T[z] = gamma * max( max_x E2(compile_x[z], native_x[z]),
                                                           floor[z] )

with ``gamma`` the model's (``integrations.gates.model_gamma``) and the floor
``FLOORS`` by result family. One threshold per result, pooled over the samples.

Why native and not a float32 truth: at this level the question is whether the
installed kernel leaves the model computing what it computed, and a float32 copy
of a whole model does not fit beside its runs on the largest models (OLMoE-1B-7B,
MegaFold). Each call's accuracy against the truth is Gate 1's and operator Gate 2's.
Why ``gamma`` and not 2: the candidate and the compiled native round differently,
and a rounding difference at one site reaches every later tensor as a fresh draw;
the largest of several hundred tensors over four samples is then a few times the
control's on some tensor (against compile alone, P8's frontend seeds need gamma up to
2.7 on E2, Qwen3.5-4B's 5.5).
Why E2 only: the model's job is amplification and integration, which move whole
tensors; one tensor's worst element is a noisy statistic here (the seeds' worst
Emax ratio was 1.3-1.8 times their worst E2 ratio against the same control), and a
few wrong elements are Gate 1's.

A deployment whose compiled native reproduces native bit for bit on every sample
(a vendor GEMM: Qwen3.5-4B's gdn_in_proj) has no trusted alternative at this level:
its numbers are left to Gate 1 and operator Gate 2, and model Gate 2 checks its
coverage only. A control that cannot be built, or fails on every sample, leaves the
report incomplete. Coverage, every sample: each replaced site is called as often as
the scope declares.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import NATIVE, Deployment, Sample
from evograd.verification.common.reports import (
    CaseReport,
    ResultCheck,
    VerificationReport,
    failure,
    missing,
)
from evograd.verification.common.tensors import relative_error
from evograd.verification.gate2.controls import compile_control
from evograd.verification.gate2.runner import GATE, execute, integration_contract, structure_checks

PROTOCOL = "evograd-verification-gate2-model/1"
RULE = "evograd-gate2-model-native/1"
#: The least threshold, before gamma, by result family.
FLOORS = {"output": 1e-5, "gradient": 1e-4}
COMPARISON = ("E2 = ||c-r||_2 / ||r||_2 per result and sample against native eager, float64; "
              "T[z] = gamma * max(max over samples of E2(torch.compile at the replaced sites), "
              "FLOORS[family of z]); a deployment compile reproduces bit for bit: coverage only")
NOT_JUDGED = "compile_bitwise"


def rule_identity(gamma: float) -> dict[str, Any]:
    """The rule, as recorded in reports and stage keys."""
    return {"rule": RULE, "gamma": float(gamma), "floors": dict(FLOORS), "metric": "E2",
            "reference": "native eager", "trusted": "torch.compile at the replaced sites",
            "pooling": "per result, the maximum over the samples", "comparison": COMPARISON}


def e2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """``E2`` of ``candidate`` against ``reference`` in float64 (both finite): 0 when
    identical, infinite against an all-zero reference otherwise."""
    err = relative_error(candidate, reference)
    if err.bitwise:
        return 0.0
    return err.diff_l2 / err.ref_l2 if err.ref_l2 > 0 else math.inf


def verify_model(scope, deployment: Deployment, *, samples: Sequence[Sample], gamma: float,
                 compile_backend: str = "inductor", compile_fullgraph: bool = True
                 ) -> VerificationReport:
    """Judge ``deployment`` on every one of ``samples``."""
    try:
        implementation = scope.reference_identity()
    except Exception as exc:  # recorded, never a reason to skip the check
        implementation = {"unidentifiable": f"{type(exc).__name__}: {exc}"}
    report = VerificationReport(
        gate=GATE, protocol=PROTOCOL,
        subject={"scope": scope.scope, "adapter": dict(scope.identity()),
                 "numerics": dict(scope.numerics()), "deployment": deployment.describe(),
                 "reference": {"kind": "native_eager", "implementation": implementation},
                 "rule": rule_identity(gamma), "samples": [s.to_dict() for s in samples]})
    if scope.scope != "model":
        report.findings.append(missing("unknown_scope", f"{scope.scope!r} is not the model scope"))
        return report
    if deployment.role == NATIVE:
        report.findings.append(missing("not_a_candidate", "the native deployment is the reference"))
    unknown = sorted(set(deployment.patch_set) - set(scope.sites()))
    if unknown:
        report.findings.append(failure("unknown_sites", f"sites {unknown} are not replaceable here"))
        return report
    try:
        control, graphs = compile_control(scope, deployment.patch_set, backend=compile_backend,
                                          fullgraph=compile_fullgraph)
        report.evidence["compile"] = {"deployment": control.describe(), "graphs": graphs}
    except Exception as exc:
        control = None
        report.evidence["compile"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    trusted: dict[str, float] = {}        # result -> compile's largest E2 over the samples
    kinds: dict[str, str] = {}
    judged: list[ResultCheck] = []
    compiled = bitwise = 0                # samples compile ran on / reproduced native on
    for sample in samples:
        case = CaseReport(case_id=sample.sample_id, evidence={"sample": sample.to_dict()})
        report.cases.append(case)
        native = execute(scope, Deployment.native(), sample)
        if native.error:
            case.findings.append(missing("reference_runtime_error",
                                         f"native failed at {native.error_stage}", trace=native.error))
            continue
        contract = integration_contract(native)
        names = [s.name for s in contract.compared()
                 if not s.exact and C.is_float_tensor(native.results.get(s.name))]
        invalid = [n for n in names if not bool(torch.isfinite(native.results[n]).all())]
        if invalid:
            case.findings.append(missing("invalid_reference", f"non-finite native results {invalid[:8]}"))
            continue
        kinds.update({s.name: s.kind for s in contract.specs if s.name in names})
        if control is not None:
            ctl = execute(scope, control, sample)
            if ctl.error or ctl.counts != ctl.expected:
                case.evidence["compile"] = {"failed_at": ctl.error_stage, "trace": ctl.error,
                                            "site_calls": ctl.counts, "expected": ctl.expected}
            else:
                compiled += 1
                same = True
                for name in names:
                    value = ctl.results.get(name)
                    if not (torch.is_tensor(value) and value.shape == native.results[name].shape
                            and bool(torch.isfinite(value).all())):
                        same = False
                        continue
                    err = e2(value, native.results[name])
                    same = same and err == 0.0
                    if math.isfinite(err):
                        trusted[name] = max(trusted.get(name, 0.0), err)
                bitwise += same
            del ctl
        cand = execute(scope, deployment, sample)
        case.evidence["candidate"] = {"seconds": cand.seconds, "site_calls": cand.counts,
                                      "expected_site_calls": cand.expected}
        if cand.error:
            code = "output_structure" if cand.error_stage == "outputs" else "candidate_runtime_error"
            case.findings.append(failure(code, f"deployment failed at {cand.error_stage}",
                                         trace=cand.error))
            continue
        if cand.counts != cand.expected:
            case.findings.append(failure("coverage_mismatch",
                                         f"site calls {cand.counts} differ from the declared "
                                         f"{cand.expected}", observed=cand.counts,
                                         expected=cand.expected))
        checks, numeric, loose = structure_checks(contract, cand.results, native.results)
        case.findings.extend(loose)
        for name in numeric:
            check = checks[name]
            check.compared = True
            check.metrics = {"E2": e2(cand.results[name], native.results[name])}
            judged.append(check)
        case.results.extend(checks.values())
        del cand, native
    report.evidence.update(compiled_samples=compiled, bitwise_samples=bitwise)
    if compiled == 0:
        report.findings.append(missing("no_trusted_control", "torch.compile of the native sites "
                                       "did not run on any sample: no threshold"))
        return report
    if bitwise == compiled == len(samples):
        report.evidence["numeric"] = (f"not judged ({NOT_JUDGED}): torch.compile reproduces native "
                                      "bit for bit at the replaced sites on every sample, so there "
                                      "is no trusted alternative; coverage only")
        for check in judged:
            check.limits = {"E2": None, "not_judged": NOT_JUDGED}
        return report
    thresholds: dict[str, dict[str, Any]] = {}
    for name, kind in sorted(kinds.items()):
        floor = FLOORS[C.FLOOR_FAMILY[kind]]
        own = trusted.get(name, 0.0)
        thresholds[name] = {"threshold": gamma * max(own, floor), "compile": own, "floor": floor,
                            "from": "compile" if own > floor else "floor"}
    report.subject["thresholds"] = thresholds
    for check in judged:
        entry = thresholds[check.name]
        check.limits = {"E2": entry["threshold"]}
        value, bound = check.metrics["E2"], entry["threshold"]
        if not value <= bound:
            basis = (f"{gamma:g} x compile's {entry['compile']:.3g}" if entry["from"] == "compile"
                     else f"{gamma:g} x the floor {entry['floor']:g}")
            check.findings.append(failure("threshold_exceeded",
                                          f"E2 {value:.3g} > {bound:.3g} ({value / bound:.3g}x; {basis})",
                                          subject=check.name, metric="E2", value=value,
                                          threshold=bound))
    return report
