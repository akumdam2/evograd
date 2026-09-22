"""Every one of the 140 invocations, checked against its own contract.

The earlier boundary check compared one representative layer. That is enough to
show the wiring is right and not enough to show the *provider* is: a kernel that
is correct in layer 14 and wrong in layer 3 passes it, and so does one that is
correct for its first eight calls.

This validates all 140 invocations of one canonical step -- 28 `qkv_norm_rope`,
28 `attention`, 28 `swiglu_mlp`, 56 `residual_rmsnorm` -- each with a stable
identity, each against the declaration's own ``runtime_forward`` on the same
live inputs, and each against the same live upstream gradient the model actually
delivered.

**Shadow, not substitute.** The reference is recomputed during the backward,
once the real upstream gradient is known, and it:

* runs outside every timed and peak-memory region -- this mode is never on when
  anything is measured;
* mutates no model input and no provider state; every captured tensor is
  detached and the reference runs on its own leaves;
* saves and restores CPU and CUDA RNG around each reference call, so a reference
  that consumed randomness could not shift the model's stream;
* releases each invocation's tensors as soon as its comparison is done, so the
  peak is one boundary rather than a hundred and forty.

Only summaries are serialized: identities, counts, worst error per site and per
layer, thresholds. No tensor reaches disk.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch

SCHEMA_VERSION = "evograd-qwen3-t3-boundary/3"

#: How many invocations of each site one canonical step must make.
EXPECTED = {"qkv_norm_rope": 28, "attention": 28, "swiglu_mlp": 28,
            "residual_rmsnorm": 56}


def expected_counts(layers: int) -> dict[str, int]:
    """The same law the site registry declares, restated for a given depth."""
    from .sites import expected_counts as sites_expected

    return sites_expected(layers)


@dataclass(frozen=True)
class SitePlan:
    """Which sites this run patched, which it carries, and how often each runs.

    The distinction is load-bearing. Patching ``qkv_norm_rope`` installs the
    composite Qwen3Attention adapter, which then runs the ``attention`` boundary
    too -- through its production spelling, because no candidate was asked for
    there. Both are real boundaries worth validating, and both must meet their
    own declared count; what neither may be measured against is a total built
    from the patched sites alone. That comparison is what a single aggregate
    identity silently performed, failing a QKV-only patch on arithmetic rather
    than on anything a kernel did.
    """

    patched: tuple[str, ...]
    supporting: tuple[str, ...]
    expected: dict[str, int]

    @classmethod
    def build(cls, requested, *, layers: int) -> SitePlan:
        """Derive the plan from the adapter grouping and the declared counts."""
        from .sites import live_sites, supporting_sites

        patched = tuple(sorted(requested))
        supporting = supporting_sites(patched)
        declared = expected_counts(layers)
        live = live_sites(patched)
        missing_law = [site for site in live if site not in declared]
        if missing_law:
            raise ValueError(
                f"sites {missing_law} become live boundaries but declare no "
                "expected count; the registry and the adapter grouping disagree"
            )
        return cls(patched=patched, supporting=supporting,
                   expected={site: declared[site] for site in live})

    def role(self, site: str) -> str:
        if site in self.patched:
            return "patched"
        if site in self.supporting:
            return "supporting"
        return "unexpected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "patched": list(self.patched),
            "supporting": list(self.supporting),
            "expected_by_site": dict(self.expected),
            "roles": {site: self.role(site) for site in self.expected},
        }

#: The three structurally different residual fusions, reported apart.
RESIDUAL_CATEGORIES = ("post_attention", "mlp_to_next_input", "final_model_norm")

# ── the reference-calibrated local envelope ──────────────────────────────────
#
# The declared per-result ``allclose(atol, rtol)`` was calibrated on the
# synthetic canonical workload, where |q| <= ~4 and the MLP output is O(1). On
# the pretrained checkpoint the same boundaries carry |k| ~ 500 and MLP outputs
# ~ 5e3, and one bfloat16 ULP there (2 and 32) is far above the declared atol
# wherever an output element is small because two large operands cancelled
# (RoPE's q*cos + rotate_half(q)*sin; the down projection's 3072-term sum). The
# probe of 2026-09-11 showed the trusted torch.compile reference failing the
# declared clause at 18/28 (q) and 13/28 (MLP out) invocations while being
# *closer* to a float32 reference than the eager spelling itself.
#
# So a second clause, derived the same way B and C are: from references only,
# with the same margin, frozen before any candidate is judged, and applied on
# holdout seeds. A result passes part A if it passes the declared clause OR it
# is within the envelope on both its worst element and its whole-tensor
# relative L2. The declared clause is untouched; the envelope admits only what
# a trusted correct implementation itself needs.
ENVELOPE_SCHEMA = "evograd-qwen3-t3-local-envelope/1"
ENVELOPE_MARGIN = 2.0
#: Below this whole-tensor relative L2 two bfloat16 tensors cannot be told apart
#: on purpose (unit roundoff 2^-8 = 3.9e-3 per element).
ENVELOPE_REL_FLOOR = 1e-4


def _rel_l2(actual: torch.Tensor, want: torch.Tensor) -> float:
    a, b = actual.detach().to(torch.float64), want.detach().to(torch.float64)
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def envelope_bounds(envelope, site: str, name: str):
    """``(max_abs, rel_l2)`` thresholds for one result, or ``None``."""
    if not envelope:
        return None
    entry = ((envelope.get("sites") or {}).get(site) or {}).get(name)
    if not entry:
        return None
    return float(entry["max_abs"]), float(entry["rel_l2"])


def derive_local_envelope(per_result_worst_reports, *, margin: float = ENVELOPE_MARGIN,
                          rel_floor: float = ENVELOPE_REL_FLOOR, sources=None) -> dict[str, Any]:
    """T = margin * max(worst trusted-reference disagreement over the calibration
    seeds, floor) per (site, result); the floor is the declared atol for the
    worst element and ``rel_floor`` for the relative L2. Candidate-free by
    construction: the inputs are boundary reports of the trusted compile
    provider, never of a program under evaluation."""
    reports = list(per_result_worst_reports)
    if not reports:
        raise ValueError("an envelope needs at least one reference boundary report")
    sites: dict[str, dict[str, Any]] = {}
    for report in reports:
        for site, results in report.items():
            for name, worst in results.items():
                entry = sites.setdefault(site, {}).setdefault(name, {
                    "reference_max_abs": 0.0, "reference_rel_l2": 0.0,
                    "atol": float(worst.get("atol", 0.0)), "rtol": float(worst.get("rtol", 0.0)),
                    "samples": 0})
                entry["reference_max_abs"] = max(entry["reference_max_abs"], float(worst["max_abs_err"]))
                entry["reference_rel_l2"] = max(entry["reference_rel_l2"], float(worst.get("rel_l2", 0.0)))
                entry["atol"] = max(entry["atol"], float(worst.get("atol", 0.0)))
                entry["samples"] += 1
    for site, results in sites.items():
        for name, entry in results.items():
            base_abs = max(entry["reference_max_abs"], entry["atol"])
            base_rel = max(entry["reference_rel_l2"], rel_floor)
            entry.update({
                "max_abs": margin * base_abs, "rel_l2": margin * base_rel,
                "binding_abs": "reference" if base_abs == entry["reference_max_abs"] else "declared_atol",
                "binding_rel": "reference" if base_rel == entry["reference_rel_l2"] else "floor",
            })
    return {"schema": ENVELOPE_SCHEMA, "margin": margin, "rel_floor": rel_floor,
            "rule": ("a result passes part A if allclose(declared atol, rtol) OR "
                     "(max_abs_err <= max_abs AND rel_l2 <= rel_l2)"),
            "derived_from": list(sources or []), "calibration_reports": len(reports),
            "sites": sites}


@contextlib.contextmanager
def _rng_preserved():
    """A reference call must not move the model's random stream."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def invocation_id(site: str, key, ordinal: int) -> str:
    """A stable name for one invocation: site, layer, category, ordinal.

    Stable across runs and orderings, so a failure names the same thing twice
    and two reports can be diffed.
    """
    if isinstance(key, (tuple, list)):
        layer, category = key
    else:
        layer, category = key, None
    layer_part = "?" if layer is None else str(layer)
    parts = [site, f"layer{layer_part}"]
    if category:
        parts.append(category)
    parts.append(f"#{ordinal}")
    return ":".join(parts)


@dataclass
class BoundaryReport:
    """What the shadow saw, without what it saw it on."""

    invocations: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)
    duplicates: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pending: list[Any] = field(default_factory=list)
    param_owner: dict[int, str] = field(default_factory=dict)
    shared_parameters: list[str] = field(default_factory=list)

    def finalize(self) -> None:
        """Settle invocations whose outputs the model never consumed.

        ``model.norm``'s ``summed`` is real -- the fusion computes it -- but
        nothing downstream reads it, so no gradient ever arrives and the hook
        that would complete that invocation never fires. Zero is the correct
        upstream for an output off the loss path, and it is only knowable once
        the backward pass is over.
        """
        for settle in self.pending:
            settle(final=True)
        self.pending.clear()

    def worst(self) -> dict[str, Any]:
        by_site: dict[str, dict[str, Any]] = {}
        for record in self.invocations:
            site = record["site"]
            for kind in ("outputs", "gradients"):
                for entry in record.get(kind, []):
                    current = by_site.get(site)
                    if current is None or entry["max_abs_err"] > current["max_abs_err"]:
                        by_site[site] = {
                            "id": record["id"], "layer": record["layer"],
                            "category": record.get("category"),
                            "which": kind, "result": entry["name"],
                            "max_abs_err": entry["max_abs_err"],
                            "atol": entry["atol"], "rtol": entry["rtol"],
                            "ok": entry["ok"],
                        }
        return by_site

    def per_result_worst(self) -> dict[str, dict[str, Any]]:
        """Per site and per result: the loudest element and the largest relative L2."""
        out: dict[str, dict[str, Any]] = {}
        for record in self.invocations:
            site = record["site"]
            for kind in ("outputs", "gradients"):
                for entry in record.get(kind, []):
                    cur = out.setdefault(site, {}).get(entry["name"])
                    if cur is None:
                        cur = out[site][entry["name"]] = {
                            "max_abs_err": 0.0, "rel_l2": 0.0, "atol": entry["atol"],
                            "rtol": entry["rtol"], "kind": kind, "worst_id": None,
                            "worst_rel_id": None}
                    if entry["max_abs_err"] >= cur["max_abs_err"]:
                        cur["max_abs_err"], cur["worst_id"] = entry["max_abs_err"], record["id"]
                    if entry.get("rel_l2", 0.0) >= cur["rel_l2"]:
                        cur["rel_l2"], cur["worst_rel_id"] = entry.get("rel_l2", 0.0), record["id"]
        return out

    def to_dict(self, *, plan: SitePlan | None = None,
                expected: dict[str, int] | None = None,
                envelope: dict[str, Any] | None = None) -> dict[str, Any]:
        if plan is None:
            declared = expected or EXPECTED
            plan = SitePlan(patched=tuple(sorted(declared)), supporting=(),
                            expected=dict(declared))
        expected = dict(plan.expected)
        residual = {
            category: sum(
                1 for r in self.invocations if r.get("category") == category
            )
            for category in RESIDUAL_CATEGORIES
        }
        # Every failed invocation is kept: a truncated machine-readable list
        # cannot be acted on, and the count here is bounded by invocations x
        # results. `summarize()` is what trims this for a console report.
        failures = [
            {"id": r["id"], "site": r["site"], "layer": r.get("layer"),
             "category": r.get("category"), "ordinal": r.get("ordinal"),
             "result": e["name"], "role": ("gradient" if kind == "gradients" else "output"),
             "max_abs_err": e["max_abs_err"], "atol": e["atol"], "rtol": e["rtol"],
             "rel_l2": e.get("rel_l2"), "envelope": e.get("envelope"),
             "finite": e.get("finite"), "discrepancy": e.get("discrepancy")}
            for r in self.invocations
            for kind in ("outputs", "gradients")
            for e in r.get(kind, [])
            if not e["ok"]
        ]
        non_finite = [f["id"] for f in failures if f.get("finite") is False]
        # Accounting is per site, and every site the plan names is held to its
        # own declared count -- carried ones included. There is deliberately no
        # aggregate total: an expected sum built from the patched sites alone
        # cannot describe a run whose adapter also drives a carried boundary.
        observed = dict(self.counts)
        missing = {
            site: expected[site] - observed.get(site, 0)
            for site in expected
            if observed.get(site, 0) != expected[site]
        }
        unexpected = {
            site: count for site, count in observed.items() if site not in expected
        }
        by_role: dict[str, dict[str, Any]] = {}
        for site in sorted(set(expected) | set(observed)):
            by_role[site] = {
                "role": plan.role(site),
                "expected": expected.get(site),
                "observed": observed.get(site, 0),
                "shortfall": missing.get(site),
            }
        # `counts` and `invocations` are written by the same listener, one skip
        # apart on a duplicate. A disagreement means a record was lost rather
        # than a boundary missed, which is a different defect and worth naming.
        record_desync = len(self.invocations) != sum(observed.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "site_plan": plan.to_dict(),
            "sites": by_role,
            "expected_counts": dict(expected),
            "observed_counts": observed,
            "patched_sites": list(plan.patched),
            "supporting_sites": list(plan.supporting),
            "coverage_ok": not missing and not unexpected and not self.duplicates,
            "missing_or_extra": missing,
            "unexpected_sites": unexpected,
            "duplicate_ids": self.duplicates[:16],
            "record_desync": record_desync,
            "residual_categories": residual,
            "checked_invocations": len(self.invocations),
            "worst_per_site": self.worst(),
            "per_result_worst": self.per_result_worst(),
            "local_check_mode": ("declared_or_reference_envelope" if envelope
                                 else "declared_tolerance_only"),
            "local_check_rule": ENVELOPE_RULE if envelope else "allclose(declared atol, rtol)",
            "envelope_ulp_floor_used": sum(
                1 for r in self.invocations for kind in ("outputs", "gradients")
                for e in r.get(kind, [])
                if e.get("ok") and not e.get("declared_ok", True)
                and (e.get("envelope") or {}).get("abs_binding") == "ulp_floor"
                and e["max_abs_err"] > (e.get("envelope") or {}).get("max_abs", float("inf"))),
            "envelope_admitted": sum(
                1 for r in self.invocations for kind in ("outputs", "gradients")
                for e in r.get(kind, []) if e.get("ok") and not e.get("declared_ok", True)),
            "failures": failures,
            "failure_count": len(failures),
            "non_finite_results": non_finite,
            # Is everything that went wrong here a finite tolerance
            # disagreement? Coverage, duplicates, shared parameters, lost
            # records, shadow errors and non-finite values are not, and a
            # report-first run must not treat them as one.
            "numerical_only": bool(
                failures and not non_finite and not missing and not unexpected
                and not self.duplicates and not self.errors
                and not self.shared_parameters and not record_desync),
            "errors": self.errors[:16],
            "shared_parameter_boundaries": self.shared_parameters[:8],
            "summed_is_the_residual_stream": all(
                r.get("summed_is_live", True) for r in self.invocations
                if r["site"] == "residual_rmsnorm"
            ),
            "ok": bool(
                not missing and not unexpected and not self.duplicates
                and not failures and not self.errors
                and not self.shared_parameters and not record_desync
            ),
        }


#: The rule the envelope clause applies. Version 2 adds the dtype's own
#: resolution floor to the worst-element bound: one unit in the last place of
#: the tensor's largest magnitude. A worst-element envelope calibrated as the
#: maximum over three seeds is an extreme-value statistic, and on 2026-09-11 the
#: trusted compile reference itself exceeded the four-site calibration on a
#: holdout seed (MLP output, |out| ~ 4e3, disagreement 16 = half a ULP at that
#: magnitude) -- an erroneous rejection of a positive control. A single element
#: differing by no more than one ULP at the tensor's own scale cannot be told
#: from rounding by any comparison in that dtype; anything a kernel gets wrong
#: on purpose is either larger than that at some element or visible in the
#: whole-tensor relative L2, which stays calibrated and unchanged.
ENVELOPE_RULE = ("v2: allclose(declared) OR (max_abs_err <= max(T_abs, ulp(max|reference|)) "
                 "AND rel_l2 <= T_rel)")


def ulp_at_max(want: torch.Tensor) -> float:
    """One unit in the last place of ``want``'s dtype at its largest magnitude."""
    peak = float(want.detach().abs().max())
    if peak == 0.0 or not torch.is_floating_point(want):
        return 0.0
    eps = torch.finfo(want.dtype).eps  # 2^-7 for bfloat16, 2^-10 for float16, 2^-23 for float32
    return eps * 2.0 ** math.floor(math.log2(peak))


def _judge(actual, want, atol, rtol, bounds, *, name=None, role=None, reference=None):
    """Apply the declared tolerance or calibrated envelope; detail every failure."""
    diff = float((actual.float() - want.float()).abs().max())
    declared_ok = bool(torch.allclose(actual.float(), want.float(), atol=atol, rtol=rtol))
    rel = _rel_l2(actual, want)
    entry = {"max_abs_err": diff, "rel_l2": rel, "atol": atol, "rtol": rtol,
             "declared_ok": declared_ok, "ok": declared_ok}
    if bounds is not None:
        max_abs, rel_l2 = bounds
        ulp = ulp_at_max(want)
        abs_bound = max(max_abs, ulp)
        env_ok = bool(diff <= abs_bound and rel <= rel_l2)
        entry["envelope"] = {"max_abs": max_abs, "ulp_at_max_ref": ulp, "abs_bound": abs_bound,
                             "rel_l2": rel_l2, "ok": env_ok,
                             "abs_binding": "calibrated" if abs_bound == max_abs else "ulp_floor"}
        entry["ok"] = declared_ok or env_ok
    if not entry["ok"]:
        # A failing comparison has to say which elements disagree and by how
        # much; one streaming pass, no tensor retained (gate.discrepancy).
        from evograd.evaluation.tier3.gate.discrepancy import elementwise_record

        entry["discrepancy"] = elementwise_record(
            actual, want, atol=atol, rtol=rtol, name=name, role=role, reference=reference,
            rule=(ENVELOPE_RULE if bounds is not None else "allclose(declared atol, rtol)"),
            envelope=(entry.get("envelope") if bounds is not None else None))
    return entry


def make_validator(op_lookup, *, workload_case, report: BoundaryReport, envelope=None):
    """A tap that shadow-checks every invocation it is handed.

    Gradients are the interesting half, and getting them right needs a real
    boundary. The tap supplies one: each output is aliased before it leaves the
    operator, so the gradient arriving at the alias is the model's own upstream
    and nothing of the operator's internals. The gradient the operator *emits*
    is read from a hook on each differentiable input. When both are in hand the
    reference is differentiated with the identical upstream and compared against
    what the provider actually delivered.
    """
    from evograd.opdecl.inputs import as_output_tuple
    from evograd.opdecl.oracle import resolve_runtime_forward

    ordinal = {"n": 0}

    def listener(site, key, inputs, outputs, boundary=None):
        ordinal["n"] += 1
        identity = invocation_id(site, key, ordinal["n"])
        if identity in report.ids:
            report.duplicates.append(identity)
            return
        report.ids.add(identity)
        report.counts[site] = report.counts.get(site, 0) + 1

        op = op_lookup(site)
        workload = workload_case(op)
        reference = resolve_runtime_forward(op)
        layer, category = (key if isinstance(key, (tuple, list)) else (key, None))
        got = outputs if isinstance(outputs, tuple) else (outputs,)

        # Detached copies: the shadow must not join the model's graph, and it
        # must not hold the live tensors alive past its own comparison.
        args = [
            inputs.get(arg.name, getattr(arg, "default", None)) for arg in op.args
        ]
        detached = [a.detach() if torch.is_tensor(a) else a for a in args]

        record: dict[str, Any] = {
            "id": identity, "site": site, "layer": layer, "category": category,
            "ordinal": ordinal["n"], "op": op.name, "outputs": [], "gradients": [],
        }
        if site == "residual_rmsnorm":
            # `summed` must be the tensor the model carries forward, not a
            # recomputation: it is the residual stream, and a copy would mean
            # the fusion is decorative.
            record["summed_is_live"] = bool(
                len(got) > 1 and got[1].requires_grad and got[1].grad_fn is not None
            )

        try:
            with torch.no_grad(), _rng_preserved():
                expected = as_output_tuple(op, reference(*detached))
            for name, actual, want in zip(op.output_names, got, expected):
                atol, rtol = op.tolerance_for(workload, name)
                record["outputs"].append({
                    "name": name,
                    **_judge(actual.detach(), want, atol, rtol,
                             envelope_bounds(envelope, site, name),
                             name=name, role="forward_output",
                             reference=f"{op.name}: declared runtime spelling "
                                       f"(resolve_runtime_forward) on this invocation's live inputs"),
                    "finite": bool(torch.isfinite(actual.detach()).all()),
                })
            del expected
        except Exception as exc:  # a boundary that cannot be checked is a failure
            report.errors.append(f"{identity}: forward shadow: {type(exc).__name__}: {exc}")

        _arm_gradient_shadow(op, workload, reference, record, inputs, got,
                             boundary, report, envelope=envelope)
        report.invocations.append(record)

    listener.probes = True  # ask the tap for aliased outputs
    return listener


def _arm_gradient_shadow(op, workload, reference, record, inputs, outputs,
                         boundary, report, envelope=None):
    """Compare the gradient the provider emitted with the reference's.

    "Emitted" is the operative word. For activations the tap supplies aliases,
    so what arrives is this operator's contribution alone. Parameters are read
    off their module by the production spelling and cannot be aliased, so their
    gradient is taken from a hook on the parameter itself -- correct exactly
    when the parameter belongs to one invocation, which is recorded and checked
    rather than assumed.
    """
    from evograd.opdecl.inputs import as_output_tuple

    if boundary is None:
        return
    upstream_sink = boundary["upstream"]
    produced: dict[str, torch.Tensor] = dict(boundary["emitted"])
    aliased = set(boundary["emitted"])
    active = [a for a in op.active_args() if torch.is_tensor(inputs.get(a.name))]
    wanted = {a.name for a in active if inputs[a.name].requires_grad}
    # Outputs off the loss path carry no upstream; zero is the honest value.
    dead = {
        index: torch.zeros_like(tensor)
        for index, tensor in enumerate(outputs)
        if not (torch.is_tensor(tensor) and tensor.requires_grad)
    }
    if not wanted:
        return
    done = {"n": False}

    def settle(final: bool = False):
        if done["n"]:
            return
        upstream = {**dead, **upstream_sink}
        if not final and (len(upstream) != len(op.output_names)
                          or len(produced) != len(wanted)):
            return
        for index, tensor in enumerate(outputs):  # never consumed: zero upstream
            upstream.setdefault(index, torch.zeros_like(tensor))
        done["n"] = True
        try:
            with _rng_preserved(), torch.enable_grad():
                leaves: dict[str, torch.Tensor] = {}
                args = []
                for arg in op.args:
                    value = inputs.get(arg.name, getattr(arg, "default", None))
                    if torch.is_tensor(value) and arg.name in produced:
                        value = value.detach().clone().requires_grad_(True)
                        leaves[arg.name] = value
                    elif torch.is_tensor(value):
                        value = value.detach()
                    args.append(value)
                if not leaves:
                    return
                shadow = as_output_tuple(op, reference(*args))
                grads = torch.autograd.grad(
                    shadow, list(leaves.values()),
                    tuple(upstream[i] for i in range(len(op.output_names))),
                    allow_unused=True,
                )
            order = list(leaves)
            by_grad = {a.grad_name: a.name for a in op.active_args()}
            for grad_name in op.grad_names():
                source = by_grad[grad_name]
                if source not in leaves:
                    continue
                want = grads[order.index(source)]
                actual = produced[source]
                if want is None:
                    want = torch.zeros_like(actual)
                atol, rtol = op.tolerance_for(workload, grad_name)
                record["gradients"].append({
                    "name": grad_name,
                    **_judge(actual, want, atol, rtol,
                             envelope_bounds(envelope, record["site"], grad_name),
                             name=grad_name, role="backward_gradient",
                             reference=f"{op.name}: declared runtime spelling differentiated with "
                                       f"the model's own upstream cotangent"),
                    "aliased_boundary": source in aliased,
                    "finite": bool(torch.isfinite(actual).all()),
                })
            missing = sorted(wanted - set(produced))
            if missing:
                report.errors.append(
                    f"{record['id']}: no gradient reached {', '.join(missing)}"
                )
            del shadow, grads, leaves, args
        except Exception as exc:
            report.errors.append(
                f"{record['id']}: gradient shadow: {type(exc).__name__}: {exc}"
            )
        upstream_sink.clear()
        produced.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for arg in active:
        if arg.name in wanted and arg.name not in aliased:
            tensor = inputs[arg.name]
            # A parameter read straight off its module: sound only if this is
            # its one and only consuming invocation.
            owner = report.param_owner.setdefault(id(tensor), record["id"])
            if owner != record["id"]:
                report.shared_parameters.append(
                    f"{record['id']}:{arg.name} also used by {owner}"
                )
            tensor.register_hook(
                lambda g, _n=arg.name: (produced.__setitem__(_n, g.detach()),
                                        settle(), g)[-1]
            )
    report.pending.append(settle)


def declared_case_for(op, live_dtype: str):
    """The declared workload whose tolerances apply to live tensors of ``live_dtype``.

    The observed configuration is the canonical answer and the only one a
    real run meets. A reduced test model runs a different dtype and width,
    and quoting the observed bfloat16 tolerance at it would be applying a
    threshold measured somewhere else -- so a same-dtype declared case is
    preferred when the live run is not the canonical one.
    """
    observed = op.benchmark_workloads(suite="qwen3_0_6b_observed")
    if observed and observed[0].dtype == live_dtype:
        return observed[0]
    matching = [w for w in op.correctness if w.dtype == live_dtype]
    if matching:
        return max(matching, key=lambda w: sum(w.dims.values()))
    return observed[0] if observed else op.benchmark[0]


def validate_all_invocations(workload, kernels, *, data_seed: int = 0,
                             envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    """One canonical step with the shadow on. Untimed, and never near a timer.

    ``envelope`` is a frozen reference-calibrated local envelope (see
    :func:`derive_local_envelope`); absent, only the declared clause judges.
    """
    if envelope and envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError(f"local envelope schema {envelope.get('schema')!r} is not {ENVELOPE_SCHEMA!r}")
    from evograd.benchmark import get_task

    report = BoundaryReport()
    registry = workload.site_registry

    def op_lookup(site: str):
        return get_task(registry.require(site).op)

    live_dtype = workload.spec.dtype

    def workload_case(op):
        return declared_case_for(op, live_dtype)

    from .sites import set_tap

    model, provenance = workload.build_patched(kernels)
    set_tap(model, make_validator(op_lookup, workload_case=workload_case, report=report,
                                  envelope=envelope))
    ids, labels = workload.batch_for(seed=data_seed)
    outputs = model(input_ids=ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    report.finalize()
    set_tap(model, None)
    built = workload.last_build
    layers = workload.spec.arch["num_hidden_layers"]
    plan = SitePlan.build(kernels.patched, layers=layers)
    summary = report.to_dict(plan=plan, envelope=envelope)
    summary["provenance"] = provenance.to_dict()
    summary["site_counters"] = built.observed() if built else {}
    summary["data_seed"] = data_seed
    del model, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
