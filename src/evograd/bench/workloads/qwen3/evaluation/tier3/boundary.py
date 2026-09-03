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
from dataclasses import dataclass, field
from typing import Any

import torch

SCHEMA_VERSION = "evograd-qwen3-t3-boundary/1"

#: How many invocations of each site one canonical step must make.
EXPECTED = {"qkv_norm_rope": 28, "attention": 28, "swiglu_mlp": 28,
            "residual_rmsnorm": 56}


def expected_counts(layers: int) -> dict[str, int]:
    """The same law the site registry declares, restated for a given depth."""
    from .sites import expected_counts as sites_expected

    return sites_expected(layers)

#: The three structurally different residual fusions, reported apart.
RESIDUAL_CATEGORIES = ("post_attention", "mlp_to_next_input", "final_model_norm")


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

    def to_dict(self, *, expected: dict[str, int] | None = None) -> dict[str, Any]:
        expected = expected or EXPECTED
        residual = {
            category: sum(
                1 for r in self.invocations if r.get("category") == category
            )
            for category in RESIDUAL_CATEGORIES
        }
        failures = [
            {"id": r["id"], "site": r["site"], "result": e["name"],
             "max_abs_err": e["max_abs_err"], "atol": e["atol"], "rtol": e["rtol"]}
            for r in self.invocations
            for kind in ("outputs", "gradients")
            for e in r.get(kind, [])
            if not e["ok"]
        ]
        missing = {
            site: expected[site] - self.counts.get(site, 0)
            for site in expected
            if self.counts.get(site, 0) != expected[site]
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "expected_counts": dict(expected),
            "observed_counts": dict(self.counts),
            "coverage_ok": not missing and not self.duplicates,
            "missing_or_extra": missing,
            "duplicate_ids": self.duplicates[:16],
            "residual_categories": residual,
            "checked_invocations": len(self.invocations),
            "worst_per_site": self.worst(),
            "failures": failures[:32],
            "failure_count": len(failures),
            "errors": self.errors[:16],
            "shared_parameter_boundaries": self.shared_parameters[:8],
            "summed_is_the_residual_stream": all(
                r.get("summed_is_live", True) for r in self.invocations
                if r["site"] == "residual_rmsnorm"
            ),
            "ok": bool(
                not missing and not self.duplicates and not failures
                and not self.errors and not self.shared_parameters and len(self.invocations) == sum(expected.values())
            ),
        }


def make_validator(op_lookup, *, workload_case, report: BoundaryReport):
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
                diff = float((actual.detach().float() - want.float()).abs().max())
                record["outputs"].append({
                    "name": name, "max_abs_err": diff, "atol": atol, "rtol": rtol,
                    "ok": bool(torch.allclose(actual.detach().float(), want.float(),
                                              atol=atol, rtol=rtol)),
                    "finite": bool(torch.isfinite(actual.detach()).all()),
                })
            del expected
        except Exception as exc:  # a boundary that cannot be checked is a failure
            report.errors.append(f"{identity}: forward shadow: {type(exc).__name__}: {exc}")

        _arm_gradient_shadow(op, workload, reference, record, inputs, got,
                             boundary, report)
        report.invocations.append(record)

    listener.probes = True  # ask the tap for aliased outputs
    return listener


def _arm_gradient_shadow(op, workload, reference, record, inputs, outputs,
                         boundary, report):
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
                diff = float((actual.float() - want.float()).abs().max())
                record["gradients"].append({
                    "name": grad_name, "max_abs_err": diff, "atol": atol,
                    "rtol": rtol, "aliased_boundary": source in aliased,
                    "ok": bool(torch.allclose(actual.float(), want.float(),
                                              atol=atol, rtol=rtol)),
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


def validate_all_invocations(workload, kernels, *, data_seed: int = 0) -> dict[str, Any]:
    """One canonical step with the shadow on. Untimed, and never near a timer."""
    from evograd.ops import get_op

    report = BoundaryReport()
    registry = workload.site_registry

    def op_lookup(site: str):
        return get_op(registry.require(site).op)

    live_dtype = workload.spec.dtype

    def workload_case(op):
        """The declared workload whose tolerances apply to these live tensors.

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

    from .sites import set_tap

    model, provenance = workload.build_patched(kernels)
    set_tap(model, make_validator(op_lookup, workload_case=workload_case, report=report))
    ids, labels = workload.batch_for(seed=data_seed)
    outputs = model(input_ids=ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    report.finalize()
    set_tap(model, None)
    built = workload.last_build
    layers = workload.spec.arch["num_hidden_layers"]
    expected = {
        site: count
        for site, count in expected_counts(layers).items()
        if site in kernels.patched
    }
    summary = report.to_dict(expected=expected)
    summary["provenance"] = provenance.to_dict()
    summary["site_counters"] = built.observed() if built else {}
    summary["data_seed"] = data_seed
    del model, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
