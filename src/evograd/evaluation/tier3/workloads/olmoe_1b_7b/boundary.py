"""Every site invocation in one model call, checked against its own contract.

The block gate compares what the whole block returned. This checks each site on
the live tensors: the adapters hand every invocation's inputs, outputs and
boundary gradients to a :class:`BoundaryValidator` (``sites._run_site``), and
after the backward each invocation is recomputed through its declaration's
``runtime_forward`` on the same inputs and differentiated with the same
upstream gradient. Compared per invocation: every output, and the gradient the
site emitted into each activation input (read on an identity view of that input,
so it is this site's contribution alone). Parameter gradients are read off
their modules, not passed through the boundary.

``layers`` restricts the records to some layers, so the held tensors stay
bounded on a 16-layer model. Tolerances are the
declaration's (calibrated) at the live dtype.
Untimed, and only on when a listener is attached.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

SCHEMA_VERSION = "evograd-olmoe-t3-boundary/1"


def _detached(value):
    return value.detach() if torch.is_tensor(value) else value


class BoundaryValidator:
    """Collects live invocations during a call; judges them afterwards."""

    def __init__(self, op_lookup: Callable[[str], Any], *, dtype: str,
                 layers: Iterable[int] | None = None):
        self.op_lookup = op_lookup
        self.dtype = dtype
        self.layers = None if layers is None else frozenset(int(i) for i in layers)
        self.records: list[dict[str, Any]] = []
        self.skipped: dict[str, int] = {}

    def __call__(self, site, inputs, outputs, upstream, emitted, layer=None) -> None:
        if self.layers is not None and layer not in self.layers:
            self.skipped[site] = self.skipped.get(site, 0) + 1
            return
        self.records.append({
            "site": site,
            "layer": layer,
            "ordinal": sum(1 for r in self.records if r["site"] == site),
            "inputs": {k: _detached(v) for k, v in inputs.items()},
            "outputs": tuple(_detached(t) for t in outputs),
            # Filled by the gradient hooks during the backward.
            "upstream": upstream,
            "emitted": emitted,
        })

    def _tolerance(self, op, name):
        from evograd.opdecl import Workload

        workload = Workload(dims=dict(op.benchmark[0].dims) if op.benchmark else {},
                            dtype=self.dtype)
        return op.tolerance_for(workload, name)

    def _judge(self, op, record) -> dict[str, Any]:
        from evograd.opdecl.inputs import as_output_tuple
        from evograd.opdecl.oracle import resolve_runtime_forward

        forward = resolve_runtime_forward(op)
        active = {a.name for a in op.active_args()}
        leaves: dict[str, torch.Tensor] = {}
        args = []
        for arg in op.args:
            if arg.name in record["inputs"] and record["inputs"][arg.name] is not None:
                value = record["inputs"][arg.name]
            elif getattr(arg, "default", None) is not None:
                value = arg.default
            else:
                raise KeyError(f"{record['site']}: the live call did not supply {arg.name!r}")
            if torch.is_tensor(value) and arg.name in record["emitted"] and arg.name in active:
                value = value.clone().requires_grad_(True)
                leaves[arg.name] = value
            args.append(value)
        with torch.enable_grad():
            reference = as_output_tuple(op, forward(*args))
        checks: list[dict[str, Any]] = []
        for out, live, want in zip(op.outputs, record["outputs"], reference):
            atol, rtol = self._tolerance(op, out.name)
            err = float((live.float() - want.detach().float()).abs().max()) if live.numel() else 0.0
            ok = (live.shape == want.shape and live.dtype == want.dtype
                  and bool(torch.allclose(live.float(), want.detach().float(), atol=atol,
                                          rtol=rtol)))
            checks.append({"result": out.name, "kind": "output", "ok": ok, "max_abs_err": err,
                           "atol": atol, "rtol": rtol})
        upstream = record["upstream"]
        if leaves and len(upstream) == len(reference):
            grads = torch.autograd.grad(
                [reference[i] for i in range(len(reference))], list(leaves.values()),
                [upstream[i] for i in range(len(reference))], allow_unused=True)
            by_arg = {a.name: a.grad_name for a in op.active_args()}
            for (name, _leaf), want in zip(leaves.items(), grads):
                live = record["emitted"].get(name)
                if live is None or want is None:
                    checks.append({"result": by_arg[name], "kind": "gradient", "ok": False,
                                   "reason": "gradient missing on one side"})
                    continue
                atol, rtol = self._tolerance(op, by_arg[name])
                err = float((live.float() - want.float()).abs().max())
                ok = bool(torch.allclose(live.float(), want.float(), atol=atol, rtol=rtol))
                checks.append({"result": by_arg[name], "kind": "gradient", "ok": ok,
                               "max_abs_err": err, "atol": atol, "rtol": rtol})
        elif leaves:
            checks.append({"result": "upstream", "kind": "gradient", "ok": False,
                           "reason": f"{len(upstream)} upstream gradients for "
                                     f"{len(reference)} outputs"})
        return {"site": record["site"], "layer": record["layer"], "ordinal": record["ordinal"],
                "op": op.name, "ok": all(c["ok"] for c in checks), "checks": checks}

    def finalize(self, *, expected: dict[str, int]) -> dict[str, Any]:
        """Judge every record (releasing each as it goes) and compare the counts.

        ``expected`` counts the *checked* invocations: with a ``layers`` filter,
        the expected counts of those layers only.
        """
        invocations = []
        while self.records:
            record = self.records.pop(0)
            invocations.append(self._judge(self.op_lookup(record["site"]), record))
            del record
        counts: dict[str, int] = {}
        for item in invocations:
            counts[item["site"]] = counts.get(item["site"], 0) + 1
        failures = [i for i in invocations if not i["ok"]]
        counts_ok = counts == expected
        report = {
            "schema": SCHEMA_VERSION,
            "ok": counts_ok and not failures,
            "invocations": counts,
            "expected": dict(expected),
            "skipped_by_layer_filter": dict(self.skipped),
            "layers": None if self.layers is None else sorted(self.layers),
            "checked": len(invocations),
            "per_invocation": invocations,
            "local_check_mode": "declared_tolerance_only",
        }
        if not counts_ok:
            report["reason"] = f"invocations {counts} != expected {expected}"
        elif failures:
            first = failures[0]
            bad = next(c for c in first["checks"] if not c["ok"])
            report["reason"] = (
                f"{first['site']}#{first['ordinal']} (layer {first['layer']}) {bad['result']}: "
                + (bad.get("reason") or "max_abs_err %.3g > atol %.3g" % (bad["max_abs_err"],
                                                                          bad["atol"])))
        return report


def validate_invocations(model, kernels, batch, loss_fn, *, registry, dtype: str,
                         layers: Iterable[int] | None, expected: dict[str, int]) -> dict[str, Any]:
    """One untimed forward + backward of a patched model with every live site checked.

    ``loss_fn(model, batch)`` is the workload's loss; ``layers`` the layer
    filter (``None``: all). The model must already hold the kernels' adapters.
    """
    from evograd.benchmark import get_task

    from .sites import set_tap

    validator = BoundaryValidator(lambda site: get_task(registry.require(site).op), dtype=dtype,
                                  layers=layers)
    set_tap(model, validator)
    try:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
    finally:
        set_tap(model, None)
        model.zero_grad(set_to_none=True)
    report = validator.finalize(expected=expected)
    report["patched"] = sorted(kernels.patched)
    return report


__all__ = ["BoundaryValidator", "SCHEMA_VERSION", "validate_invocations"]
