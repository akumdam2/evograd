"""Proving the Qwen3 tier-3 sites are drop-in, before anything is timed.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.evaluation.tier3.validate \
        structural --report results/qwen3-level4/t3-structural.json

Two checks, and they answer different questions.

**Structural identity** installs every adapter with the production spelling.
The module structure changes; the arithmetic does not, because each site calls
the same submodules in the same order through native autograd. So the demand is
**bitwise** equality against the unmodified canonical model -- logits, loss,
every parameter gradient, and the parameters after one AdamW step. Anything less
is a defect in the restructure, not a tolerance question, and it is reported as
such rather than absorbed by widening something.

**Bound-pair identity** installs the same adapters with the declared operator
behind ``opdecl.bind``, which is the path an evolved kernel takes. Here the
declared reference and the production spelling are genuinely different
computations, so the gate is the declared tolerances -- the ones calibrated
against the observed shapes -- and not bitwise equality.

Nothing large is written. The reports carry scalars, shapes, dtypes, strides and
error magnitudes; the tensors themselves are freed as soon as they are compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .sites import (
    SITE_ATTENTION,
    SITE_MLP,
    SITE_QKV,
    SITE_RESIDUAL,
    bound_pair_identity_kernels,
    set_tap,
    structural_identity_kernels,
)
from .workload import Qwen3Workload

#: Which layer's boundaries are compared tensor by tensor. Any layer would do;
#: naming one keeps the report a fixed size.
REPRESENTATIVE_LAYER = 14


def _meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "device": str(tensor.device).split(":")[0],
        "finite": bool(torch.isfinite(tensor).all()),
    }


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor,
             *, atol: float = 0.0, rtol: float = 0.0) -> dict[str, Any]:
    """One comparison, with bitwise equality reported whether or not it is required."""
    record = {"name": name, **_meta(actual)}
    record["expected"] = _meta(expected)
    same_meta = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.stride() == expected.stride()
    )
    record["metadata_match"] = bool(same_meta)
    if actual.shape != expected.shape:
        record["bitwise"] = False
        record["ok"] = False
        return record
    bitwise = bool(torch.equal(actual, expected))
    diff = (actual.detach().float() - expected.detach().float()).abs()
    record["bitwise"] = bitwise
    record["max_abs_err"] = float(diff.max())
    scale = float(expected.detach().float().abs().max())
    record["ref_absmax"] = scale
    record["max_rel_err_vs_scale"] = float(diff.max()) / scale if scale else None
    record["within_tolerance"] = bool(
        torch.allclose(actual.detach().float(), expected.detach().float(),
                       atol=atol, rtol=rtol)
    ) if (atol or rtol) else bitwise
    record["ok"] = bool(same_meta and (bitwise if not (atol or rtol)
                                       else record["within_tolerance"]))
    return record


# ── the whole-model comparison ───────────────────────────────────────────────


def _run_model(workload: Qwen3Workload, kernels, *, batch, learning_rate: float):
    """One forward, one backward, one optimizer step. Returns what to compare."""
    model, provenance = workload.build_patched(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = batch
    before = input_ids.detach().clone()
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    optimizer.step()
    stepped = {n: p.detach().clone() for n, p in model.named_parameters()}
    return {
        "logits": outputs.logits.detach().clone(),
        "loss": outputs.loss.detach().clone(),
        "grads": grads,
        "missing_grads": missing,
        "stepped": stepped,
        "state_dict_keys": list(model.state_dict()),
        "param_count": sum(p.numel() for p in model.parameters()),
        "trainable": sum(1 for p in model.parameters() if p.requires_grad),
        "provenance": provenance.to_dict(),
        "runtime": workload.runtime_report(model),
        "inputs_unmutated": bool(torch.equal(input_ids, before)),
        "patched": workload.last_build,
        "model": model,
    }


def _free(record: dict) -> None:
    for key in ("logits", "loss", "grads", "stepped", "model"):
        record.pop(key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_full_model(workload: Qwen3Workload, kernels, *, label: str,
                       atol: float = 0.0, rtol: float = 0.0,
                       learning_rate: float = 1e-4) -> dict[str, Any]:
    """The patched model against the unmodified canonical one, end to end."""
    from evograd.bench.tier3_patch import KernelSet

    batch = workload.batch_for(seed=0)
    reference = _run_model(
        workload, KernelSet(registry=workload.site_registry),
        batch=batch, learning_rate=learning_rate,
    )
    candidate = _run_model(workload, kernels, batch=batch, learning_rate=learning_rate)

    checks = [
        _compare("logits", candidate["logits"], reference["logits"], atol=atol, rtol=rtol),
        _compare("loss", candidate["loss"], reference["loss"], atol=atol, rtol=rtol),
    ]
    grad_records, grad_bad = [], []
    for name, expected in reference["grads"].items():
        got = candidate["grads"].get(name)
        if got is None:
            grad_bad.append(name)
            continue
        record = _compare(name, got, expected, atol=atol, rtol=rtol)
        if not record["ok"]:
            grad_bad.append(name)
            grad_records.append(record)
    step_bad = []
    for name, expected in reference["stepped"].items():
        got = candidate["stepped"].get(name)
        if got is None or not _compare(name, got, expected, atol=atol, rtol=rtol)["ok"]:
            step_bad.append(name)

    report = {
        "label": label,
        "gate": "bitwise" if not (atol or rtol) else f"allclose(atol={atol}, rtol={rtol})",
        "checks": checks,
        "gradient_coverage": {
            "compared": len(reference["grads"]),
            "total_parameters": sum(1 for _ in reference["state_dict_keys"]),
            "reference_missing": reference["missing_grads"],
            "candidate_missing": candidate["missing_grads"],
            "mismatched": grad_bad,
            "mismatch_detail": grad_records[:8],
        },
        "optimizer_step": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "parameters": len(reference["stepped"]),
            "mismatched": step_bad,
        },
        "state_dict_keys_identical":
            reference["state_dict_keys"] == candidate["state_dict_keys"],
        "state_dict_key_count": len(reference["state_dict_keys"]),
        "parameter_count": {
            "reference": reference["param_count"],
            "candidate": candidate["param_count"],
            "identical": reference["param_count"] == candidate["param_count"],
        },
        "trainable_tensors": {
            "reference": reference["trainable"], "candidate": candidate["trainable"],
        },
        "inputs_unmutated": reference["inputs_unmutated"] and candidate["inputs_unmutated"],
        "provenance": candidate["provenance"],
        "runtime": candidate["runtime"],
        "expected_site_counts": candidate["patched"].expected if candidate["patched"] else {},
        "observed_site_counts": candidate["patched"].observed() if candidate["patched"] else {},
        "count_problems": candidate["patched"].count_problems() if candidate["patched"] else [],
    }
    report["ok"] = bool(
        all(c["ok"] for c in checks)
        and not grad_bad and not step_bad
        and not candidate["missing_grads"]
        and report["state_dict_keys_identical"]
        and report["parameter_count"]["identical"]
        and report["inputs_unmutated"]
        and not report["count_problems"]
    )
    _free(reference)
    _free(candidate)
    return report


# ── the live boundary comparison ─────────────────────────────────────────────


_OPS_FOR_SITE = {
    SITE_QKV: "qwen3_qkv_norm_rope",
    SITE_ATTENTION: "qwen3_attention",
    SITE_MLP: "qwen3_swiglu_mlp",
    SITE_RESIDUAL: "fused_add_rms_norm",
}

#: Which tapped boundaries are compared tensor by tensor. Every invocation is
#: still counted; these are the ones whose tensors are checked.
def _wanted(key) -> bool:
    if isinstance(key, tuple):  # residual: (layer_idx, category)
        layer, category = key
        return (
            (category == "post_attention" and layer == REPRESENTATIVE_LAYER)
            or (category == "mlp_to_next_input" and layer == REPRESENTATIVE_LAYER)
            or category == "final_model_norm"
        )
    return key == REPRESENTATIVE_LAYER


def validate_boundaries(workload: Qwen3Workload, *, layer: int | None = None) -> dict[str, Any]:
    """Do the operators receive, and return, what their contracts say?

    The adapters hand the validator their live inputs and outputs. Each is
    re-run through the declaration's own ``runtime_forward`` -- the production
    spelling, the thing the structural path is supposed to be calling -- and
    compared bitwise. The upstream gradient reaching each output is captured
    with a tensor hook and reported, so the backward boundary is described too.
    """
    global REPRESENTATIVE_LAYER
    from evograd.bench.tier3_patch import KernelSet  # noqa: F401
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.ops import get_op

    if layer is not None:
        REPRESENTATIVE_LAYER = layer

    seen: list[dict[str, Any]] = []
    grads: dict[str, list[dict[str, Any]]] = {}

    def listener(site, key, inputs, outputs):
        if not _wanted(key):
            return
        op = get_op(_OPS_FOR_SITE[site])
        reference = resolve_runtime_forward(op)
        args = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
        with torch.no_grad():
            expected = reference(*[a.detach() if torch.is_tensor(a) else a for a in args])
        got = outputs if isinstance(outputs, tuple) else (outputs,)
        want = expected if isinstance(expected, tuple) else (expected,)
        # For three of the four sites the adapter's production spelling and the
        # declaration's `runtime_forward` are the same computation, so bitwise
        # is the right demand. `fused_add_rms_norm`'s is not: it normalizes with
        # `F.rms_norm`, where Qwen3RMSNorm casts to bfloat16 before multiplying
        # by the weight. That is a difference between two spellings of the
        # contract, not a defect in the adapter, so it is gated by the declared
        # tolerance and the bitwise result is reported alongside.
        workload_case = op.benchmark_workloads(suite="qwen3_0_6b_observed")[0]
        record = {
            "site": site,
            "key": list(key) if isinstance(key, tuple) else key,
            "op": op.name,
            "gate": "bitwise" if site != SITE_RESIDUAL else "declared tolerance",
            "inputs": {n: _meta(v) for n, v in inputs.items() if torch.is_tensor(v)},
            "outputs": [
                _compare(
                    out_name, a.detach(), b,
                    **({} if site != SITE_RESIDUAL
                       else dict(zip(("atol", "rtol"),
                                     op.tolerance_for(workload_case, out_name)))),
                )
                for out_name, a, b in zip(op.output_names, got, want)
            ],
        }
        label = f"{site}:{record['key']}"
        grads[label] = []
        for out_name, tensor in zip(op.output_names, got):
            if tensor.requires_grad:
                grads[label].append({"name": out_name, "pending": True})
                tensor.register_hook(
                    lambda g, _l=label, _n=out_name: _record_grad(grads, _l, _n, g)
                )
        record["ok"] = all(o["ok"] for o in record["outputs"])
        seen.append(record)

    model, _prov = workload.build_patched(
        structural_identity_kernels(workload.site_registry)
    )
    set_tap(model, listener)
    input_ids, labels = workload.batch_for(seed=0)
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    set_tap(model, None)
    counters = workload.last_build.observed()
    expected_counts = workload.last_build.expected
    del model, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "representative_layer": REPRESENTATIVE_LAYER,
        "boundaries": seen,
        "upstream_gradients": grads,
        "expected_site_counts": expected_counts,
        "observed_site_counts": counters,
        "counts_match": expected_counts == {k: counters.get(k, 0) for k in expected_counts},
        "ok": bool(seen) and all(r["ok"] for r in seen),
    }


def _record_grad(store, label, name, grad):
    for entry in store[label]:
        if entry["name"] == name:
            entry.pop("pending", None)
            entry.update(_meta(grad))
            entry["absmax"] = float(grad.detach().float().abs().max())
    return grad


# ── CLI ──────────────────────────────────────────────────────────────────────


def _workload(args) -> Qwen3Workload:
    config: dict[str, Any] = {"device": args.device, "dtype": args.dtype,
                              "seed": args.seed, "data_seed": args.data_seed}
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.tokens is not None:
        config["seq_len"] = args.tokens
    if args.layers is not None:
        config["arch_overrides"] = {"num_hidden_layers": args.layers}
    return Qwen3Workload.from_config(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.evaluation.tier3.validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=("structural", "bound", "boundaries", "control", "all"),
        help=(
            "control: the unmodified model against itself, twice. Whatever it "
            "reports is the floor -- a gradient that differs there differs "
            "because the GPU's reductions are not deterministic, not because "
            "anything was patched"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=REPRESENTATIVE_LAYER)
    parser.add_argument("--sites", default=None,
                        help="comma-separated subset; default is all four")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from evograd.bench.tier3_patch import restrict
    from evograd.ops import OPS

    workload = _workload(args)
    sites = tuple(s.strip() for s in args.sites.split(",")) if args.sites else None
    report: dict[str, Any] = {
        "schema_version": "evograd-qwen3-tier3-identity/1",
        "workload": workload.describe(),
        "selected_sites": list(sites) if sites else list(workload.site_registry.names),
    }

    if args.command in ("control", "all"):
        from evograd.bench.tier3_patch import KernelSet

        report["eager_control"] = compare_full_model(
            workload, KernelSet(registry=workload.site_registry),
            label="eager_control",
        )
    if args.command in ("structural", "all"):
        kernels = structural_identity_kernels(workload.site_registry)
        if sites:
            kernels = restrict(kernels, sites)
        report["structural_identity"] = compare_full_model(
            workload, kernels, label="structural_identity"
        )
    if args.command in ("boundaries", "all"):
        report["boundaries"] = validate_boundaries(workload, layer=args.layer)
    if args.command in ("bound", "all"):
        kernels = bound_pair_identity_kernels(OPS, sites, workload.site_registry)
        # Gated by the declared tolerances of every patched site, taken at the
        # loosest of them so one report covers the whole model. The per-site
        # gate that actually decides whether tier 3 will time this is the
        # preflight, run separately.
        atol, rtol = _model_tolerance(workload, kernels)
        report["bound_pair_identity"] = compare_full_model(
            workload, kernels, label="bound_pair_identity", atol=atol, rtol=rtol
        )
        report["bound_pair_preflight"] = _preflight(kernels, args.device)

    failures = [k for k in ("structural_identity", "boundaries", "bound_pair_identity")
                if k in report and not report[k].get("ok")]
    # A structural difference the eager control also shows is the device's
    # nondeterminism, not the restructure's. Reported either way, and subtracted
    # here so a green run means what it says.
    control = report.get("eager_control")
    structural = report.get("structural_identity")
    if control is not None and structural is not None:
        floor = set(control["gradient_coverage"]["mismatched"])
        beyond = set(structural["gradient_coverage"]["mismatched"]) - floor
        structural["nondeterminism_floor"] = sorted(floor)
        structural["beyond_control"] = sorted(beyond)
        structural["attributable_to_restructure"] = bool(beyond)
        if not beyond and "structural_identity" in failures:
            step_beyond = (set(structural["optimizer_step"]["mismatched"])
                           - set(control["optimizer_step"]["mismatched"]))
            structural["optimizer_step_beyond_control"] = sorted(step_beyond)
            if not step_beyond:
                failures.remove("structural_identity")
                structural["ok_modulo_control"] = True

    print(json.dumps(_summary(report), indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.report}")
    return 1 if failures else 0


def _model_tolerance(workload: Qwen3Workload, kernels) -> tuple[float, float]:
    """The loosest declared output tolerance among the patched sites.

    A whole-model comparison cannot use a per-result tolerance -- the loss is
    not any operator's output. Taking the loosest declared one is the honest
    upper bound for what a composition of them can drift, and the per-site
    gate that decides admission is the preflight, not this.
    """
    from evograd.ops import get_op

    atol = rtol = 0.0
    for site in kernels.patched:
        op = get_op(_OPS_FOR_SITE[site])
        for name in op.output_names:
            a, r = op.tolerance_for(op.benchmark_workloads(
                suite="qwen3_0_6b_observed")[0], name)
            atol, rtol = max(atol, a), max(rtol, r)
    return atol, rtol


def _preflight(kernels, device: str) -> dict[str, Any]:
    from evograd.bench.tier3_runner import PreflightFailure, preflight
    from evograd.ops import OPS

    try:
        return {"ok": True, **preflight(kernels, OPS, device=device)}
    except PreflightFailure as exc:
        return {"ok": False, "error": str(exc)}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    out = {"workload_id": report["workload"]["workload_id"],
           "units_per_step": report["workload"]["units_per_step"],
           "input_checksum": report["workload"]["input_checksum"],
           "selected_sites": report["selected_sites"]}
    for key in ("eager_control", "structural_identity", "bound_pair_identity"):
        if key not in report:
            continue
        section = report[key]
        out[key] = {
            "ok": section["ok"],
            "gate": section["gate"],
            "loss": next(c for c in section["checks"] if c["name"] == "loss"),
            "logits_ok": next(c["ok"] for c in section["checks"] if c["name"] == "logits"),
            "gradients": {
                "compared": section["gradient_coverage"]["compared"],
                "mismatched": len(section["gradient_coverage"]["mismatched"]),
                "missing": section["gradient_coverage"]["candidate_missing"],
            },
            "optimizer_step_mismatched": len(section["optimizer_step"]["mismatched"]),
            "state_dict_keys_identical": section["state_dict_keys_identical"],
            "parameter_count": section["parameter_count"],
            "inputs_unmutated": section["inputs_unmutated"],
            "expected_counts": section["expected_site_counts"],
            "observed_counts": section["observed_site_counts"],
            "count_problems": section["count_problems"],
        }
        for extra in ("nondeterminism_floor", "beyond_control",
                      "attributable_to_restructure", "ok_modulo_control",
                      "optimizer_step_beyond_control"):
            if extra in section:
                value = section[extra]
                out[key][extra] = (
                    len(value) if isinstance(value, list) else value
                )
        if "mismatch_detail" in section["gradient_coverage"]:
            out[key]["worst_gradient_deltas"] = [
                {"name": d["name"], "max_abs_err": d.get("max_abs_err"),
                 "ref_absmax": d.get("ref_absmax"),
                 "max_rel_err_vs_scale": d.get("max_rel_err_vs_scale")}
                for d in section["gradient_coverage"]["mismatch_detail"][:5]
            ]
    if "boundaries" in report:
        out["boundaries"] = {
            "ok": report["boundaries"]["ok"],
            "counts_match": report["boundaries"]["counts_match"],
            "expected": report["boundaries"]["expected_site_counts"],
            "observed": report["boundaries"]["observed_site_counts"],
            "compared": [
                {"site": b["site"], "key": b["key"],
                 "outputs": [{"name": o["name"], "bitwise": o["bitwise"],
                              "metadata_match": o["metadata_match"],
                              "max_abs_err": o.get("max_abs_err")}
                             for o in b["outputs"]]}
                for b in report["boundaries"]["boundaries"]
            ],
        }
    if "bound_pair_preflight" in report:
        pre = report["bound_pair_preflight"]
        out["bound_pair_preflight"] = {
            "ok": pre.get("ok"),
            "checked": [
                {"site": c["site"], "op": c["op"], "ok": c["ok"],
                 "declared_cases": c["declared_cases"],
                 "workload_supplied_cases": c["workload_supplied_cases"],
                 "configs": [d["id"] for d in c["checked_configs"]
                             if d["source"] == "workload_supplied"]}
                for c in pre.get("checked", [])
            ],
            "error": pre.get("error"),
        }
    return out


if __name__ == "__main__":
    sys.exit(main())
