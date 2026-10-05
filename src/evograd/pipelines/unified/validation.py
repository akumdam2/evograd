"""Deterministic, domain-aware numerical variants for unified seed admission.

This is a finite correctness gate, not a proof over every input value. It uses
the declaration's input recipes, autograd oracle, and existing tolerances.
"""

from __future__ import annotations

from dataclasses import asdict
import math

import torch

from evograd.evaluation.tier2.runner import candidate_module, check_module
from evograd.opdecl.activity import bind_shape
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import (
    arg_dtype_name, as_output_tuple, case_seed, make_case_inputs, resolve_dtype,
    upstream_grad_values,
)
from evograd.opdecl.oracle import oracle
from evograd.opdecl.verify import _check


class _FactoryTorch:
    """Preserve a factory's recipe while varying its explicit RNG seed."""

    def __init__(self, offset):
        self.offset = offset

    def __getattr__(self, name):
        return getattr(torch, name)

    def manual_seed(self, seed):
        return torch.manual_seed((int(seed) + self.offset) % (2**63 - 1))


def _copy_values(values):
    return {name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in values.items()}


def _factory_sample(op, workload, device, offset=0):
    # Recipes receive torch explicitly, so no global monkeypatching is needed.
    # Restore all visible generators, including recipes that seed CUDA while
    # constructing a CPU case.
    devices = None if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        if not offset:
            return make_case_inputs(op, workload, device=device)
        proxy = _FactoryTorch(offset)
        proxy.manual_seed(case_seed(workload))
        if op.make_inputs is not None:
            return op.make_inputs(proxy, op, workload, device)
        values = make_case_inputs(op, workload, device=device)
        # The default factory resets its own seed. Draw its unconstrained
        # floating tensors again after selecting the alternate seed.
        proxy.manual_seed(case_seed(workload))
        names = [arg.name for arg in op.active_args()] + list(op.upstream_grad_names)
        for name in names:
            value = values[name]
            values[name] = torch.randn(value.shape, device=value.device, dtype=value.dtype) * 0.5
        return values


def _stress_arguments(op, module):
    family = op.family or getattr(module, "UNIFIED_DEPLOYMENT_CONFIG", {}).get("family")
    eligible = family == "norm" or (op.make_inputs is None and not op.extra_constraints)
    if not eligible:
        return (), "custom/constrained input domain has no generic stress recipe"
    parameters = set(op.parameter_args or ())
    names = tuple(arg.name for arg in op.active_args() if arg.name not in parameters)
    return names, "normalization activations" if family == "norm" else "default unconstrained activations"


def _variants(op, workload, module, device):
    base = _factory_sample(op, workload, device)
    yield "factory_default", base, False, {}
    for offset in (104729, 209759):
        values = _factory_sample(op, workload, device, offset)
        # Deployment scalar parameters belong to c, not the random sample.
        for arg in op.scalar_inactive_args():
            values[arg.name] = base.get(arg.name, arg.default)
        changed = [arg.name for arg in op.args if torch.is_tensor(base.get(arg.name))
                   and torch.is_tensor(values.get(arg.name))
                   and not torch.equal(base[arg.name], values[arg.name])]
        yield f"factory_seed_{offset}", values, False, {"changed_primal_tensors": changed}
    for pattern in ("ones", "alternating"):
        values = _copy_values(base)
        for name in op.upstream_grad_names:
            value = values[name]
            if pattern == "ones":
                values[name] = torch.ones_like(value)
            else:
                signs = torch.arange(value.numel(), device=value.device) % 2
                values[name] = (2 * signs - 1).to(value.dtype).reshape(value.shape)
        yield f"upstream_{pattern}", values, False, {}
    names, reason = _stress_arguments(op, module)
    names = tuple(name for name in names if torch.is_tensor(base.get(name))
                  and base[name].is_floating_point())
    for pattern in ("large_constant", "cancellation") if names else ():
        values = _copy_values(base)
        for name in names:
            value = values[name]
            if pattern == "large_constant":
                values[name] = torch.full_like(value, 1000.0)
            else:
                signs = torch.arange(value.numel(), device=value.device) % 2
                values[name] = ((1 - 2 * signs) * 1000).to(value.dtype).reshape(value.shape)
        yield f"stress_{pattern}", values, True, {"changed_activations": list(names), "recipe": reason}


def _input_contract(op, workload, values, device):
    expected_device = torch.device(device)
    entries = [(arg.name, arg) for arg in op.args]
    entries += [(arg.grad_name, arg) for arg in op.outputs]
    for name, arg in entries:
        if getattr(arg, "shape", None) is None:
            continue
        value = values.get(name)
        shape = bind_shape(arg.shape, workload.dims)
        dtype = resolve_dtype(arg_dtype_name(arg, workload))
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(f"input {name}: expected tensor {tuple(shape)}/{dtype}")
        if value.device.type != expected_device.type or (
            expected_device.index is not None and value.device.index != expected_device.index
        ):
            raise ValueError(f"input {name}: expected device {device}, got {value.device}")
    for name in op.upstream_grad_names:
        if not bool(torch.isfinite(values[name]).all()):
            raise ValueError(f"upstream gradient {name} is non-finite")


def _pair_checks(op, module, workload, values, expected_output, expected_grads):
    values = _copy_values(values)
    forward, backward = lookup_pair(op, module)
    args = [values.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
    with torch.no_grad():
        output, saved = forward(*args)
        gradients = backward(upstream_grad_values(op, values), saved,
                             **backward_inactive_kwargs(op, backward, values))
    gradients = (gradients,) if torch.is_tensor(gradients) else tuple(gradients)
    if len(gradients) != len(op.grad_names()):
        raise ValueError(f"expected {len(op.grad_names())} gradients, got {len(gradients)}")
    checks = []
    for declared, actual, expected in zip(op.outputs, as_output_tuple(op, output),
                                         as_output_tuple(op, expected_output)):
        checks.append(asdict(_check(declared.name, actual, expected,
                                    *op.tolerance_for(workload, declared.name))))
    for name, actual in zip(op.grad_names(), gradients):
        checks.append(asdict(_check(name, actual, expected_grads[name],
                                    *op.tolerance_for(workload, name))))
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def _run_variant(op, module, workload, values, device, stress):
    try:
        _input_contract(op, workload, values, device)
        expected_output, expected_grads = oracle(op, values)
        references = (*as_output_tuple(op, expected_output), *expected_grads.values())
        if not all(bool(torch.isfinite(value).all()) for value in references):
            raise ValueError("independent oracle produced non-finite outputs or gradients")
    except Exception as exc:
        return {"status": "skipped_invalid_reference" if stress else "failed_reference",
                "ok": None if stress else False, "error_type": type(exc).__name__, "error": str(exc)}
    try:
        pair = _pair_checks(op, module, workload, values, expected_output, expected_grads)
        # Reuse the deployment checker, including its declared gradient routing
        # and per-result tolerances. Inputs are isolated from direct-pair mutation.
        deployment_values = _copy_values(values)
        deployed = candidate_module(op, module, values=deployment_values)
        deployment = check_module(op, deployed, deployment_values, workload)
        passed = bool(pair["ok"] and deployment["ok"])
        return {"status": "passed" if passed else "failed_candidate", "ok": passed,
                "pair": pair, "deployment": deployment}
    except Exception as exc:
        return {"status": "failed_candidate", "ok": False,
                "error_type": type(exc).__name__, "error": str(exc)}


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(item) for item in value]
    return value


def validate_numerical_variants(op, module, *, device="cuda") -> dict:
    """Check finite factory, tangent, and eligible activation stress variants."""
    cases = []
    for workload in op.correctness:
        try:
            for name, values, stress, metadata in _variants(op, workload, module, device):
                result = _run_variant(op, module, workload, values, device, stress)
                cases.append({"variant": name, "dims": dict(workload.dims),
                              "dtype": workload.dtype, **metadata, **result})
        except Exception as exc:
            cases.append({"variant": "input_factory", "dims": dict(workload.dims),
                          "dtype": workload.dtype, "status": "failed_reference", "ok": False,
                          "error_type": type(exc).__name__, "error": str(exc)})
    passed = sum(case["ok"] is True for case in cases)
    failed = sum(case["ok"] is False for case in cases)
    skipped = sum(case["ok"] is None for case in cases)
    stress_names, stress_reason = _stress_arguments(op, module)
    return _json_finite({
        "ok": passed > 0 and failed == 0, "cases": cases,
        "coverage": {"checked": passed + failed, "passed": passed, "failed": failed,
                     "skipped_invalid_reference": skipped,
                     "stress_eligible_activations": list(stress_names), "stress_recipe": stress_reason},
        "scope": "finite deterministic samples; not a proof for all input values",
    })
