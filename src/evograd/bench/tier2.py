"""Tier 2 (operator), `fair` protocol — the operator measured the way training reaches it.

At tier 1 *you* are the autograd engine — you call ``forward_with_saved``, you
hold the saved state in a local, you hand the upstream gradient back to
``backward_from_saved`` yourself. No training loop does that. It writes

    y = model(x)
    y.backward(dy)

and PyTorch does everything in between: records a graph node, keeps the saved
state alive in its ``ctx``, schedules the backward, routes the gradient, and
runs ``AccumulateGrad`` to write into ``.grad``. That work is real and a
training step pays it every iteration, which is why a tier-1 speedup cannot be
reported as a training speedup — and why tier 2 exists to put every provider
behind the same door and charge them all the same framework tax.

Four providers, one shape. Each is an ``nn.Module`` holding the declared
``parameter_args`` as ``nn.Parameter`` and calling a differentiable callable:

    eager      resolve_runtime_forward(op)          — F.layer_norm, autograd's own
    compile    torch.compile(eager, dynamic=False)
    <baseline> bind(op, baseline_candidate_module(op, name))   — liger, cublas_pair…
    candidate  bind(op, evolved_module)             — the evolved Triton pair

The last two go through ``opdecl.bind``, which is the same
``torch.autograd.Function`` a deployed evolved kernel would use, so tier 2
measures the deployment path rather than a benchmark-only spelling of it.

Parameters are built once and copied into every provider, so all four run on
byte-identical weights. That matters more than it sounds: ``nn.LayerNorm``
initializes weight=1/bias=0 while the declaration's ``make_inputs`` draws them
from ``randn``, and letting each provider self-initialize would compare
implementations on different data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from evograd.opdecl.activity import OpDecl, Workload, bind_shape
from evograd.opdecl.baselines import baseline_candidate_module
from evograd.opdecl.bind import bind
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle, resolve_runtime_forward

TIER2_PROTOCOL_VERSION = "evograd-tier2-operator-v1"

#: Liger's published benchmark sizes its loops by a time budget and reports
#: these quantiles; matching them is what makes our numbers commensurable with
#: theirs. `rep` is a duration in milliseconds, not an iteration count.
DEFAULT_REP_MS = 500
DEFAULT_WARMUP_MS = 25
QUANTILES = (0.5, 0.2, 0.8)


class OperatorModule(torch.nn.Module):
    """One provider: declared parameters, plus a differentiable callable.

    ``call`` takes the operator's full declared argument list by keyword. The
    module supplies the parameters and the scalar ``Inactive`` args; the caller
    supplies the activations. That split is exactly what ``parameter_args``
    declares, and it is the only thing separating an ``nn.Module`` from the
    positional pair interface of tier 1.
    """

    def __init__(
        self,
        op: OpDecl,
        call: Callable[..., torch.Tensor],
        *,
        parameters: dict[str, torch.Tensor],
        adapter_kind: str,
    ):
        super().__init__()
        self._op = op
        self._call = call
        self.adapter_kind = adapter_kind
        self._parameter_names = tuple(op.parameter_args or ())
        self._activation_names = tuple(
            arg.name for arg in op.active_args()
            if arg.name not in self._parameter_names
        )
        for name in self._parameter_names:
            self.register_parameter(
                name, torch.nn.Parameter(parameters[name].detach().clone())
            )
        for arg in op.tensor_inactive_args():
            self.register_buffer(f"_inactive_{arg.name}", parameters[arg.name])

        # A positional call template, resolved once. `forward` runs inside the
        # timed region for *every* provider, eager included — and eager is the
        # row every speedup is divided by — so anything spent here inflates all
        # of them and compresses the ratios toward 1. At 1024 rows the forward
        # is 12 us, where a few microseconds of dict building is not noise.
        # Positional also skips the keyword validation in `bind`'s wrapper.
        arg_names = [arg.name for arg in op.args]
        self._template: list[Any] = [
            getattr(self, f"_inactive_{arg.name}")
            if arg.name in {a.name for a in op.tensor_inactive_args()}
            else parameters.get(arg.name, getattr(arg, "default", None))
            for arg in op.args
        ]
        self._activation_slots = [arg_names.index(n) for n in self._activation_names]
        self._parameter_slots = [
            (arg_names.index(n), n) for n in self._parameter_names
        ]

    def forward(self, *activations: torch.Tensor) -> torch.Tensor:
        args = list(self._template)
        for slot, activation in zip(self._activation_slots, activations):
            args[slot] = activation
        for slot, name in self._parameter_slots:
            args[slot] = getattr(self, name)
        return self._call(*args)

    def activations(self, values: dict[str, Any]) -> list[torch.Tensor]:
        """The call arguments, taken from a declared input dict."""
        return [values[name] for name in self._activation_names]


def _require_declared_split(op: OpDecl) -> None:
    """The declaration must say which Active args are module state.

    Only ``None`` — undeclared — is refused. An empty tuple is a positive
    statement that the operator has no parameters, which is true of ``geglu``,
    ``softmax`` and every pure elementwise op, and is perfectly measurable
    here: the framework overhead this tier exists to charge — graph recording,
    engine scheduling, ``AccumulateGrad`` into the activations' ``.grad`` — is
    paid whether or not the module owns state. (An earlier version of this
    function refused parameter-free operators on the reasoning that tier 1
    already measured them. That was wrong: tier 1 never enters the autograd
    engine at all, which is the entire difference between the two tiers.)
    """
    if op.parameter_args is None:
        raise ValueError(
            f"{op.name}: the operator tier measures an nn.Module, which holds "
            "parameters and takes activations as call arguments. The "
            "declaration does not say which Active args are which. Set "
            "`parameter_args` on the declaration — to the names of its "
            "parameters, or to `()` if it has none."
        )


def build_parameters(
    op: OpDecl, workload: Workload, device: str = "cuda"
) -> dict[str, Any]:
    """Declared inputs, shared verbatim by every provider."""
    return make_case_inputs(op, workload, device=device)


def eager_module(
    op: OpDecl, values: dict[str, Any], *, compile_it: bool = False
) -> OperatorModule:
    """Eager PyTorch, through the production spelling.

    ``resolve_runtime_forward`` is ``F.layer_norm`` where the declaration has a
    fused equivalent — the same function ``nn.LayerNorm.forward`` calls, so this
    module is that module, built generically. Timing the primitive ``forward``
    instead would measure a strawman; ``verify_runtime_forward`` has already
    checked the two agree numerically.
    """
    reference = resolve_runtime_forward(op)
    module = OperatorModule(
        op,
        reference,
        parameters=values,
        adapter_kind="pytorch_eager_module",
    )
    if compile_it:
        module.adapter_kind = "torch_compile_module"
    return module


def pair_module(
    op: OpDecl,
    candidate,
    *,
    adapter_kind: str,
    values: dict[str, Any],
) -> OperatorModule:
    """Any forward/backward pair, made differentiable and given parameters.

    ``bind`` is the load-bearing part: it turns ``forward_with_saved`` /
    ``backward_from_saved`` into a ``torch.autograd.Function``, routing the
    saved state through ``save_for_backward`` (splitting tensors from plain
    values so the pair may save whatever it likes) and placing each returned
    gradient in its declared argument slot with ``None`` for inactive args.
    """
    return OperatorModule(
        op, bind(op, candidate), parameters=values, adapter_kind=adapter_kind
    )


def baseline_pair_module(
    op: OpDecl, baseline: str, values: dict[str, Any]
) -> OperatorModule:
    """A reviewed pair baseline (liger, cublas_pair, …) as an nn.Module."""
    return pair_module(
        op,
        baseline_candidate_module(op, baseline),
        adapter_kind=f"declared_{baseline}_pair_module",
        values=values,
    )


# ── correctness ──────────────────────────────────────────────────────────────


def check_module(
    op: OpDecl,
    module: OperatorModule,
    values: dict[str, Any],
    workload: Workload,
) -> dict[str, Any]:
    """Verify a provider through the path that will be timed.

    Tier 1 checks ``forward_with_saved`` / ``backward_from_saved`` directly.
    That is not what tier 2 runs: here the gradients arrive via
    ``AccumulateGrad`` into ``.grad`` after ``y.backward(dy)``, through
    ``bind``'s slot routing. A pair can be correct called directly and wrong
    once wrapped — a misplaced gradient slot looks exactly like that — so the
    gate has to exercise the wrapped path.
    """
    workload_values = {
        name: (value.detach().clone() if torch.is_tensor(value) else value)
        for name, value in values.items()
    }
    dy = workload_values[op.upstream_grad_name]
    y_ref, expected = oracle(op, workload_values)

    activations = [
        a.detach().clone().requires_grad_(True) for a in module.activations(workload_values)
    ]
    for parameter in module.parameters():
        parameter.grad = None
    y = module(*activations)
    y.backward(dy)
    torch.cuda.synchronize()

    checks: dict[str, Any] = {}
    atol, rtol = op.tolerance_for(workload)
    checks["y"] = _compare(y, y_ref, atol, rtol)

    by_grad_name = {arg.grad_name: arg.name for arg in op.active_args()}
    got: dict[str, torch.Tensor | None] = {}
    for name, activation in zip(module._activation_names, activations):
        got[name] = activation.grad
    for name in module._parameter_names:
        got[name] = getattr(module, name).grad
    for grad_name in op.grad_names():
        source = by_grad_name[grad_name]
        grad_atol, grad_rtol = op.tolerance_for(workload, grad_name)
        checks[grad_name] = _compare(
            got.get(source), expected[grad_name], grad_atol, grad_rtol
        )

    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def _compare(actual, expected, atol: float, rtol: float) -> dict[str, Any]:
    if not torch.is_tensor(actual):
        return {"ok": False, "reason": "no gradient produced"}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "ok": False,
            "reason": (
                f"shape/dtype {tuple(actual.shape)}/{actual.dtype} != "
                f"{tuple(expected.shape)}/{expected.dtype}"
            ),
        }
    difference = (actual.float() - expected.float()).abs()
    return {
        "ok": bool(
            torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
        ),
        "max_abs_error": float(difference.max()),
    }


# ── timing ───────────────────────────────────────────────────────────────────


def _summarize(samples) -> dict[str, float]:
    """do_bench's quantile triple, in the shape the canonical report reads."""
    median, low, high = (float(v) for v in samples)
    return {"median_ms": median, "q20_ms": low, "q80_ms": high}


def measure_module(
    op: OpDecl,
    module: OperatorModule,
    values: dict[str, Any],
    *,
    rep_ms: int = DEFAULT_REP_MS,
    warmup_ms: int = DEFAULT_WARMUP_MS,
) -> dict[str, Any]:
    """Time one provider's forward and full training step.

    Uses ``triton.testing.do_bench``, which is what Liger's published benchmark
    uses, so the numbers are commensurable with theirs: loops sized by a wall
    clock budget rather than a fixed count, L2 flushed between samples, and the
    (0.5, 0.2, 0.8) quantiles rather than a bare mean.

    ``grad_to_none`` covers the activations only, matching Liger. Parameter
    gradients therefore accumulate across the whole run. That is faithful to
    training, where ``.grad`` accumulates until ``zero_grad``, and it is applied
    identically to every provider — but it is a real cost inside the timed
    region and worth stating rather than discovering.
    """
    from triton.testing import do_bench

    activations = [
        a.detach().clone().requires_grad_(True) for a in module.activations(values)
    ]
    dy = values[op.upstream_grad_name]

    def forward_only():
        return module(*activations)

    def full_step():
        # No `retain_graph`. Liger's published snippet sets it, because there
        # the forward sits outside the timed callable and the graph has to
        # survive repeated backwards. Here the forward is inside, so every call
        # builds its own graph and retaining is not merely pointless — it
        # breaks `torch.compile`, whose donated-buffer optimization requires
        # `retain_graph=False` and raises rather than silently degrading, and it
        # keeps every iteration's saved tensors alive for the whole 500 ms
        # measurement window. Training frees the graph; so does this.
        y = module(*activations)
        y.backward(dy)

    forward = do_bench(
        forward_only, warmup=warmup_ms, rep=rep_ms, quantiles=list(QUANTILES)
    )
    step = do_bench(
        full_step,
        warmup=warmup_ms,
        rep=rep_ms,
        grad_to_none=activations,
        quantiles=list(QUANTILES),
    )

    # Peak memory is measured outside the timed loops: the tier-1 saved-bytes
    # accounting does not exist here, because the saved state lives inside
    # autograd's ctx where nothing can read it.
    for parameter in module.parameters():
        parameter.grad = None
    for activation in activations:
        activation.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    y = module(*activations)
    y.backward(dy)
    torch.cuda.synchronize()
    peak_memory = int(torch.cuda.max_memory_allocated())

    return {
        "forward": _summarize(forward),
        "full_step": _summarize(step),
        "peak_memory_bytes": peak_memory,
        "adapter_kind": module.adapter_kind,
    }


# ── orchestration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    """How to build one provider, resolved before any GPU work happens."""

    name: str
    kind: str  # eager | compile | baseline_pair | candidate_pair
    baseline: str | None = None
    candidate_module: Any = None


def identity_control_specs() -> tuple[ProviderSpec, ...]:
    """The same eager module under two names.

    Whatever speedup this reports is the driver's noise floor, not a result.
    It answers the one question a new benchmark cannot answer about itself:
    when the two things being compared are provably identical, does the
    protocol say so? `tier1.py` has had this since it was written; a tier
    without it is a tier whose numbers nobody has calibrated.
    """
    return (
        ProviderSpec(name="eager", kind="eager"),
        ProviderSpec(name="eager_control", kind="eager"),
    )


def default_provider_specs(
    *, candidate_module=None, baseline: str | None = "liger", compile_baseline: bool = True
) -> tuple[ProviderSpec, ...]:
    specs = [ProviderSpec(name="eager", kind="eager")]
    if compile_baseline:
        specs.append(ProviderSpec(name="torch_compile", kind="compile"))
    if baseline:
        specs.append(
            ProviderSpec(name=baseline, kind="baseline_pair", baseline=baseline)
        )
    if candidate_module is not None:
        specs.append(
            ProviderSpec(
                name="candidate", kind="candidate_pair", candidate_module=candidate_module
            )
        )
    return tuple(specs)


def build_provider(
    op: OpDecl, spec: ProviderSpec, values: dict[str, Any]
) -> OperatorModule:
    if spec.kind == "eager":
        return eager_module(op, values)
    if spec.kind == "compile":
        module = eager_module(op, values, compile_it=True)
        # `dynamic=False` specializes on the shape, which is the comparison the
        # benchmark intends: a per-shape compiled kernel, not one guarded for
        # every size. Compilation happens on the first call, inside do_bench's
        # warmup, so it is outside every timed sample.
        #
        # `fullgraph=True` is the graph-break assertion. Dynamo reports a break
        # by falling back to eager and running slower, not by raising, so
        # without it this row could be eager wearing a different label — and it
        # would look like a plausible compiled result. With it, a break raises
        # and `run_case` records the provider as failed, which is the honest
        # outcome.
        module._call = torch.compile(module._call, dynamic=False, fullgraph=True)
        return module
    if spec.kind == "baseline_pair":
        return baseline_pair_module(op, spec.baseline, values)
    if spec.kind == "candidate_pair":
        return pair_module(
            op,
            spec.candidate_module,
            adapter_kind="candidate_pair_module",
            values=values,
        )
    raise ValueError(f"unknown provider kind {spec.kind!r}")


def run_case(
    op: OpDecl,
    workload: Workload,
    specs: tuple[ProviderSpec, ...],
    *,
    device: str = "cuda",
    rep_ms: int = DEFAULT_REP_MS,
    warmup_ms: int = DEFAULT_WARMUP_MS,
    check: bool = True,
) -> dict[str, Any]:
    """One shape, every provider, correctness before timing."""
    values = build_parameters(op, workload, device=device)
    providers: dict[str, Any] = {}
    for spec in specs:
        entry: dict[str, Any] = {"kind": spec.kind}
        try:
            module = build_provider(op, spec, values)
            if check:
                verdict = check_module(op, module, values, workload)
                entry["correctness"] = verdict
                if not verdict["ok"]:
                    entry["ok"] = False
                    entry["error"] = "failed correctness at this shape"
                    providers[spec.name] = entry
                    continue
            entry.update(measure_module(
                op, module, values, rep_ms=rep_ms, warmup_ms=warmup_ms
            ))
            entry["ok"] = True
        except Exception as exc:  # a provider that dies must not take the case
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        providers[spec.name] = entry

    return {
        "dims": dict(workload.dims),
        "dtype": workload.dtype,
        "providers": providers,
    }


def run_tier2(
    op: OpDecl,
    *,
    workloads: tuple[Workload, ...],
    specs: tuple[ProviderSpec, ...],
    device: str = "cuda",
    rep_ms: int = DEFAULT_REP_MS,
    warmup_ms: int = DEFAULT_WARMUP_MS,
    check: bool = True,
) -> dict[str, Any]:
    _require_declared_split(op)
    from evograd.bench.tier1 import environment_fingerprint

    cases = [
        run_case(
            op, workload, specs,
            device=device, rep_ms=rep_ms, warmup_ms=warmup_ms, check=check,
        )
        for workload in workloads
    ]
    return {
        "protocol": TIER2_PROTOCOL_VERSION,
        "op": op.name,
        "timing_protocol": {
            "driver": "triton.testing.do_bench",
            "rep_ms": rep_ms,
            "warmup_ms": warmup_ms,
            "quantiles": list(QUANTILES),
            "grad_to_none": "activations only, matching Liger; parameter .grad accumulates",
            "step": "y = model(*activations); y.backward(dy)",
            "parameters": "built once and copied into every provider",
        },
        "environment": environment_fingerprint(),
        "cases": cases,
    }
