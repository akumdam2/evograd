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

import hashlib
import statistics
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from evograd.opdecl.activity import OpDecl, Workload, bind_shape
from evograd.opdecl.baselines import baseline_candidate_module
from evograd.opdecl.bind import bind
from evograd.opdecl.inputs import (
    as_output_tuple,
    make_case_inputs,
    upstream_grad_values,
)
from evograd.opdecl.oracle import oracle, resolve_runtime_forward

TIER2_PROTOCOL_VERSION = "evograd-tier2-operator-v1"

#: Liger's published benchmark sizes its loops by a time budget and reports
#: these quantiles; matching them is what makes our numbers commensurable with
#: theirs. `rep` is a duration in milliseconds, not an iteration count.
DEFAULT_REP_MS = 500
DEFAULT_WARMUP_MS = 25
QUANTILES = (0.5, 0.2, 0.8)

#: A fixed sample count, not a wall-clock budget. `do_bench`'s `rep` is a
#: duration, so a fast provider is sampled many more times than a slow one and
#: the two quantile estimates rest on different amounts of evidence. Every
#: provider now contributes the same number of measurements.
REPETITIONS = 500
#: Untimed iterations before sampling. Enough to finish Triton autotuning and
#: settle the allocator; compilation is already done by the correctness gate.
WARMUP_ITERS = 10
#: Evicts L2 between samples (GH200 has 60 MB), the discipline `do_bench`
#: applies, so one sample does not read the previous sample's cache.
L2_FLUSH_BYTES = 256 * 1024 * 1024


def _timed_samples(fn, *, grad_to_none=None, reps=REPETITIONS,
                   warmup=WARMUP_ITERS) -> list[float]:
    """`warmup` untimed iterations, then exactly `reps` event-timed ones."""
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for index in range(reps):
        if grad_to_none:
            for tensor in grad_to_none:
                tensor.grad = None
        flush.zero_()
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def tensor_checksum(value) -> str:
    """A content fingerprint, so "identical inputs" is checked, not assumed."""
    if not torch.is_tensor(value):
        return f"scalar:{value}"
    payload = value.detach().to(torch.float32).cpu().numpy().tobytes()
    return (f"{hashlib.sha256(payload).hexdigest()[:16]}:"
            f"{tuple(value.shape)}:{value.dtype}:{value.stride()}")


def install_call_template(
    module: torch.nn.Module,
    op: OpDecl,
    values: dict[str, Any],
    *,
    parameter_names: tuple[str, ...],
    activation_names: tuple[str, ...],
) -> None:
    """Register the inactive tensors and resolve the positional call template.

    The declaration's argument order is the only thing that says where a value
    goes. Concatenating activations, parameters and scalars gives the same list
    only when the declaration happens to list them in that order and declares no
    inactive tensors; where it does not — ``qwen3_qkv_norm_rope``'s ``cos`` and
    ``sin`` sit between the parameters and ``eps`` — that shortcut drops them.

    Resolved once, because ``forward`` runs inside the timed region for *every*
    provider, eager included — and eager is the row every speedup is divided by —
    so anything spent here inflates all of them and compresses the ratios toward
    1. At 1024 rows the forward is 12 us, where a few microseconds of dict
    building is not noise.
    """
    inactive = {arg.name for arg in op.tensor_inactive_args()}
    for arg in op.tensor_inactive_args():
        module.register_buffer(f"_inactive_{arg.name}", values[arg.name])
    arg_names = [arg.name for arg in op.args]
    module._template = [
        getattr(module, f"_inactive_{arg.name}") if arg.name in inactive
        else values.get(arg.name, getattr(arg, "default", None))
        for arg in op.args
    ]
    module._activation_slots = [arg_names.index(n) for n in activation_names]
    module._parameter_slots = [(arg_names.index(n), n) for n in parameter_names]


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
        install_call_template(
            self, op, parameters,
            parameter_names=self._parameter_names,
            activation_names=self._activation_names,
        )

    def forward(self, *activations: torch.Tensor) -> Any:
        # Returns whatever the declaration says: a Tensor, or an ordered tuple
        # of them. Nothing here unwraps a tuple -- the caller normalizes
        # against `op.outputs`, so a provider that returns the wrong arity is
        # caught by the contract rather than silently reduced to its first
        # element.
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


def candidate_module(op: OpDecl, candidate, *, values: dict[str, Any]):
    """A generated candidate, through whichever route it declares.

    An artifact that exports a deployment entry is benchmarked through its own
    static ``torch.autograd.Function`` -- the same object tier 3 patches into
    the model, so the two tiers measure one implementation. A pair-only
    candidate predating the contract still works, through the binder, and says
    so in its adapter kind.
    """
    from evograd.pipelines.shared.artifact import deployment_entry, validate_artifact

    entry = deployment_entry(candidate)
    if entry is not None:
        validate_artifact(op, candidate)     # malformed never silently demotes
        return native_module(op, entry, values=values,
                             adapter_kind="evolved_direct_autograd_module")
    return pair_module(op, candidate, adapter_kind="legacy_bind_pair_module",
                       values=values)


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


class NativeModule(torch.nn.Module):
    """A provider that supplies its own autograd, with no binder in between.

    ``OperatorModule`` exists to make an arbitrary generated pair trainable: it
    reads the declaration, routes gradients by name, and validates arity on
    every call. A provider that already ships a ``torch.autograd.Function``
    needs none of that, and paying for it measures EvoGrad rather than the
    kernel. This wrapper holds the parameters and forwards positionally; it
    contains no logic of its own.
    """

    def __init__(self, op: OpDecl, call, *, parameters: dict[str, torch.Tensor],
                 parameter_names: tuple[str, ...],
                 activation_names: tuple[str, ...], adapter_kind: str):
        super().__init__()
        self._call = call
        self.adapter_kind = adapter_kind
        self._parameter_names = parameter_names
        self._activation_names = activation_names
        for name in parameter_names:
            self.register_parameter(
                name, torch.nn.Parameter(parameters[name].detach().clone())
            )
        install_call_template(
            self, op, parameters,
            parameter_names=parameter_names,
            activation_names=activation_names,
        )

    def forward(self, *activations):
        args = list(self._template)
        for slot, activation in zip(self._activation_slots, activations):
            args[slot] = activation
        for slot, name in self._parameter_slots:
            args[slot] = getattr(self, name)
        return self._call(*args)

    def activations(self, values: dict[str, Any]) -> list[torch.Tensor]:
        return [values[name] for name in self._activation_names]


def native_module(op: OpDecl, call, *, values: dict[str, Any],
                  adapter_kind: str) -> NativeModule:
    """Wrap a provider's own differentiable callable, keeping the declared split."""
    parameter_names = tuple(op.parameter_args or ())
    return NativeModule(
        op,
        call,
        parameters=values,
        parameter_names=parameter_names,
        activation_names=tuple(
            arg.name for arg in op.active_args()
            if arg.name not in parameter_names
        ),
        adapter_kind=adapter_kind,
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
    # One upstream gradient per declared output, in declared order.
    upstream = upstream_grad_values(op, workload_values)
    dys = upstream if isinstance(upstream, tuple) else (upstream,)
    y_ref, expected = oracle(op, workload_values)

    activations = [
        a.detach().clone().requires_grad_(True) for a in module.activations(workload_values)
    ]
    for parameter in module.parameters():
        parameter.grad = None
    y = module(*activations)
    outputs = as_output_tuple(op, y)
    # `torch.autograd.backward`, not `y.backward(dy)`: a multi-output module has
    # no single `y` to call it on, and every output has to contribute its own
    # gradient or the parameters would be checked against a partial backward.
    torch.autograd.backward(outputs, dys)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    checks: dict[str, Any] = {}
    # Each output under its own declared name, with its own tolerance -- a task
    # whose results have genuinely different magnitudes (a normalized value and
    # the plain sum it came from) is not gated by whichever one is largest.
    for out, actual, reference in zip(
        op.outputs, outputs, as_output_tuple(op, y_ref)
    ):
        atol, rtol = op.tolerance_for(workload, out.name)
        checks[out.name] = _compare(actual, reference, atol, rtol)

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
    """Compare one result against the oracle: shape, dtype, stride, finiteness, values.

    ``stride_match`` is measured and reported but does not gate ``ok``. The
    providers at this tier are different implementations of the same
    mathematics, and a compiled or fused one may legitimately hand back an
    equivalently-valued tensor under another layout; tier 1's gate does not
    require stride equality either, so requiring it here would fail operators
    tier 1 accepts for a property nothing downstream of this report reads. It
    is recorded because a layout change is worth seeing when it happens.
    """
    if not torch.is_tensor(actual):
        return {"ok": False, "reason": "no gradient produced"}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "ok": False,
            "reason": (
                f"shape/dtype {tuple(actual.shape)}/{actual.dtype} != "
                f"{tuple(expected.shape)}/{expected.dtype}"
            ),
            "stride": list(actual.stride()),
            "expected_stride": list(expected.stride()),
            "stride_match": tuple(actual.stride()) == tuple(expected.stride()),
        }
    difference = (actual.detach().float() - expected.detach().float()).abs()
    finite = bool(torch.isfinite(actual.detach()).all())
    return {
        "ok": bool(
            finite
            and torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
        ),
        "max_abs_error": float(difference.max()),
        "finite": finite,
        "stride": list(actual.stride()),
        "expected_stride": list(expected.stride()),
        "stride_match": tuple(actual.stride()) == tuple(expected.stride()),
    }


# ── timing ───────────────────────────────────────────────────────────────────


def _summarize_samples(samples: list[float]) -> dict[str, float]:
    """Median and the 20/80 quantiles of a fixed-size sample."""
    ordered = sorted(samples)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"median_ms": quantile(0.5), "q20_ms": quantile(0.2),
            "q80_ms": quantile(0.8), "samples": len(ordered)}


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
    reps: int = REPETITIONS,
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
    upstream = upstream_grad_values(op, values)
    dys = upstream if isinstance(upstream, tuple) else (upstream,)

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
        # Every declared output is produced and differentiated inside the timed
        # region; none can be left out of what is being measured.
        torch.autograd.backward(as_output_tuple(op, y), dys)

    # Ten explicit untimed iterations first: `do_bench`'s own warmup is a
    # duration, and Triton autotuning must be finished before it starts
    # estimating an iteration count from a timed sample.
    for _ in range(WARMUP_ITERS):
        full_step()
    for activation in activations:
        activation.grad = None
    torch.cuda.synchronize()

    # `do_bench` is the driver Liger's published benchmark uses, so the numbers
    # are commensurable with theirs: `rep` is a *millisecond budget*, L2 is
    # flushed between samples, and the reported statistic is the (0.5, 0.2, 0.8)
    # quantile triple rather than a mean.
    forward = _summarize(do_bench(
        forward_only, warmup=warmup_ms, rep=rep_ms, quantiles=list(QUANTILES)
    ))
    step = _summarize(do_bench(
        full_step, warmup=warmup_ms, rep=rep_ms,
        grad_to_none=activations, quantiles=list(QUANTILES),
    ))

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
    torch.autograd.backward(as_output_tuple(op, y), dys)
    torch.cuda.synchronize()
    peak_memory = int(torch.cuda.max_memory_allocated())

    return {
        "forward": forward,
        "full_step": step,
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
        return candidate_module(op, spec.candidate_module, values=values)
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
    order_seed: int = 0,
    only: str | None = None,
) -> dict[str, Any]:
    """One shape, every provider, correctness before timing."""
    values = build_parameters(op, workload, device=device)
    # Order is randomized and recorded. A fixed order lets slow drift -- clocks,
    # thermals, allocator state -- land on whichever provider is always last,
    # and the seed keeps the shuffle reproducible.
    ordered = list(specs)
    random.Random(order_seed).shuffle(ordered)
    upstream = upstream_grad_values(op, values)
    dys = upstream if isinstance(upstream, tuple) else (upstream,)
    providers: dict[str, Any] = {}
    for spec in ordered:
        if only is not None and spec.name != only:
            continue
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
        "provider_order": [spec.name for spec in ordered],
        "order_seed": order_seed,
        # Every provider is handed these same tensors; the fingerprints make
        # that checkable from the report instead of trusted.
        "input_checksums": {
            name: tensor_checksum(value) for name, value in sorted(values.items())
        },
        "upstream_grad_checksums": [tensor_checksum(d) for d in dys],
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
            "step": "y = model(*activations); torch.autograd.backward(y, output_grads)",
            "parameters": "built once and copied into every provider",
        },
        "environment": environment_fingerprint(),
        "cases": cases,
    }
