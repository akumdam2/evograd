"""**Block scope** -- one architectural block, forward plus a supplied-cotangent VJP.

Tier 3's model scope times a whole training step. This is the other execution
scope: one block of a real architecture, with candidates installed at its
declared sites, run as ``outputs = block(*args, **kwargs)`` followed by
``torch.autograd.backward(outputs, cotangents)``. No loss is fabricated, no
optimizer steps, and nothing above the block is built.

What is here is architecture-independent and says so by construction: it
imports no model package, never selects ``result[0]`` for a caller, fixes no
site name or call count, and reads every model fact -- how to build the block,
how to call it, what its sites are, how many times each runs, which
parameters exist -- off a :class:`BlockAdapter` the model package supplies.

    block.py        this file -- invocation, VJP execution, timing, memory,
                    provider orchestration and the block report
    gate/block.py   the architecture-neutral correctness gate and its policy
    workloads/<model>/block.py   the model's adapter

Three rules the model scope did not have to state:

* **Every declared output gets its cotangent and every declared differentiable
  input gets a gradient.** The invocation names both by path; an output the
  block returns that no path declares is an error, not something to ignore.
* **Reset before every independent repetition, warmup included, outside the
  timed region.** Gradients to ``None``, fresh leaf inputs, the adapter's
  counters and state cleared -- then the timer starts.
* **The reference lives on the CPU and is compared incrementally.** A candidate's
  tensors are moved one at a time; no run keeps a growing list of full GPU
  gradient sets.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

import torch

from evograd.evaluation.tier3.patch import KernelSet, PatchProvenance, SiteRegistry

#: The protocol string block reports carry. The reader in
#: :mod:`evograd.evaluation.tier3.report` already knows it.
TIER3_BLOCK_PROTOCOL_VERSION = "evograd-tier3-block-v1"
EXECUTION_SCOPE_BLOCK = "block"
TIMING_BOUNDARY = "forward + vjp"

#: The unpatched block. Speedups are measured against it.
REFERENCE_PROVIDER = "native"


class BlockError(RuntimeError):
    """A block provider cannot be measured. Marks that provider failed, not the run."""


class InvocationError(BlockError):
    """The block's call or return does not match the declared invocation."""


class BlockCorrectnessFailure(BlockError):
    """The block disagreed with its reference. Never timed."""


class PatchCoverageFailure(BlockError):
    """The installed sites did not run the declared number of times."""


class NoPolicyForProvider(BlockError):
    """The frozen policy has no entry for this provider's patch set."""


# ── tensor paths ─────────────────────────────────────────────────────────────

_PATH_TOKEN = re.compile(r"\.(\w+)|\[(\d+)\]")


def parse_path(path: str) -> tuple[str, tuple[Any, ...]]:
    """``"args[0]"`` -> ``("args", (0,))``; ``"kwargs.position_embeddings[1]"``
    -> ``("kwargs", ("position_embeddings", 1))``; ``"result"`` -> ``("result", ())``."""
    match = re.match(r"^(\w+)", path)
    if not match:
        raise InvocationError(f"malformed tensor path {path!r}")
    root, rest = match.group(1), path[match.end():]
    steps: list[Any] = []
    position = 0
    for token in _PATH_TOKEN.finditer(rest):
        if token.start() != position:
            raise InvocationError(f"malformed tensor path {path!r}")
        steps.append(token.group(1) if token.group(1) is not None else int(token.group(2)))
        position = token.end()
    if position != len(rest):
        raise InvocationError(f"malformed tensor path {path!r}")
    return root, tuple(steps)


def resolve_path(roots: Mapping[str, Any], path: str) -> Any:
    root, steps = parse_path(path)
    if root not in roots:
        raise InvocationError(f"{path}: no such root {root!r}; have {sorted(roots)}")
    value = roots[root]
    for step in steps:
        try:
            value = value[step]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvocationError(f"{path}: cannot take {step!r}: {exc}") from None
    return value


def _replace_at(value: Any, steps: tuple[Any, ...], new: Any) -> Any:
    if not steps:
        return new
    head, tail = steps[0], steps[1:]
    if isinstance(value, tuple):
        items = list(value)
        items[head] = _replace_at(items[head], tail, new)
        return tuple(items)
    if isinstance(value, list):
        items = list(value)
        items[head] = _replace_at(items[head], tail, new)
        return items
    if isinstance(value, dict):
        items = dict(value)
        items[head] = _replace_at(items[head], tail, new)
        return items
    raise InvocationError(f"cannot index a {type(value).__name__} with {head!r}")


def tensor_paths(value: Any, prefix: str) -> list[str]:
    """Every tensor inside ``value``, by path. Containers are walked; opaque
    objects are reported as their own path so they cannot hide a tensor."""
    if torch.is_tensor(value):
        return [prefix]
    if isinstance(value, (tuple, list)):
        return [p for i, v in enumerate(value) for p in tensor_paths(v, f"{prefix}[{i}]")]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in tensor_paths(v, f"{prefix}.{k}")]
    return []


def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "device": tensor.device.type,
    }


def describe_tree(value: Any) -> Any:
    """A JSON-safe description of an argument tree: tensors by metadata,
    scalars by value, containers by structure."""
    if torch.is_tensor(value):
        return {"kind": "tensor", **tensor_meta(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (tuple, list)):
        return {"kind": "sequence", "items": [describe_tree(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "mapping", "items": {k: describe_tree(v) for k, v in value.items()}}
    return {"kind": type(value).__name__}


def tensor_tree_hash(value: Any, *, label: str = "") -> str:
    """A content hash over every tensor and scalar in a tree, in a fixed walk.

    Tensor bytes are hashed as stored (a bfloat16 tensor by its 16-bit words),
    so two trees hash alike exactly when their numbers are identical; layout
    metadata (shape, dtype, stride) is hashed too, so a transposed copy of the
    same numbers does not.
    """
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))

    def feed(item: Any, path: str) -> None:
        hasher.update(path.encode("utf-8"))
        if torch.is_tensor(item):
            t = item.detach().to("cpu")
            hasher.update(json.dumps(tensor_meta(t), sort_keys=True).encode("utf-8"))
            hasher.update(t.contiguous().view(torch.uint8).numpy().tobytes()
                          if t.dtype != torch.bool else t.numpy().tobytes())
        elif isinstance(item, (tuple, list)):
            hasher.update(f"seq{len(item)}".encode("utf-8"))
            for i, v in enumerate(item):
                feed(v, f"{path}[{i}]")
        elif isinstance(item, dict):
            hasher.update(f"map{len(item)}".encode("utf-8"))
            for k in sorted(item):
                feed(item[k], f"{path}.{k}")
        else:
            hasher.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))

    feed(value, "$")
    return hasher.hexdigest()


# ── the invocation ───────────────────────────────────────────────────────────


@dataclass
class BlockInvocation:
    """One structured call of the block, with every differentiable path named.

    ``outputs`` are paths into the block's return value (``"result"`` for a bare
    tensor, ``"result[0]"`` for the first element of a tuple -- stated, never
    assumed); ``cotangents`` line up with them one to one. ``metadata_outputs``
    are non-tensor parts of the return compared for equality.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    differentiable_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cotangents: tuple[torch.Tensor, ...]
    metadata_outputs: tuple[str, ...] = ()
    #: Where the tensors came from: ``"captured"`` or ``"config"``. Carried so a
    #: report can say which claim its numbers support.
    source_mode: str = "unspecified"

    def __post_init__(self) -> None:
        if len(self.outputs) != len(self.cotangents):
            raise InvocationError(
                f"{len(self.outputs)} declared outputs but {len(self.cotangents)} "
                "cotangents; every declared output needs exactly one"
            )
        if not self.outputs:
            raise InvocationError("an invocation must declare at least one output")
        roots = {"args": self.args, "kwargs": self.kwargs}
        for path in self.differentiable_inputs:
            value = resolve_path(roots, path)
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                raise InvocationError(f"{path}: a differentiable input must be a floating tensor")
        declared_inputs = set(tensor_paths(self.args, "args")) | set(tensor_paths(self.kwargs, "kwargs"))
        unknown = set(self.differentiable_inputs) - declared_inputs
        if unknown:
            raise InvocationError(f"differentiable inputs not present in the call: {sorted(unknown)}")

    def fresh(self) -> "BlockInvocation":
        """Every differentiable input as a new leaf; everything else shared.

        ``clone()`` preserves the strides of a dense tensor, so a captured
        layout survives; the leaf is detached from any earlier graph, so one
        repetition cannot reach into another's.
        """
        args, kwargs = self.args, self.kwargs
        roots = {"args": args, "kwargs": kwargs}
        for path in self.differentiable_inputs:
            root, steps = parse_path(path)
            value = resolve_path(roots, path)
            leaf = value.detach().clone().requires_grad_(True)
            if root == "args":
                args = _replace_at(args, steps, leaf)
            else:
                kwargs = _replace_at(kwargs, steps, leaf)
            roots = {"args": args, "kwargs": kwargs}
        return replace(self, args=args, kwargs=kwargs)

    def leaves(self) -> dict[str, torch.Tensor]:
        roots = {"args": self.args, "kwargs": self.kwargs}
        return {path: resolve_path(roots, path) for path in self.differentiable_inputs}

    def select_outputs(self, result: Any) -> tuple[torch.Tensor, ...]:
        """The declared outputs of ``result``, and nothing implicit.

        Every tensor the block returned must be a declared output or a declared
        metadata output: a tensor the declaration does not know about would
        otherwise be silently dropped from the backward.
        """
        roots = {"result": result}
        selected = []
        for path in self.outputs:
            value = resolve_path(roots, path)
            if not torch.is_tensor(value):
                raise InvocationError(f"{path}: declared output is not a tensor")
            selected.append(value)
        returned = set(tensor_paths(result, "result"))
        declared = set(self.outputs) | set(self.metadata_outputs)
        undeclared = sorted(returned - declared)
        if undeclared:
            raise InvocationError(
                f"the block returned tensors no path declares: {undeclared}; "
                f"declared outputs are {list(self.outputs)}"
            )
        return tuple(selected)

    def metadata(self, result: Any) -> dict[str, Any]:
        roots = {"result": result}
        return {path: resolve_path(roots, path) for path in self.metadata_outputs}

    def describe(self) -> dict[str, Any]:
        return {
            "args": describe_tree(self.args),
            "kwargs": describe_tree(self.kwargs),
            "differentiable_inputs": list(self.differentiable_inputs),
            "outputs": list(self.outputs),
            "metadata_outputs": list(self.metadata_outputs),
            "cotangents": [tensor_meta(c) for c in self.cotangents],
            "source_mode": self.source_mode,
        }

    def to(self, device: str) -> "BlockInvocation":
        def move(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value

        return replace(self, args=move(self.args), kwargs=move(self.kwargs),
                       cotangents=tuple(move(c) for c in self.cotangents))


# ── the case and the adapter ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockCase:
    """One block plus one input recipe, with its identity spelled out."""

    architecture: str
    architecture_revision: str
    block_kind: str
    block_index: int
    source_mode: str                    # "captured" | "config"
    source: dict[str, Any]              # artifact identity, or the recipe
    dims: dict[str, int]
    dtype: str
    boundary: str
    sites: dict[str, str]               # site -> declared task
    excluded: tuple[str, ...] = ()
    state_contract: dict[str, Any] = field(default_factory=dict)
    weights_hash: str | None = None
    inputs_hash: str | None = None
    cotangents_hash: str | None = None

    @property
    def case_id(self) -> str:
        dims = ".".join(f"{k}{v}" for k, v in sorted(self.dims.items()) if k in ("B", "T"))
        return (f"{self.architecture}/{self.block_kind}@{self.block_index}/"
                f"{self.source_mode}/{dims}.{self.dtype}")

    @property
    def case_hash(self) -> str:
        identity = {
            "architecture": self.architecture, "revision": self.architecture_revision,
            "block_kind": self.block_kind, "block_index": self.block_index,
            "source_mode": self.source_mode, "source": self.source,
            "dims": self.dims, "dtype": self.dtype, "sites": self.sites,
            "weights_hash": self.weights_hash, "inputs_hash": self.inputs_hash,
            "cotangents_hash": self.cotangents_hash,
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "architecture_revision": self.architecture_revision,
            "block_kind": self.block_kind,
            "block_index": self.block_index,
            "case_id": self.case_id,
            "case_hash": self.case_hash,
            "source_mode": self.source_mode,
            "source": dict(self.source),
            "weights_hash": self.weights_hash,
            "inputs_hash": self.inputs_hash,
            "cotangents_hash": self.cotangents_hash,
            "dims": dict(self.dims),
            "dtype": self.dtype,
            "boundary": self.boundary,
            "sites": dict(self.sites),
            "excluded": list(self.excluded),
            "state_contract": dict(self.state_contract),
        }


@dataclass
class BuiltBlock:
    module: torch.nn.Module
    #: state-dict order, which install() must not change.
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Installed:
    provenance: PatchProvenance
    #: ``.snapshot() -> {site: count}`` and ``.reset()``.
    counters: Any


class BlockAdapter(Protocol):
    """What the executor asks a model package for, and nothing more."""

    case: BlockCase
    registry: SiteRegistry
    #: Benchmark suite carrying this model's observed shapes, for purity.
    observed_suite: str | None

    def build(self, *, device: str) -> BuiltBlock: ...
    def prepare(self, built: BuiltBlock, *, device: str) -> BlockInvocation: ...
    def install(self, built: BuiltBlock, kernels: KernelSet) -> Installed: ...
    def expected_invocations(self, kernels: KernelSet) -> dict[str, int]: ...
    def reset(self, built: BuiltBlock) -> None: ...
    def local_checks(self, built: BuiltBlock, kernels: KernelSet,
                     invocation: BlockInvocation) -> dict[str, Any] | None: ...
    def capture_reference(self) -> dict[str, Any] | None: ...
    def controls(self, kernels: KernelSet, ops: Mapping[str, Any],
                 names: tuple[str, ...]) -> dict[str, KernelSet]: ...


# ── execution ────────────────────────────────────────────────────────────────


@dataclass
class BlockResult:
    """What one forward + VJP produced, detached. Where it lives is the caller's
    choice: :meth:`cpu` moves it off the accelerator."""

    outputs: dict[str, torch.Tensor]
    input_grads: dict[str, torch.Tensor | None]
    param_grads: dict[str, torch.Tensor | None]
    metadata: dict[str, Any]
    structure: Any
    #: Input tensor paths the block wrote into during the repetition.
    mutated_inputs: list[str] = field(default_factory=list)

    def cpu(self) -> "BlockResult":
        move = lambda t: None if t is None else t.detach().to("cpu")  # noqa: E731
        return BlockResult(
            outputs={k: move(v) for k, v in self.outputs.items()},
            input_grads={k: move(v) for k, v in self.input_grads.items()},
            param_grads={k: move(v) for k, v in self.param_grads.items()},
            metadata=dict(self.metadata), structure=self.structure,
            mutated_inputs=list(self.mutated_inputs),
        )


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def vjp_step(module: torch.nn.Module, invocation: BlockInvocation) -> Any:
    """THE timed region: forward, then backward from the supplied cotangents.

    ``invocation`` must already be fresh (see :meth:`BlockInvocation.fresh`):
    the executor makes the leaves outside the timer, so what is timed is the
    block's own arithmetic and autograd's, not tensor allocation for the test.
    """
    result = module(*invocation.args, **invocation.kwargs)
    outputs = invocation.select_outputs(result)
    torch.autograd.backward(outputs, invocation.cotangents)
    return result


def run_vjp(module: torch.nn.Module, invocation: BlockInvocation,
            parameters: Mapping[str, torch.Tensor], *, reset: Callable[[], None]) -> BlockResult:
    """One independent repetition: reset, fresh leaves, forward + VJP, collect.

    Gradients are collected for every parameter the block declares, present or
    not -- a missing one is ``None`` and the gate decides what that means.
    """
    reset()
    fresh = invocation.fresh()
    # Every input tensor, by content, before the block sees it. A kernel that
    # writes into an activation or a rotary table changes what the *next*
    # repetition computes, and nothing downstream would say why.
    roots = {"args": fresh.args, "kwargs": fresh.kwargs}
    before = {path: resolve_path(roots, path).detach().clone()
              for path in (*tensor_paths(fresh.args, "args"), *tensor_paths(fresh.kwargs, "kwargs"))}
    result = vjp_step(module, fresh)
    outputs = fresh.select_outputs(result)
    mutated = [path for path, snapshot in before.items()
               if not torch.equal(resolve_path(roots, path).detach(), snapshot)]
    del before
    return BlockResult(
        mutated_inputs=mutated,
        outputs={path: out.detach() for path, out in zip(fresh.outputs, outputs)},
        input_grads={path: (leaf.grad.detach() if leaf.grad is not None else None)
                     for path, leaf in fresh.leaves().items()},
        param_grads={name: (p.grad.detach() if p.grad is not None else None)
                     for name, p in parameters.items()},
        metadata={k: (v.detach().to("cpu") if torch.is_tensor(v) else v)
                  for k, v in fresh.metadata(result).items()},
        structure=describe_tree(result),
    )


def time_vjp(module: torch.nn.Module, invocation: BlockInvocation, *,
             reset: Callable[[], None], device: str, warmup: int, samples: int,
             blocks: int) -> dict[str, Any]:
    """Warmed, repeated forward + VJP; reset before every sample, outside the timer.

    Each sample is one independent repetition: gradients cleared, fresh leaf
    inputs made, the adapter's state and counters reset, the device drained,
    *then* the timer starts. The timed region is exactly :func:`vjp_step`.
    On CUDA the region is bracketed by events; on CPU by ``perf_counter``.
    """
    if samples < 1 or blocks < 1:
        raise ValueError("samples and blocks must both be >= 1")
    use_events = str(device).startswith("cuda") and torch.cuda.is_available()

    def one_sample() -> float:
        reset()
        fresh = invocation.fresh()
        _sync(device)
        if use_events:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            vjp_step(module, fresh)
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end))
        started = time.perf_counter()
        vjp_step(module, fresh)
        _sync(device)
        return (time.perf_counter() - started) * 1e3

    for _ in range(warmup):
        one_sample()
    per_block: list[float] = []
    per_sample: list[list[float]] = []
    for _ in range(blocks):
        times = [one_sample() for _ in range(samples)]
        per_sample.append(times)
        per_block.append(statistics.median(times))
    flat = [t for block in per_sample for t in block]
    return {
        "forward_backward_ms": statistics.median(per_block),
        "forward_ms": None,
        "backward_ms": None,
        "per_block_ms": per_block,
        "per_sample_ms": per_sample,
        "min_ms": min(flat),
        "samples": samples,
        "blocks": blocks,
        "warmup": warmup,
        "timer": "cuda events" if use_events else "perf_counter with device sync",
        "reset": "gradients to None, fresh leaf inputs, adapter state and counters "
                 "cleared before every sample including warmup; outside the timed region",
    }


def memory_probe(module: torch.nn.Module, invocation: BlockInvocation, *,
                 reset: Callable[[], None], device: str) -> int | None:
    """Peak allocation over one settled forward + VJP, with nothing else attached."""
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None
    reset()
    fresh = invocation.fresh()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    vjp_step(module, fresh)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def saved_state_probe(module: torch.nn.Module, invocation: BlockInvocation,
                      parameters: Mapping[str, torch.Tensor],
                      buffers: Mapping[str, torch.Tensor], *,
                      reset: Callable[[], None]) -> dict[str, Any]:
    """What one forward saves for its backward, by unique storage.

    Observed through ``saved_tensors_hooks``: every tensor autograd stashes is
    seen once by the pack hook. Storages already owned by a parameter or buffer
    are excluded (they exist whether or not backward needs them), aliases of one
    storage are counted once, and the logical request count is reported beside
    the byte total. Not observable inside a compiled region, which is why the
    result carries ``method`` and a caller reports ``null`` where it cannot see.
    """
    owned = {t.untyped_storage().data_ptr() for t in (*parameters.values(), *buffers.values())}
    seen: dict[int, int] = {}
    logical = {"n": 0}

    def pack(tensor: torch.Tensor):
        logical["n"] += 1
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in owned:
            seen[storage.data_ptr()] = int(storage.nbytes())
        return tensor

    def unpack(tensor: torch.Tensor):
        return tensor

    reset()
    fresh = invocation.fresh()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = module(*fresh.args, **fresh.kwargs)
    outputs = fresh.select_outputs(result)
    torch.autograd.backward(outputs, fresh.cotangents)
    del result, outputs
    return {
        "saved_state_bytes": int(sum(seen.values())),
        "saved_unique_storages": len(seen),
        "saved_tensors_logical": logical["n"],
        "method": "torch.autograd.graph.saved_tensors_hooks; unique storages not owned "
                  "by a parameter or buffer",
    }


# ── one provider ─────────────────────────────────────────────────────────────


def _reset_fn(adapter: BlockAdapter, built: BuiltBlock, installed: Installed | None):
    def reset() -> None:
        built.module.zero_grad(set_to_none=True)
        adapter.reset(built)
        if installed is not None:
            installed.counters.reset()
    return reset


def native_reference(adapter: BlockAdapter, *, device: str) -> tuple[BlockResult, BuiltBlock, BlockInvocation]:
    """The unpatched block's result for this case, kept on the CPU."""
    built = adapter.build(device=device)
    invocation = adapter.prepare(built, device=device)
    result = run_vjp(built.module, invocation, built.parameters,
                     reset=_reset_fn(adapter, built, None)).cpu()
    return result, built, invocation


def measure_block_provider(
    adapter: BlockAdapter,
    name: str,
    kernels: KernelSet,
    *,
    role: str,
    device: str,
    ops: Mapping[str, Any] | None,
    policy: dict[str, Any] | None,
    reference: BlockResult | None,
    verify: bool = True,
    purity: bool = True,
    noise_repeats: int = 3,
    warmup: int = 3,
    samples: int = 10,
    blocks: int = 3,
) -> dict[str, Any]:
    """Correctness first, then timing, for one provider of one case.

    Stages, each recorded under ``correctness`` with its own verdict:
    ``site_preflight`` (the tier-1 gate on each patched kernel),
    ``provider_purity``, ``live_boundary`` (the adapter's per-invocation
    output/gradient shadow, when it offers one), ``patch_coverage`` (actual
    site counts against the declared ones), then ``block`` -- every declared
    output, input gradient and parameter gradient against the native reference
    under the frozen policy, plus finiteness, gradient presence, parameter
    aliasing and input mutation. Only after all of that: warmed timing, peak
    memory and saved state. A failure names its stage in ``failed_at`` and
    nothing is timed.
    """
    from evograd.evaluation.tier3.gate import block as block_gate

    entry: dict[str, Any] = {
        "provider": name, "role": role, "patched": list(kernels.patched),
        "kernel_sources": [s.to_dict() for s in kernels.sources],
        "correctness": {"ok": False, "stages": []},
        "latency": None, "memory": None,
    }
    correctness = entry["correctness"]

    def stage(label: str, verdict: dict[str, Any]) -> None:
        correctness[label] = verdict
        correctness["stages"].append(label)

    built = adapter.build(device=device)
    installed = adapter.install(built, kernels)
    entry["patch_coverage"] = {
        "requested": list(installed.provenance.requested_sites),
        "actual": list(installed.provenance.actual_sites),
        "expected_invocations": adapter.expected_invocations(kernels),
        "provenance": installed.provenance.to_dict(),
    }
    invocation = adapter.prepare(built, device=device)
    entry["invocation"] = invocation.describe()
    reset = _reset_fn(adapter, built, installed)

    # 1. site preflight: the declaration's oracle, at its grid and the observed shapes.
    if kernels.patched and verify:
        from evograd.evaluation.tier3.runner import preflight
        stage("site_preflight", preflight(kernels, dict(ops or {}), device=device))
    else:
        stage("site_preflight", {"gate": "skipped" if kernels.patched else "unpatched", "ok": True})

    # 2. purity: a function of its arguments, at production width.
    if kernels.patched and purity and adapter.observed_suite:
        from evograd.evaluation.tier3.gate import purity as purity_gate
        expected = adapter.expected_invocations(kernels)
        report = purity_gate.check_kernels(
            kernels, suite=adapter.observed_suite, device=device,
            calls={site: max(8, 4 * n) for site, n in expected.items()},
        )
        stage("provider_purity", {"ok": bool(report.get("ok")), "sites": [
            {k: v for k, v in s.items() if k not in ("compared", "checkpoints")}
            for s in report.get("sites", [])]})
        if not report.get("ok"):
            raise BlockCorrectnessFailure("provider is not a function of its arguments")
    else:
        stage("provider_purity", {"ok": True, "skipped": True,
                                  "reason": "unpatched" if not kernels.patched else "not requested"})

    # 3. the adapter's own per-invocation shadow, on the block's live tensors.
    local = adapter.local_checks(built, kernels, invocation) if kernels.patched else None
    if local is not None:
        stage("live_boundary", local)
        if not local.get("ok", False):
            raise BlockCorrectnessFailure(f"live boundary: {local.get('reason', 'failed')}")
    else:
        stage("live_boundary", {"ok": True, "skipped": True})

    # 4. one untimed repetition: counts and the candidate's own result.
    result = run_vjp(built.module, invocation, built.parameters, reset=reset)
    observed = installed.counters.snapshot()
    expected = adapter.expected_invocations(kernels)
    coverage_ok = observed == expected
    stage("patch_coverage", {"ok": coverage_ok, "observed": observed, "expected": expected})
    entry["patch_coverage"]["invocations"] = observed
    if not coverage_ok:
        raise PatchCoverageFailure(f"observed site invocations {observed} != declared {expected}")

    # 5. the block against its reference.
    if role == "reference":
        verdict = block_gate.check_reference(
            adapter, built, invocation, result, reset=reset,
            noise_repeats=noise_repeats, capture=adapter.capture_reference(),
            run=lambda: run_vjp(built.module, invocation, built.parameters, reset=reset),
        )
        stage("block", verdict)
        if not verdict["ok"]:
            raise BlockCorrectnessFailure(f"native reference: {verdict.get('reason')}")
    else:
        if reference is None:
            raise BlockError("no native reference to compare against")
        verdict = block_gate.check_candidate(
            reference, result, invocation=invocation, parameters=built.parameters,
            kernels=kernels, policy=policy, adapter=adapter,
        )
        stage("block", verdict)
        if not verdict["ok"]:
            from evograd.evaluation.tier3.gate import enforcement as enf

            kind = block_gate.finding_kind(verdict)
            verdict["enforcement"] = {"mode": enf.active_mode(), "finding_kind": kind}
            verdict["numerical_ok"] = False if kind == enf.NUMERICAL else None
            if enf.blocks(kind, enf.active_mode()):
                if verdict.get("failed_at") == "no_policy":
                    raise NoPolicyForProvider(str(verdict.get("reason")))
                raise BlockCorrectnessFailure(str(verdict.get("reason")))
            # report-first: the numerical verdict stands and the block is timed
            # anyway, so a number exists beside the disagreement.
            correctness["numerical_ok"] = False
            correctness["numerical_reason"] = str(verdict.get("reason"))
    del result
    correctness["ok"] = True

    # 6. timing and memory, on a block that has earned it.
    entry["latency"] = time_vjp(built.module, invocation, reset=reset, device=device,
                                warmup=warmup, samples=samples, blocks=blocks)
    saved = saved_state_probe(built.module, invocation, built.parameters,
                              built.buffers, reset=reset)
    entry["memory"] = {
        "execution_peak_bytes": memory_probe(built.module, invocation, reset=reset, device=device),
        "saved_state_bytes": saved["saved_state_bytes"],
        "saved_state_detail": saved,
        # The gate's own peak is a property of the validation process and is
        # recorded by the driver, which sees the whole child.
        "validation_peak_bytes": None,
    }
    reset()
    return entry


def measure_block_one(adapter: BlockAdapter, name: str, kernels: KernelSet,
                      **options: Any) -> dict[str, Any]:
    """One provider, with its failure captured rather than raised."""
    peak_before = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        entry = {"ok": True, **measure_block_provider(adapter, name, kernels, **options)}
    except Exception as exc:  # one provider must not take the run down
        entry = {"ok": False, "provider": name, "role": options.get("role"),
                 "patched": list(kernels.patched),
                 "kernel_sources": [s.to_dict() for s in kernels.sources],
                 "error": f"{type(exc).__name__}: {exc}",
                 "failed_at": _failure_stage(exc), "latency": None, "memory": None}
    finally:
        if torch.cuda.is_available():
            peak_before = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
    if entry.get("memory") is not None:
        entry["memory"]["validation_peak_bytes"] = peak_before
    return entry


def _failure_stage(exc: Exception) -> str:
    from evograd.evaluation.tier3.runner import PreflightFailure

    if isinstance(exc, NoPolicyForProvider):
        return "no_policy"
    if isinstance(exc, PatchCoverageFailure):
        return "patch_coverage"
    if isinstance(exc, BlockCorrectnessFailure):
        return "block_correctness"
    if isinstance(exc, InvocationError):
        return "invocation"
    if isinstance(exc, PreflightFailure):
        return "preflight"
    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return "out_of_memory"
    return "measurement"


# ── the report ───────────────────────────────────────────────────────────────


def _bootstrap(reference_blocks, candidate_blocks, *, seed):
    from evograd.evaluation.tier3.runner import _bootstrap_ratio

    return _bootstrap_ratio(reference_blocks, candidate_blocks, seed=seed)


def block_speedup_intervals(providers: dict[str, Any], *, seed: int) -> dict[str, Any]:
    base = providers.get(REFERENCE_PROVIDER)
    if not base or not base.get("ok"):
        return {"reference": REFERENCE_PROVIDER, "available": False}
    out: dict[str, Any] = {"reference": REFERENCE_PROVIDER, "available": True, "vs_reference": {}}
    ref = base["latency"]
    for index, (name, entry) in enumerate(sorted(providers.items())):
        if name == REFERENCE_PROVIDER or not entry.get("ok"):
            continue
        lat = entry["latency"]
        out["vs_reference"][name] = {
            "ratio": ref["forward_backward_ms"] / lat["forward_backward_ms"],
            "ci95": _bootstrap(ref["per_block_ms"], lat["per_block_ms"], seed=seed + index),
        }
    return out


def assemble_block_report(
    adapter: BlockAdapter,
    results: dict[str, Any],
    order: list[str],
    *,
    policy: dict[str, Any] | None,
    ops: Mapping[str, Any] | None,
    seed: int,
    isolation: str,
    warmup: int,
    samples: int,
    blocks: int,
    verify: bool,
    purity: bool,
) -> dict[str, Any]:
    """The block report, in the format :mod:`evograd.evaluation.tier3.report` reads."""
    from evograd.evaluation.tier3.runner import _environment

    def _enforcement_note():
        from evograd.evaluation.tier3.gate import enforcement as enf

        mode = enf.active_mode()
        return {"mode": mode, "rule": enf.DESCRIPTION[mode]}

    levels: set[int] = set()
    for entry in results.values():
        if not entry.get("ok"):
            continue
        for source in entry.get("kernel_sources", []):
            op = (ops or {}).get(source.get("op"))
            level = getattr(op, "level", None)
            if level is not None:
                levels.add(int(level))
    report = {
        "protocol": TIER3_BLOCK_PROTOCOL_VERSION,
        "evaluation_tier": 3,
        "execution_scope": EXECUTION_SCOPE_BLOCK,
        "benchmark_level": 3,
        "candidate_task_levels": sorted(levels),
        "case": adapter.case.to_dict(),
        "policy": policy if policy is not None else {"available": False},
        "timing_protocol": {
            "boundary": TIMING_BOUNDARY,
            "step": "result = block(*args, **kwargs); torch.autograd.backward("
                    "select_declared_outputs(result), cotangents)",
            "cotangent_source": adapter.case.source_mode,
            "warmup": warmup, "samples_per_block": samples, "blocks": blocks,
            "reset": "gradients to None, fresh leaf inputs, adapter state and counters "
                     "cleared before every sample including warmup; outside the timed region",
            "l2_policy": "never flushed inside a sample",
            "verification": {"site_preflight": verify, "purity": purity},
            "reference": REFERENCE_PROVIDER,
        },
        "seed": seed,
        "provider_order": list(order),
        "isolation": isolation,
        "environment": _environment(),
        "numerical_enforcement": _enforcement_note(),
        "providers": results,
    }
    report["speedup_intervals"] = block_speedup_intervals(results, seed=seed)
    return report


__all__ = [
    "EXECUTION_SCOPE_BLOCK",
    "REFERENCE_PROVIDER",
    "TIER3_BLOCK_PROTOCOL_VERSION",
    "TIMING_BOUNDARY",
    "BlockAdapter",
    "BlockCase",
    "BlockCorrectnessFailure",
    "BlockError",
    "BlockInvocation",
    "BlockResult",
    "BuiltBlock",
    "Installed",
    "InvocationError",
    "NoPolicyForProvider",
    "PatchCoverageFailure",
    "assemble_block_report",
    "block_speedup_intervals",
    "describe_tree",
    "measure_block_one",
    "measure_block_provider",
    "memory_probe",
    "native_reference",
    "parse_path",
    "resolve_path",
    "run_vjp",
    "saved_state_probe",
    "tensor_paths",
    "tensor_tree_hash",
    "time_vjp",
    "vjp_step",
]
