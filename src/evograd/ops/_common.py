"""Shared declaration helpers for the 2-D Liger-derived operator suite."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable

from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import AlphaFoldConfig, ModelConfig


STANDARD_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}
STANDARD_CORRECTNESS_SHAPES = {
    "float32": ((8, 64), (17, 128)),
    "float16": ((32, 256), (64, 512)),
    "bfloat16": ((32, 256), (64, 512)),
}


def workloads_2d(
    shapes: Iterable[tuple[int, int]],
    dtypes: Iterable[str],
    *,
    row_dim: str = "rows",
    col_dim: str = "cols",
    tolerances: dict[str, tuple[float, float]] | None = None,
) -> tuple[Workload, ...]:
    tolerances = tolerances or {}
    return tuple(
        Workload(
            dims={row_dim: rows, col_dim: cols},
            dtype=dtype,
            atol=tolerances.get(dtype, (None, None))[0],
            rtol=tolerances.get(dtype, (None, None))[1],
        )
        for rows, cols in shapes
        for dtype in dtypes
    )


def standard_correctness(
    *,
    dtypes: Iterable[str] = ("float32", "float16", "bfloat16"),
    row_dim: str = "rows",
    col_dim: str = "cols",
    tolerances: dict[str, tuple[float, float]] | None = None,
) -> tuple[Workload, ...]:
    tolerances = tolerances or STANDARD_TOLERANCES
    return tuple(
        Workload(
            dims={row_dim: rows, col_dim: cols},
            dtype=dtype,
            atol=tolerances[dtype][0],
            rtol=tolerances[dtype][1],
        )
        for dtype in dtypes
        for rows, cols in STANDARD_CORRECTNESS_SHAPES[dtype]
    )


def model_workloads(
    config: ModelConfig | AlphaFoldConfig,
    component: str,
    free_sweep: Iterable[dict[str, int]],
    dtypes: Iterable[str],
    *,
    tolerances: dict[str, tuple[float, float]] | None = None,
    scaled: bool = False,
    note: str = "",
) -> tuple[Workload, ...]:
    """Derive timed workloads from a model configuration, with provenance.

    ``component`` names a ``<component>_dims`` method on the config, and each
    entry of ``free_sweep`` supplies only what the configuration does not fix
    (token count, batch, crop length). The same pair is stored on the workload,
    so ``models.rederive_dims`` can reproduce the shape and the provenance test
    can prove the declaration has not drifted from the model it cites.

    Prefer this over writing shape tuples by hand. A literal ``(8192, 4096,
    14336)`` is indistinguishable from a guess; ``mlp_down_dims(tokens=8192)``
    on ``LLAMA_3_8B`` cannot silently stop being Llama-3's MLP.
    """
    builder = getattr(config, f"{component}_dims", None)
    if builder is None:
        raise AttributeError(
            f"{config.name}: no component {component!r} "
            f"(expected a {component}_dims method)"
        )
    tolerances = tolerances or {}
    workloads = []
    for free in free_sweep:
        dims = builder(**free)
        for dtype in dtypes:
            atol, rtol = tolerances.get(dtype, (None, None))
            workloads.append(
                Workload(
                    dims=dims,
                    dtype=dtype,
                    atol=atol,
                    rtol=rtol,
                    provenance=Provenance(
                        model=config.name,
                        component=component,
                        free=dict(free),
                        scaled=scaled,
                        note=note,
                    ),
                )
            )
    return tuple(workloads)


def fixed_shape_suites(
    workloads: Iterable[Workload], *, prefix: str = "fixed"
) -> dict[str, tuple[Workload, ...]]:
    """One single-case suite per workload, for the fixed-shape evaluation mode.

    Variable-shape evaluation asks for one deployable implementation across a
    whole grid; fixed-shape evaluation lets a candidate specialize hard for one
    configuration. The second mode needs a suite containing exactly one case,
    and ``EVOGRAD_BENCHMARK_SUITE`` already routes any named suite through
    evolution, so nothing downstream changes.

    This generalizes what ``layernorm`` did by hand with ``tb_i12``..``tb_i27``:
    those were bespoke names for the same idea, usable by exactly one operator.
    """
    suites: dict[str, tuple[Workload, ...]] = {}
    for workload in workloads:
        dims = "-".join(f"{name}{value}" for name, value in workload.dims.items())
        key = f"{prefix}/{dims}-{workload.dtype}"
        # A grid may legitimately repeat a shape across dtypes; the dtype is
        # already in the key, so a collision means a duplicated workload.
        if key in suites:
            continue
        suites[key] = (workload,)
    return suites


def regime_suites(
    workloads: Iterable[Workload],
    feature: Callable[[Workload], float],
    split: float,
) -> dict[str, tuple[Workload, ...]]:
    cases = tuple(workloads)
    return {
        "full": cases,
        "small": tuple(case for case in cases if feature(case) < split),
        "large": tuple(case for case in cases if feature(case) >= split),
    }


def log_distance_weight(feature: Callable[[Workload], float], split: float):
    def weight(workload: Workload) -> float:
        distance = abs(math.log2(max(feature(workload), 1.0)) - math.log2(split))
        return float(max(1.0, round(distance)))

    return weight


def dtype_for(torch, name: str):
    aliases = {"bf16": "bfloat16"}
    return getattr(torch, aliases.get(name, name))


def seed_2d(workload: Workload) -> int:
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    return rows * 100003 + cols


def recorded_layout(shape: Iterable[int], stride: Iterable[int]) -> str:
    """Classify one harvested tensor's memory layout from its recorded strides.

    A decoder reaches attention and RoPE by projecting into ``[B, T, H, D]`` and
    transposing, so what the kernel receives is a non-contiguous view of shape
    ``[B, H, T, D]`` whose strides are ``[T*H*D, D, H*D, 1]``. That signature is
    exact, so this reads the layout off the harvest rather than asking anyone to
    declare it.

    Anything else is reported as ``contiguous``: the layouts this distinguishes
    are the ones an input generator can actually reproduce, and a shape it does
    not recognise should fall back to the plain tensor rather than to a guess.
    """
    shape, stride = tuple(shape), tuple(stride)
    if len(shape) == 4 and len(stride) == 4:
        batch, heads, tokens, dim = shape
        if stride == (tokens * heads * dim, dim, heads * dim, 1) and heads > 1:
            return "head_major_view"
    return "contiguous"


def contiguous_stride(shape: Iterable[int]) -> tuple[int, ...]:
    """Row-major strides for ``shape``."""
    shape = tuple(int(v) for v in shape)
    stride, step = [], 1
    for size in reversed(shape):
        stride.append(step)
        step *= max(size, 1)
    return tuple(reversed(stride))


def view_stride(shape: Iterable[int], stride: Iterable[int],
                target: Iterable[int]) -> tuple[int, ...] | None:
    """The strides of ``reshape(target)`` when it is a view, else ``None``.

    A harvested tensor is often recorded at a higher rank than the operator
    declares -- ``[B, T, H, D]`` for a ``[rows, D]`` norm -- and the rank
    adapter flattens the leading dims with ``reshape``. That is a view exactly
    when each merged group of dims is itself contiguous; then this returns the
    strides the operator really receives. Pure Python, so declarations can call
    it without torch.
    """
    shape, stride, target = (tuple(int(v) for v in x) for x in (shape, stride, target))
    if shape == target:
        return stride
    total = 1
    for size in shape:
        total *= size
    goal = 1
    for size in target:
        goal *= size
    if total != goal:
        return None
    # Drop size-1 dims: their strides carry no information.
    dims = [(n, s) for n, s in zip(shape, stride) if n != 1]
    out: list[int] = []
    i = 0
    for size in target:
        if size == 1:
            out.append(0)
            continue
        if i >= len(dims):
            return None
        group_size, group_stride = dims[i]
        i += 1
        while group_size < size:
            if i >= len(dims):
                return None
            next_size, next_stride = dims[i]
            if group_stride != next_stride * next_size:
                return None
            group_size *= next_size
            group_stride = next_stride
            i += 1
        if group_size != size:
            return None
        out.append(group_stride)
    if i != len(dims):
        return None
    # A size-1 output dim's stride is free; use the row-major value.
    natural = contiguous_stride(target)
    return tuple(o if n != 1 else c for o, n, c in zip(out, target, natural))


def is_head_major_view(workload: Workload) -> bool:
    """Whether this workload's primary input is the non-contiguous view a model
    hands its attention and RoPE kernels.

    Input generators branch on this rather than on the model's name, so every
    harvested architecture reproduces the layout its own run recorded instead of
    falling back to a contiguous substitute at the right shape.
    """
    provenance = getattr(workload, "provenance", None)
    return provenance is not None and provenance.layout == "head_major_view"


def make_pair_baseline(
    factory: Callable[[], tuple[Callable, Callable]],
    forward_args: tuple[str, ...],
    backward_extras: tuple[str, ...] = (),
    *,
    sidecar: str | None = None,
):
    """Adapt a reviewed library autograd pair (Liger, FLA) to evograd's timing-hook API.

    ``sidecar`` names a library installed outside the environment
    (:mod:`evograd.opdecl.sidecar`): the harness then measures this baseline only
    in a child process started for it, and the factory is not called anywhere else.
    """

    def measure(torch, inputs, dout, warmup, reps, median_ms, median_ms_timed_region):
        forward, backward = factory()
        positional = [inputs[name] for name in forward_args]
        extras = [inputs[name] for name in backward_extras]

        def normalize(saved):
            return tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)

        def setup():
            _y, saved = forward(*positional)
            return normalize(saved)

        def run_backward(saved):
            return backward(dout, saved, *extras)

        def full_step():
            return run_backward(setup())

        backward_ms = median_ms_timed_region(setup, run_backward, warmup, reps)
        full_ms = median_ms(full_step, warmup, reps)
        return backward_ms, full_ms

    measure.__name__ = f"measure_{factory.__name__}"
    measure.available = lambda: baseline_available(factory)
    measure.pair_factory = factory
    measure.forward_args = tuple(forward_args)
    measure.backward_extras = tuple(backward_extras)
    measure.sidecar = sidecar
    return measure


class FunctionContext:
    """A stand-in ``ctx`` for calling a library's ``torch.autograd.Function``
    ``forward``/``backward`` directly, so the pair runs exactly the library's
    code with its saved state held in one object."""

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors):
        return None

    def set_materialize_grads(self, value):
        self.materialize_grads = value


def baseline_available(factory: Callable) -> bool:
    """Cheaply probe a Liger factory without importing Liger at declaration time."""
    try:
        factory()
        return True
    except Exception:
        return False


def accepted_kwargs(fn: Callable, values: dict) -> dict:
    """Return the subset of values accepted by a baseline/candidate callable."""
    parameters = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return values
    return {name: value for name, value in values.items() if name in parameters}


def head_major_randn(torch, batch: int, tokens: int, heads: int, dim: int, *, device, dtype):
    """A ``[B, heads, T, D]`` tensor with the strides a decoder hands attention.

    Built token-major and transposed, which is how a model produces q, k and v
    (``proj(x).view(B, T, heads, D).transpose(1, 2)``), so a primitive derived
    from the attention decomposition sees the non-contiguous head-major view
    the real boundary presents rather than a contiguous substitute.
    """
    token_major = torch.randn((batch, tokens, heads, dim), device=device, dtype=dtype)
    return token_major.transpose(1, 2)
