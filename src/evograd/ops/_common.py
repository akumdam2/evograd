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


def observed_workloads(
    workload_name: str,
    task: str,
    *,
    tolerances: dict[str, tuple[float, float]] | None = None,
    dtype: str | None = None,
) -> tuple[Workload, ...]:
    """Every observed configuration of one generic Level-1 task, for one workload.

    Read from that workload's tracked snapshot, which derives them from its
    Level-4 harvest: dims, dtype, roles, frequency and memory layout all come
    from what the canonical step ran. Nothing here is a second hand-written
    copy, and the provenance each workload carries re-derives the same dims from
    the published model configuration, so ``tests/test_provenance`` proves the
    two agree.

    ``workload_name`` selects the snapshot; it is also the provenance's model
    key, so a second harvested architecture reaches this function unchanged.

    A snapshot is a small tracked JSON file with no torch dependency, so this
    stays importable on a machine that has never run the workload.
    """
    from evograd.bench.workloads import load_snapshot

    entry = load_snapshot(workload_name)["level1"][task]
    tolerances = tolerances or {}
    workloads = []
    for config in entry["configurations"]:
        case_dtype = dtype or config["dtype"].replace("torch.", "")
        atol, rtol = tolerances.get(case_dtype, (None, None))
        # The layout of the operator's primary activation input. Both inputs a
        # generator branches on -- RoPE's `x`, attention's q/k/v -- are the
        # leading argument at every harvested boundary.
        primary = (config.get("inputs") or [{}])[0]
        workloads.append(
            Workload(
                dims=dict(config["dims"]),
                dtype=case_dtype,
                atol=atol,
                rtol=rtol,
                provenance=Provenance(
                    model=workload_name,
                    component=config["provenance"]["component"],
                    free=dict(config["provenance"]["free"]),
                    source="hf_config",
                    layout=recorded_layout(
                        primary.get("shape", ()), primary.get("stride", ())
                    ),
                ),
            )
        )
    return tuple(workloads)


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
):
    """Adapt a reviewed Liger autograd pair to evograd's timing-hook API."""

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
    return measure


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
