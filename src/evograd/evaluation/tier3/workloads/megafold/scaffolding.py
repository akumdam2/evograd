"""Scaffolding patches for the ``compile_clean`` provider.

``torch.compile`` on MegaFold as written breaks its graph over a hundred times,
and almost none of those breaks are the model's mathematics. They are memory
probes that print to a log, a rank-0 print helper, sequence-parallel collectives
that are identities at group size one, a decorator that builds a generator, and
a handful of host-side scalar reads. The ``compile`` provider measures the
compiler on that code as it stands. This module is what ``compile_clean``
installs on top of it: the same model with the scaffolding taken out of the
compiler's way, and **no operator, shape, dtype or random draw changed**.

Every patch here is a rebinding of a module-level name that MegaFold's forward
reaches through its module globals, done at model-build time. Each one records
the upstream file and line it neutralizes and why the result is the same
function, and :func:`install` returns that list so a report can print it.

Process-scoped, by nature. A rebound module global stays rebound for the life
of the process, and so does a ``torch._dynamo.config`` flag, so these must not
be installed in a process that will go on to build another provider: the
tier-3 CLI's default of one child process per provider is what keeps them
apart, and :func:`installed` is how the workload refuses to build anything else
afterwards in-process.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

#: What has been installed, as ``{"target": ..., "upstream": ..., "why": ...}``
#: entries, in order. Empty until :func:`install` runs.
APPLIED: list[dict[str, str]] = []

_INSTALLED = False


def installed() -> bool:
    return _INSTALLED


def _record(target: str, upstream: str, why: str) -> None:
    APPLIED.append({"target": target, "upstream": upstream, "why": why})


# -- replacements ------------------------------------------------------------


class _NoMemoryProbe:
    """Stands in for DeepSpeed's ``SynchronizedWallClockTimer`` in MegaFold's namespace.

    MegaFold interpolates ``memory_usage()`` into ten f-strings handed to its
    rank-0 print, so the probe runs on every forward whether or not anything is
    printed, and each probe is four ``torch.cuda.*memory*`` queries that return
    Python numbers, which dynamo cannot put in a graph.
    """

    @staticmethod
    def memory_usage() -> str:
        return ""


def _no_print(self, *args: Any, **kwargs: Any) -> None:
    """``MegaFold.print``: a rank-0 ``print``, which dynamo cannot trace."""


def _identity(x, *args: Any, **kwargs: Any):
    return x


def _gather_async_identity(x, dim: int = -1):
    # The real one returns ``(tensor, work)`` and its opposite waits on
    # ``work``; at group size one the tensor is returned untouched and the
    # work handle is ``None``, which is exactly this.
    return x, None


def _gather_async_opp_identity(x, work, dim: int = -1):
    return x


def _wraps_identity(fn):
    """``functools.wraps`` as MegaFold's MSA module uses it, without the ``setattr``.

    ``to_checkpointed_layers`` decorates four closures with ``@wraps(fn)`` on
    every forward. ``wraps`` copies metadata onto the closure through
    ``setattr``, which dynamo cannot trace; the metadata is never read. The
    closure is returned as it is.
    """

    def decorator(inner):
        return inner

    return decorator


def _autocasting_disable_decorator(disable_casting: bool, cache_enabled: bool = False):
    """MegaFold's decorator, with its generator expression made a tuple.

    Upstream builds the positional arguments as a generator that casts each
    floating tensor as it is consumed. Dynamo refuses to reconstruct a
    generator that mutates state and falls back to eager for the frame. A
    tuple built up front casts exactly the same tensors to exactly the same
    dtype; only the laziness is gone.
    """

    def cast(value):
        dtype = torch.float32 if disable_casting else None
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value.to(dtype=dtype)
        if isinstance(value, dict):
            return {
                key: (
                    item.to(dtype=dtype)
                    if isinstance(item, torch.Tensor) and torch.is_floating_point(item)
                    else item
                )
                for key, item in value.items()
            }
        return value

    def func_wrapper(func):
        def new_func(*args, **kwargs):
            context = (
                torch.autocast(device_type="cuda", enabled=False, cache_enabled=cache_enabled)
                if disable_casting
                else nullcontext()
            )
            with context:
                return func(
                    *tuple(cast(arg) for arg in args),
                    **{key: cast(value) for key, value in kwargs.items()},
                )

        return new_func

    return func_wrapper


def _make_batch_repeat_interleave(model_utils):
    """``batch_repeat_interleave`` with its Python dtype branch removed.

    Upstream picks the padding value with ``False if dtype == torch.bool else
    0``. That is a Python branch on the input dtype, so every distinct dtype
    the function is called with specializes the frame again, and at eight it
    hit dynamo's recompile limit and the frame fell back to eager for good.
    ``masked_fill`` with ``0`` on a boolean tensor writes ``False``, so the
    value ``0`` serves every dtype and the branch is simply not needed. Every
    other line is upstream's, in upstream's order.
    """
    from einops import rearrange, repeat

    default = model_utils.default
    exclusive_cumsum = model_utils.exclusive_cumsum
    lens_to_mask = model_utils.lens_to_mask
    pack_one = model_utils.pack_one

    def batch_repeat_interleave(feats, lens, output_padding_value=None):
        device = feats.device
        batch, seq, *dims = feats.shape

        mask = lens_to_mask(lens)

        window_size = mask.shape[-1]
        arange = torch.arange(window_size, device=device)

        offsets = exclusive_cumsum(lens)
        indices = arange[None, None, :] + offsets[..., None]

        total_lens = lens.clamp(min=0).sum(dim=-1)
        output_mask = lens_to_mask(total_lens)

        max_len = total_lens.amax()

        output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

        indices = indices.masked_fill(~mask, max_len)
        indices = rearrange(indices, "b n w -> b (n w)")

        seq_arange = torch.arange(seq, device=device)
        seq_arange = repeat(seq_arange, "n -> b (n w)", b=batch, w=window_size)

        output_indices = output_indices.scatter(1, indices, seq_arange)
        output_indices = output_indices[:, :-1]

        feats, unpack_one = pack_one(feats, "b n *")
        output_indices = repeat(output_indices, "b m -> b m d", d=feats.shape[-1])
        output = feats.gather(1, output_indices)
        output = unpack_one(output)

        # upstream: default(output_padding_value, False if dtype == torch.bool else 0)
        output_padding_value = default(output_padding_value, 0)

        output = output.masked_fill(
            ~output_mask[(...,) + (None,) * (output.dim() - output_mask.dim())],
            output_padding_value,
        )
        return output

    return batch_repeat_interleave


def _make_pad_or_slice_to(model_utils):
    """``pad_or_slice_to`` without the Python branch on the current length.

    Upstream compares ``t.shape[dim]`` to ``length`` in an ``if``. Once
    ``batch_repeat_interleave``'s output width is a symbolic size, which is
    what letting the compiler keep scalar reads in the graph makes it, that
    comparison is data-dependent control flow and dynamo abandons the frame,
    along with the two generator expressions in the loss code that call this.
    ``narrow`` to ``min(current, length)`` then pad by the remainder produces
    the same tensor for every case the branch handled: shorter is padded,
    longer is sliced, equal is copied unchanged.
    """
    pad_at_dim = model_utils.pad_at_dim

    def pad_or_slice_to(t, length, *, dim, pad_value=0):
        current = t.shape[dim]
        keep = min(current, length)
        t = t.narrow(dim, 0, keep)
        return pad_at_dim(t, (0, length - keep), dim=dim, value=pad_value)

    return pad_or_slice_to


# -- installation ------------------------------------------------------------


def install() -> list[dict[str, str]]:
    """Rebind MegaFold's scaffolding out of the compiler's way. Idempotent.

    Returns the list of patches applied, each naming the upstream site.
    """
    global _INSTALLED
    if _INSTALLED:
        return list(APPLIED)

    import megafold.model.attention as attention
    import megafold.model.megafold as megafold_model
    import megafold.utils.model_utils as model_utils
    from megafold.distributed.parallel_info import get_parallel_info

    info = get_parallel_info()
    if info is None or info.get("sp_size") != 1:
        raise RuntimeError(
            "compile_clean short-circuits the sequence-parallel helpers, which "
            "is only an identity at group size 1; the parallel info is "
            f"{info!r}"
        )

    # 1. Memory probes. DeepSpeed's accelerator wrappers (cuda_accelerator.py
    #    lines 142, 145, 151, 154) return Python numbers and were the single
    #    largest source of breaks.
    attention.get_mem = lambda: 0.0
    _record(
        "megafold.model.attention.get_mem",
        "megafold/model/attention.py:59, called at :224 on every attention",
        "returns a host float from torch.cuda.memory_allocated; only ever "
        "subtracted into an optional list nothing reads",
    )
    megafold_model.SynchronizedWallClockTimer = _NoMemoryProbe
    _record(
        "megafold.model.megafold.SynchronizedWallClockTimer",
        "megafold/model/megafold.py:9182-10016, ten memory_usage() f-strings",
        "four torch.cuda memory queries per probe, interpolated into log "
        "strings; nothing in the model reads them",
    )

    # 2. The rank-0 print.
    megafold_model.MegaFold.print = _no_print
    _record(
        "megafold.model.megafold.MegaFold.print",
        "megafold/model/megafold.py:10023",
        "dist.get_rank() and a builtin print; logging only",
    )

    # 3. Sequence-parallel helpers, identities at group size one but still
    #    routed through autograd.Functions whose backward reads a saved dim
    #    with int(tensor).
    for name in ("scatter", "gather", "row_to_col", "col_to_row",
                 "reduce_sum", "reduce_scatter_sum"):
        setattr(megafold_model, name, _identity)
    megafold_model.gather_async = _gather_async_identity
    megafold_model.gather_async_opp = _gather_async_opp_identity
    attention.gather_async_opp = _gather_async_opp_identity
    _record(
        "megafold.model.megafold.{scatter,gather,row_to_col,col_to_row,"
        "reduce_sum,reduce_scatter_sum,gather_async,gather_async_opp} and "
        "megafold.model.attention.gather_async_opp",
        "megafold/distributed/comm.py:111 and :132 (int(dim) in Scatter/Gather "
        "backward), and the All_to_All/ReduceSum wrappers",
        "every one returns its input unchanged when sp_size == 1; the "
        "autograd.Function wrappers are what dynamo could not trace",
    )

    # 4. The autocast-toggling decorator's generator expression.
    megafold_model.autocasting_disable_decorator = _autocasting_disable_decorator
    model_utils.autocasting_disable_decorator = _autocasting_disable_decorator
    _record(
        "megafold.model.megafold.autocasting_disable_decorator",
        "megafold/utils/model_utils.py:1190-1229 (generator at :1217)",
        "same casts, same autocast context, arguments built as a tuple "
        "instead of a generator",
    )

    # 5. functools.wraps inside MSAModule.to_checkpointed_layers.
    megafold_model.wraps = _wraps_identity
    _record(
        "megafold.model.megafold.wraps",
        "megafold/model/megafold.py:1107-1134, via functools.py:56 setattr",
        "metadata copied onto per-forward closures is never read",
    )

    # 6. batch_repeat_interleave's dtype branch.
    replacement = _make_batch_repeat_interleave(model_utils)
    model_utils.batch_repeat_interleave = replacement
    megafold_model.batch_repeat_interleave = replacement
    _record(
        "megafold.utils.model_utils.batch_repeat_interleave",
        "megafold/utils/model_utils.py:630 (dtype == torch.bool branch)",
        "padding value 0 writes False into a bool tensor; the Python branch "
        "that specialized the frame per dtype is gone, nothing else changed",
    )

    # 6b. pad_or_slice_to's branch on a now-symbolic length.
    replacement = _make_pad_or_slice_to(model_utils)
    model_utils.pad_or_slice_to = replacement
    if hasattr(megafold_model, "pad_or_slice_to"):
        megafold_model.pad_or_slice_to = replacement
    _record(
        "megafold.utils.model_utils.pad_or_slice_to",
        "megafold/utils/model_utils.py:381 (if curr_length < length), reached "
        "from the generators at :831 and :2062",
        "narrow to min(current, length) then pad the remainder: the same "
        "tensor as the branch produced, with no Python comparison on a size",
    )

    # 7. Dynamo: let the remaining host-side scalar reads stay in the graph as
    #    unbacked symbolic ints rather than breaking on each. This traces the
    #    same ops; it changes what the compiler is allowed to reason about.
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    _record(
        "torch._dynamo.config.capture_scalar_outputs, "
        "capture_dynamic_output_shape_ops",
        "megafold/utils/model_utils.py:459 (lens.amax() into arange), :601, "
        ":1167; megafold/model/megafold.py:9607, :6342 (boolean indexing)",
        "the reads and nonzero calls still execute; dynamo carries their "
        "results as unbacked symbols instead of breaking the graph",
    )

    _INSTALLED = True
    return list(APPLIED)
