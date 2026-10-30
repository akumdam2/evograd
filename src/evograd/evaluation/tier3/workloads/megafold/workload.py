"""The MegaFold training-step workload: the torch side of the declaration.

Implements ``evaluation.tier3.model.TrainingWorkload`` over `MegaFold`_, the
system-level-optimized AlphaFold3 training stack every AF3 shape in this
repository was transcribed from. Evolved kernels are patched in by module
surgery, which is the route for a model this repository did not write.

MegaFold is a source checkout rather than a distribution: it ships no
``setup.py`` and no ``pyproject.toml``, so it is made importable by putting its
root on ``PYTHONPATH``, not by installing an extra. Everything that reaches it
is deferred into the function that needs it, so this module imports on a
machine that has never seen MegaFold.

**Three providers, one architecture.** MegaFold ships its own fused Triton
kernels for four of the five sites, built directly into the module tree rather
than toggled by a flag, so this workload can put three implementations side by
side on identical weights and batches:

``eager``
    Every fused module replaced by the plain PyTorch mathematics it stands
    for, ``use_optimized_evo=None``. The reference.
``megafold``
    The tree exactly as MegaFold builds it, with its Triton EvoAttention on.
    No surgery.
``evolved``
    The eager tree with the candidate kernels patched into the sites.

``eager`` and ``evolved`` share a base by construction: the eager surgery runs
first and the candidate patches go on top, so the only difference between them
is the kernels.

Five determinism decisions worth knowing before touching this file:

* **The data is one frozen real structure.** A real PDB entry featurized once by
  MegaFold's own training featurization and frozen
  (:mod:`evograd.benchmark.topdown.megafold.data`), loaded only after its hash
  is checked. One structure, because every site's shapes follow its token,
  atom and MSA counts; batches differ by their seed alone.
* **The forward is stochastic.** The diffusion module samples noise and
  augmentations and the pair stack samples structured dropout, so
  :meth:`MegaFoldWorkload.loss` reseeds both ``torch`` and ``random`` from the
  seed the batch carries before every call. Without that, two providers would
  train on different noise and the loss-agreement metric would compare nothing.
* **One recycling pass.** MegaFold draws its recycling count from
  ``randint(1, num_recycling_steps)`` when training; passing 1 makes that draw
  1, so every step makes the same site calls.
* **The autocast context is MegaFold's own.** Its trainer opens none; it passes
  ``dtype=self.dtype`` into the forward and the model opens
  ``torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False)``
  around its whole body. Passing that dtype is therefore what reproduces the
  trainer's mixed precision, resolved here exactly as
  ``megafold.utils.trainer_utils.get_default_supported_precision(training=True)``
  does.
* **The distributed world is a single rank.** MegaFold is always launched by
  ``deepspeed``, so a process group exists even on one GPU, and its forward
  calls ``dist.get_rank()`` unconditionally. This workload starts a one-rank
  gloo group rather than working around the call. Sequence parallelism is
  pinned to a group of one on top of that, which makes every collective a
  no-op on its own short-circuit rather than removing it from the code path.

.. _MegaFold: https://github.com/Supercomputing-System-AI-Lab/MegaFold
"""

from __future__ import annotations

import dataclasses
import os
import random
import socket
from pathlib import Path
from typing import Any, Callable

import torch
import torch._inductor.config
import torch.nn.functional as F
from torch import nn

from evograd.evaluation.tier3.patch import (
    KernelSet,
    KernelSource,
    ModulePatch,
    PatchProvenance,
    Site,
    SiteRegistry,
    patch_modules,
)
from evograd.opdecl.activity import Workload
from evograd.opdecl.models import ALPHAFOLD3, ALPHAFOLD3_2L, AlphaFoldConfig
from evograd.benchmark.topdown.megafold import MEGAFOLD_SITE_OPS

# The LayerNorm replacement is AlphaFold3's, imported rather than copied: the
# two models wrap the same affine nn.LayerNorm and the same evolved layernorm
# pair, and a second identical class would be a second thing to keep correct.
# That module reaches alphafold3-pytorch only inside its own functions, so this
# import costs nothing at import time.
from evograd.evaluation.tier3.workloads.alphafold3.workload import _PatchedLayerNorm

#: MegaFold's own default, and not a value ``configs/megafold_1x1.yaml`` sets.
#: Reported by :meth:`MegaFoldWorkload.describe` regardless, because it sizes
#: the diffusion loss batch and therefore the step.
DIFFUSION_CHUNK_SIZE = 4

#: ``configs/megafold_1x1.yaml`` ``training.initial_training``; the trainer
#: passes it through to the model constructor.
DIFFUSION_NUM_AUGMENTATIONS = 47

#: MegaFold trains with the forward's default of 4, read as the upper end of a
#: ``randint(1, 4)`` draw. 1 makes the draw 1: every step the same site calls.
NUM_RECYCLING_STEPS = 1

#: Batch tensors MegaFold's forward writes into: ``MegaFoldLoss.calculate_confidence_labels``
#: masks ``resolved_labels`` in place (``masked_fill_``) whenever the confidence
#: head is called. :meth:`MegaFoldWorkload.loss` hands the model its own copies,
#: so a batch is the same input at every use.
WRITTEN_INPUTS = ("resolved_labels",)

#: LayerNorm epsilon. MegaFold's fused kernel hardcodes it and its own torch
#: fallback takes ``F.layer_norm``'s default, so both spellings agree on 1e-5.
LAYER_NORM_EPS = 1e-5

#: The provider names that select a tree. Everything else -- ``candidate``, the
#: identity control, a pair baseline -- gets the eager tree and then its own
#: kernels on top, so an evolved provider differs from the reference in its
#: kernels and nothing else.
#:
#: These are provider *names*, not kernel-set contents: the runner passes the
#: name to :meth:`MegaFoldWorkload.build_patched` because this workload's
#: providers are different trees rather than different kernels, and an
#: identical kernel set cannot distinguish them.
BUILD_EAGER = "eager"
BUILD_MEGAFOLD = "megafold"
BUILD_COMPILE = "compile"
#: ``compile`` with MegaFold's scaffolding rebound out of the compiler's way:
#: memory probes, the rank-0 print, sequence-parallel identities, a
#: generator-building decorator, one dtype branch. No mathematics changes; see
#: :mod:`.scaffolding` for the list. Kept beside ``compile`` rather than
#: replacing it, so a report can show what the scaffolding alone cost.
BUILD_COMPILE_CLEAN = "compile_clean"

#: Every provider this workload builds a distinct tree for.
BUILD_MODES = (BUILD_EAGER, BUILD_MEGAFOLD, BUILD_COMPILE, BUILD_COMPILE_CLEAN)

#: How a kernel set asks for the ``compile`` tree whatever its provider is
#: named: the evaluation's whole-model ``torch.compile`` comparison
#: (``evaluation.protocol.whole_model_compile``), as Qwen3's workload marks it.
WHOLE_MODEL_COMPILE_ORIGIN = "whole_model_torch_compile"


def whole_model_compile_kernels(registry):
    """A kernel set that patches nothing and asks for the whole model compiled."""
    return KernelSet(registry=registry, sources=(KernelSource(
        site="__whole_model__", op_name=None, module=None, origin=WHOLE_MODEL_COMPILE_ORIGIN),))


def _surgery_only(site: str):
    """The 'default' of a surgery-route site, which is not a callable path.

    An unpatched site here is the original module in the built tree; nothing
    ever calls through the kernel set for it. Reaching this is a wiring bug,
    and saying so beats returning something that silently computes.
    """

    def refuse(*_args, **_kwargs):
        raise RuntimeError(
            f"site {site!r} is patched by module surgery; the unpatched model "
            "does not call through the kernel set"
        )

    return refuse


def _observed(site: str) -> tuple[Workload, ...]:
    """The configurations the harvest observed at ``site``: its task's observed
    suite, those of the snapshot's configurations its recorder filed under the
    site (``provenance.note``). Two sites share the attention task, and a site
    holds many configurations, so this is what places a case at its site."""
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown import has_snapshot, load_snapshot

    if not has_snapshot("megafold"):
        return ()
    task = MEGAFOLD_SITE_OPS[site]
    snapshot = load_snapshot("megafold")
    entry = snapshot["level1"].get(task) or snapshot["tasks"].get(task) or {}
    workloads = get_task(task).benchmark_suites.get("megafold_observed", ())
    return tuple(w for w, c in zip(workloads, entry.get("configurations", ()))
                 if c["provenance"]["note"] == site)


#: The MegaFold patchable surface, owned by this workload as every registry is.
MEGAFOLD_SITES = SiteRegistry(
    name="megafold",
    sites=tuple(
        Site(site, op_name, _surgery_only(site), preflight=_observed(site))
        for site, op_name in MEGAFOLD_SITE_OPS.items()
    ),
)


#: What to tell someone whose MegaFold is present but will not import. Its
#: fused Triton modules resolve a *driver* at import time, so a machine with no
#: visible GPU fails here with a ``RuntimeError`` about active drivers, raised
#: by a checkout that is complete and whose dependencies are all installed.
_NEEDS_A_GPU = (
    "MegaFold builds its fused Triton modules at import time, which needs a "
    "visible GPU, so this is the expected failure on a login or CPU-only node. "
    "Run the megafold workload on a GPU node."
)


def _as_import_error(what: str, exc: BaseException) -> ImportError:
    """Turn any import-time failure into one sentence tier 3 can report.

    Tier 3 reports a provider's failure by its message alone, so the underlying
    cause goes into the text rather than only into ``__cause__``. Three
    outcomes, because they are three different problems with three different
    fixes and reporting them alike sends someone to repeat what already worked:
    MegaFold is not on ``PYTHONPATH``, a dependency of it is missing, or it is
    complete and this machine has no GPU.
    """
    if isinstance(exc, ImportError):
        missing = getattr(exc, "name", None)
        if missing in (None, "megafold") or str(missing).startswith("megafold"):
            return ImportError(
                "the megafold workload needs the MegaFold source checkout on "
                "PYTHONPATH, and it is not importable. MegaFold ships no "
                "setup.py and no pyproject.toml, so there is no extra to "
                "install and no pip package to add: clone "
                "https://github.com/Supercomputing-System-AI-Lab/MegaFold and "
                "run with PYTHONPATH=/path/to/MegaFold"
            )
        return ImportError(
            f"MegaFold is on PYTHONPATH but {what} cannot be imported: it "
            f"needs {missing!r}, which is missing ({type(exc).__name__}: "
            f"{exc}). Install that dependency -- re-cloning MegaFold will not "
            f"help, and MegaFold's requirements.txt lists what it expects."
        )
    return ImportError(
        f"MegaFold is on PYTHONPATH but {what} cannot be imported: "
        f"{type(exc).__name__}: {exc}. {_NEEDS_A_GPU}"
    )


def _require_model_package():
    """Import ``megafold``, distinguishing absent from broken from GPU-less.

    Not only ``ImportError``: MegaFold's own ``__init__`` reaches the fused
    Triton modules, so on a GPU-less node this raises a ``RuntimeError`` from
    inside Triton's driver resolution. Letting that propagate would put a
    Triton traceback in the middle of a benchmark run and say nothing about
    what to do next.
    """
    try:
        import megafold  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a driver error is not an ImportError
        raise _as_import_error("it", exc) from exc


def _require_model_module():
    """Import the module defining ``MegaFold``, with the same three answers.

    Kept separate from :func:`_require_model_package` because the two can fail
    independently: the package can import while the model module does not, and
    only this one is needed to *build* anything.
    """
    _require_model_package()
    try:
        from megafold.model import megafold as model_module
    except Exception as exc:  # noqa: BLE001 - a driver error is not an ImportError
        raise _as_import_error("its model module", exc) from exc
    return model_module


def _prepare_single_process_run() -> None:
    """Give MegaFold the distributed world it is always launched inside.

    MegaFold is a distributed-first codebase: it is started with
    ``deepspeed --num_gpus=N``, so a process group exists even at ``N = 1``.
    Two things in the model assume that, and both fail outside it.

    **A process group must exist.** ``megafold/model/megafold.py`` calls
    ``dist.get_rank()`` in its forward to guard a print, unconditionally, so
    without one every forward dies with "Default process group has not been
    initialized". A world of one rank is what MegaFold's own single-GPU config
    runs in, so this reproduces its launcher rather than working around it. The
    backend is gloo: with a sequence-parallel group of one no collective is
    ever executed, only ``get_rank`` is asked, and gloo costs no GPU memory and
    no NCCL setup.

    **Sequence parallelism must be declared.** The pairformer and pairwise
    blocks call ``gather_async``, ``row_to_col`` and ``col_to_row``
    unconditionally, and those read a process-global that only the DeepSpeed
    trainer sets. Unset it is ``None``, which fails on subscript rather than
    saying anything useful. Every one of those helpers returns its input
    untouched at ``sp_size == 1`` and none reaches ``torch.distributed`` on
    that path. Sequence parallelism is therefore excluded by being pinned to
    one group, not by being removed from the code path, and the declaration
    says so.
    """
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            # A free ephemeral port, because tier 3 runs each provider in its
            # own child process and a fixed one would collide between them.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(probe.getsockname()[1])
        backend = "gloo" if dist.is_gloo_available() else "nccl"
        dist.init_process_group(backend=backend, world_size=1, rank=0)

    from megafold.distributed.parallel_info import get_parallel_info, set_parallel_info

    if get_parallel_info() is not None:
        return
    set_parallel_info(
        {
            "sp_size": 1,
            "sp_rank": 0,
            "current_sp_process_group": None,
            "ddp_size": 1,
            "ddp_rank": 0,
            "current_ddp_process_group": None,
        }
    )


def _trainer_autocast_dtype() -> torch.dtype:
    """The dtype MegaFold's trainer runs its autocast in, resolved its way.

    ``Trainer.__init__`` takes ``precision=None`` from the yaml and falls back
    to ``get_default_supported_precision(training=True)``, which answers
    ``"bf16-mixed"`` when CUDA is available and supports bfloat16 and
    ``"32-true"`` otherwise; ``parse_dtype`` then maps those to bfloat16 and
    float32. Reproducing the rule rather than the answer keeps a CPU run and a
    GPU run each honest about what they measured.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


# -- replacement modules: the evolved sites ----------------------------------


def _autocast_dtype(t) -> torch.dtype | None:
    """The dtype an enclosing autocast runs reduced-precision ops in, or None."""
    kind = t.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None


def _no_autocast(t):
    """A kernel sees exactly the dtypes its contract declares, never autocast's rewrites."""
    return torch.autocast(device_type=t.device.type, enabled=False)


def _like(tensor, activation):
    """A parameter in the activation's dtype, which autocast will not do here.

    ``torch.autocast`` rewrites the dispatcher's view of *built-in* operators,
    so eager ``F.layer_norm`` happily takes a bfloat16 activation against
    float32 parameters. A ``torch.autograd.Function`` gets no such treatment:
    it is opaque to autocast unless it declares ``torch.amp.custom_fwd``, and
    every kernel patched in here arrives through one. MegaFold hit the same
    wall and answered it the same way, with
    ``@torch.amp.custom_fwd(cast_inputs=torch.bfloat16)`` on each of its fused
    kernels, so aligning to the activation is what its own kernels see.

    The failure this prevents surfaces in the *backward*, not the forward, and
    only under mixed precision: the identity control recomputes its forward to
    differentiate it, backward runs outside the autocast region, and the raw
    ``F.layer_norm`` then refuses a bfloat16 input against a float32 gamma.
    """
    if tensor is None or tensor.dtype == activation.dtype:
        return tensor
    return tensor.to(activation.dtype)


class _PatchedSwiGLUTransition(nn.Module):
    """MegaFold's transition, with its gated activation evolved.

    The module is ``ff = [in_proj, SwiGLU, out_proj]``, and only the middle is
    a site: the two projections are the original module objects, so their
    parameters are carried across rather than reinitialized, under the names
    the original gives them (``ff.0``, ``ff.2``). ``in_proj`` is a
    ``LayernormLinear`` when the transition includes its normalization (the
    default, and what the pairformer and pairwise blocks build) and a bare
    ``LinearNoBias`` when it does not (``include_ln=False``, which the
    diffusion transformer uses). ``FusedTransition`` does not record which it
    was, so this reads the structure instead of a flag.

    MegaFold's spelling chunks one tensor into ``a`` then ``b`` and computes
    ``silu(b) * a``, while the declaration is ``swiglu(a, b) = silu(a) * b``,
    so the declaration's ``a`` is MegaFold's second half and its ``b`` is the
    first. The halves reach the kernel as MegaFold holds them: ``[rows, cols]``
    views of the projection's output, a row stride of ``2 * cols``, in its
    dtype (SwiGLU is elementwise, so autocast leaves it alone).
    """

    def __init__(self, original: nn.Module, kernel: Callable):
        super().__init__()
        sequence = original.ff
        # Sequential is [in_proj, swiglu, out_proj] in every spelling; reading
        # by position rather than by type keeps this working whether the first
        # entry is fused, eager, or already patched by the layernorm_linear
        # site (the patches are independent and either order is legal).
        self.ff = nn.Sequential(sequence[0], _KernelSwiGLU(kernel), sequence[2])

    def forward(self, x):
        return self.ff(x)


class _KernelSwiGLU(nn.Module):
    """The transition's activation through a ``swiglu`` kernel, on ``[rows, cols]`` half views."""

    def __init__(self, kernel: Callable):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        first, gates = x.chunk(2, dim=-1)
        cols = gates.shape[-1]
        with _no_autocast(gates):
            out = self.kernel(gates.reshape(-1, cols), first.reshape(-1, cols))
        return out.reshape(gates.shape)


class _MegaFoldPatchedLayerNorm(_PatchedLayerNorm):
    """AlphaFold3's patched LayerNorm, in MegaFold's precision.

    Subclassed rather than copied: the constructor that carries the original's
    weight and bias across is the same one. What is new is the precision:
    autocast runs ``layer_norm`` in float32 (a bfloat16 input is cast up first)
    and returns float32, so the kernel receives exactly that -- float32 rows,
    ``[rows, hidden]`` -- and the caller gets float32 back, as from the module
    it replaced.
    """

    def forward(self, x):
        rows = x.reshape(-1, x.shape[-1])
        if _autocast_dtype(x) is not None:
            rows = rows.float()
        with _no_autocast(x):
            out = self.kernel(rows, _like(self.weight, rows), _like(self.bias, rows), self.eps)
        return out.reshape(x.shape)


class _PatchedLayernormLinear(nn.Module):
    """MegaFold's ``LayernormLinear``, rerouted through ``megafold_layernorm_linear``.

    The contract is MegaFold's torch path, ``F.linear(F.layer_norm(x, ...),
    linear_weight.T)``, in MegaFold's precision: under autocast the LayerNorm
    runs in float32 (a bfloat16 ``x`` cast up first) and the GEMM in bfloat16
    (the weight cast down), outside it everything is float32. The site makes
    exactly those casts, flattens the leading axes to ``[M, K]`` (a view) and
    hands the kernel float32 rows and the weight in the GEMM's dtype, so the
    output dtype is the original module's. ``linear_weight`` stays ``[K, N]``,
    MegaFold's layout; the cast is autograd's, so its gradient reaches the
    float32 parameter as autocast's would. The parameters keep MegaFold's
    attribute names, so the patched tree's parameters are named as the
    original's.

    Instances with a linear bias (the triangle multiplication's ``combined``)
    or without a LayerNorm bias are a different contract and are left
    unmatched: they run MegaFold's own code.
    """

    def __init__(self, original: nn.Module, kernel: Callable):
        super().__init__()
        self.WEIGHT = original.WEIGHT
        self.BIAS = original.BIAS
        self.linear_weight = original.linear_weight
        self.kernel = kernel

    def forward(self, x):
        lead, k = x.shape[:-1], x.shape[-1]
        rows = x.reshape(-1, k)
        gemm_dtype = _autocast_dtype(x)
        weight = self.linear_weight
        if gemm_dtype is not None:
            rows, weight = rows.float(), weight.to(gemm_dtype)
        with _no_autocast(x):
            out = self.kernel(rows, self.WEIGHT, self.BIAS, weight, LAYER_NORM_EPS)
        return out.reshape(*lead, out.shape[-1])


class _PatchedAttend(nn.Module):
    """MegaFold's ``Attend`` core, rerouted through ``megafold_evoattention``.

    ``Attend`` is the pure attention mathematics after projection, which is
    exactly the boundary the declaration draws, and it is also where MegaFold
    puts its own Triton EvoAttention. ``Attend`` runs ``[b, h, n, d]`` views of
    a ``[b, n, h, d]`` projection; the declaration takes the same memory viewed
    as ``[B, S, N, H, D]`` (no copy), MegaFold's boolean key mask as
    ``[B, S, N]``, and the bias as one ``[B, 1, H, N, N]`` tensor shared across
    ``S`` -- all in the dtype MegaFold computes in there (bfloat16 under
    autocast, float32 where it is off). The MSA axis ``S`` is 1 for pair-bias
    attention over the single representation.

    Configurations the declaration does not model are refused rather than
    silently computed differently: a custom scale, softclamp, attention
    dropout, windowing, or memory key/values would make the patched model a
    different function, and a fast wrong answer is worse than a loud one.
    """

    def __init__(self, original: nn.Module, kernel: Callable):
        super().__init__()
        if original.is_local_attn:
            raise ValueError("evoattention site cannot take a windowed Attend")
        if original.scale is not None:
            raise ValueError("evoattention assumes the default 1/sqrt(d) scale")
        if original.enable_attn_softclamp:
            raise ValueError("evoattention does not implement logit softclamp")
        if original.dropout:
            raise ValueError("evoattention does not implement attention dropout")
        self.kernel = kernel

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        windowed_mask=None,
        attn_bias=None,
        memory_kv=None,
        batch_size=1,
        use_optimized_evo=None,
        **_kwargs,
    ):
        if windowed_mask is not None or memory_kv is not None:
            raise ValueError(
                "evoattention site received windowed/memory-kv attention inputs"
            )
        batch, heads, n, d = q.shape
        # Layout. MegaFold folds the leading axes into one batch before
        # ``Attend`` -- ``[b, h, n, d]`` with ``b = batch_size * n1`` -- and its
        # own Triton path unfolds them again right before the kernel:
        # ``(b n1) h n2 d -> b n1 n2 h d``. The declaration's ``S`` is that
        # ``n1`` axis (one row of the pair representation per sequence), so
        # the site maps ``B = batch_size`` and ``S = b // batch_size``; for
        # pair-bias attention over the single representation ``S == 1``, for
        # triangle attention ``S == n1``. Folding ``n1`` into ``B`` instead
        # would run the kernel at a shape no tier measured and hand it ``S``
        # copies of the bias.
        B = batch_size if batch_size and batch % batch_size == 0 else batch
        S = batch // B
        dtype = q.dtype
        if attn_bias is not None and attn_bias.dtype != dtype:
            raise ValueError(f"attention bias in {attn_bias.dtype} with q in {dtype}: "
                             "MegaFold adds them in one dtype")

        def declared(t):  # [(B S), h, n, d] view -> [B, S, N, H, D] view of the same memory
            return t.permute(0, 2, 1, 3).unflatten(0, (B, S))

        if attn_bias is None:
            pair_bias = q.new_zeros((B, 1, heads, n, n))
        else:
            # The bias is one ``[b, h, n, n]`` tensor per batch element; the
            # declaration broadcasts it over ``S`` and the kernel reduces
            # ``d_pair_bias`` over ``S`` itself. Triangle attention arrives with
            # that one bias ``repeat``-ed over its rows, so the site passes the
            # first row (a view): autograd's gradient into that row is the
            # kernel's S-sum, which is what the repeat's backward would form.
            pair_bias = attn_bias.expand(batch, heads, n, n).reshape(B, S, heads, n, n)[:, :1]
        # MegaFold broadcasts one key mask over the rows (stride 0); the kernel
        # gets it materialized, a few hundred KB, so no argument overlaps itself.
        key_mask = (q.new_ones((B, S, n), dtype=torch.bool) if mask is None
                    else mask.view(B, S, n).contiguous())
        with _no_autocast(q):
            out = self.kernel(declared(q), declared(k), declared(v), key_mask, pair_bias)
        # [B, S, N, H, D] -> [(B S), h, n, d], the layout the caller folded.
        return out.flatten(0, 1).permute(0, 2, 1, 3)


# -- replacement modules: the eager reference --------------------------------


class _EagerLayernormLinear(nn.Module):
    """``LayernormLinear`` as the LayerNorm and Linear it fuses.

    This is MegaFold's own fallback, spelled exactly as MegaFold spells it and
    taken whenever its autotuner has no configuration for a shape:
    ``F.linear(F.layer_norm(X, (K,), WEIGHT, BIAS), linear_weight.T,
    linear_bias)``. The spelling matters under autocast, where ``F.linear``
    casts its input, weight and bias to bfloat16 and adds the bias inside the
    GEMM. Written functionally rather than as ``nn.LayerNorm`` followed by
    ``nn.Linear`` for one reason: ``nn.Linear`` wants its weight as ``[N, K]``
    and MegaFold stores ``[K, N]``, so building one would mean wrapping a
    transposed *view* in a new ``nn.Parameter`` and the gradients would
    accumulate on that new leaf instead of the model's. Holding the original
    parameter objects is the whole point of surgery.

    The upper-case attribute names are MegaFold's, deliberately. This class has
    to be a *drop-in* for the module it replaces, because the candidate patches
    run on top of the eager surgery and must find the same parameters under the
    same names whichever tree they land in. Renaming them to house style is
    what made the evolved provider fail with ``no attribute 'WEIGHT'``.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.WEIGHT = original.WEIGHT
        self.BIAS = original.BIAS
        self.linear_weight = original.linear_weight
        self.linear_bias = original.linear_bias
        self.K = original.K
        self.N = original.N
        self.has_layernorm_bias = original.has_layernorm_bias
        self.has_linear_bias = original.has_linear_bias

    def forward(self, x):
        normalized = F.layer_norm(x, (self.K,), self.WEIGHT, self.BIAS, LAYER_NORM_EPS)
        return F.linear(normalized, self.linear_weight.T, self.linear_bias)


class _EagerSwiGLU(nn.Module):
    """The gated activation, as the commented-out ``SwiGLU`` in MegaFold spells it.

    ``x, gates = chunk(2)`` then ``silu(gates) * x``. This is the module the
    ``transition`` site replaces, and it holds no parameters.
    """

    def forward(self, x):
        first, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * first


class _EagerTransition(nn.Module):
    """``FusedTransition`` as the projection, SwiGLU, projection it fuses.

    Keeps the original's two projections as module objects, so their parameters
    carry across. Like :class:`_EagerLayernormLinear`, it is shaped to be a
    drop-in: the inner ``Sequential`` is named ``ff`` and ordered
    ``[in_proj, activation, out_proj]`` exactly as ``FusedTransition`` builds
    it, so the ``transition`` site's replacement reads either tree by position
    without knowing which one it got.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.ff = nn.Sequential(original.ff[0], _EagerSwiGLU(), original.ff[2])

    def forward(self, x):
        return self.ff(x)


# -- the two patch tables ----------------------------------------------------


def _eager_surgery(model_module) -> tuple[ModulePatch, ...]:
    """Every fused module replaced by the mathematics it stands for.

    Applied before the candidate patches, so ``eager`` and ``evolved`` are the
    same tree except at the sites. Expressed as :class:`ModulePatch` entries
    for familiarity, but applied by :func:`_apply_surgery` rather than
    :func:`patch_modules`: there is no kernel here and nothing to record in a
    provider's provenance, because this is what the provider *is*.

    Order matters. ``LayernormLinear`` goes first so that a transition holding
    one has already had it replaced by the time the transition is rebuilt; the
    transition then keeps whatever module object is in that slot.
    """
    LayernormLinear = model_module.LayernormLinear
    FusedTransition = model_module.FusedTransition

    return (
        ModulePatch(
            "layernorm_linear",
            lambda m: type(m) is LayernormLinear,
            lambda original, _kernel: _EagerLayernormLinear(original),
        ),
        ModulePatch(
            "transition",
            lambda m: type(m) is FusedTransition,
            lambda original, _kernel: _EagerTransition(original),
        ),
    )


#: Top-level modules of ``model`` that hold no site although their modules match
#: one: no gradient reaches them in this workload's step, so a call there could
#: not show a backward fault. The template embedder's output is multiplied by
#: ``has_templates``, False for every structure MegaFold's training featurizes
#: (no kalign); with one recycling pass the recycling projections are applied to
#: zeros and multiplied by 0.0. They run their eager spelling in every provider.
NOT_SITES = ("template_embedder", "recycle_single", "recycle_pairwise")


def _module_patches(model: nn.Module) -> tuple[ModulePatch, ...]:
    """The MegaFold site table, bound to the real module classes and to ``model``'s
    tree (whose :data:`NOT_SITES` modules no matcher takes).

    Every matcher accepts both spellings of a fused module: MegaFold's own and
    the eager replacement this workload installs. That is what lets the
    candidate patches sit on top of the eager surgery, which is what makes
    ``evolved`` differ from ``eager`` in its kernels and nothing else. A
    matcher that recognized only ``FusedTransition`` would find nothing in an
    eager tree and ``patch_modules`` would refuse the provider outright.
    """
    outside = {id(m) for path, m in model.named_modules() if path.split(".", 1)[0] in NOT_SITES}
    model_module = _require_model_module()
    AttentionPairBias = model_module.AttentionPairBias
    LayernormLinear = model_module.LayernormLinear
    FusedTransition = model_module.FusedTransition
    PreLayerNorm = model_module.PreLayerNorm
    TriangleAttention = model_module.TriangleAttention

    def affine_layer_norm(module: nn.Module) -> bool:
        return (
            type(module) is nn.LayerNorm
            and module.elementwise_affine
            and module.bias is not None
        )

    def pre_layer_norm(module: nn.Module) -> bool:
        # The site is the normalization *inside* PreLayerNorm, so the match is
        # on the wrapper: MegaFold also builds bare and non-affine LayerNorms
        # elsewhere (AdaptiveLayerNorm's, for one) that the declaration does
        # not describe, and a match on nn.LayerNorm alone would take those too.
        return type(module) is PreLayerNorm and affine_layer_norm(module.norm)

    def swap_inner_norm(module: nn.Module, kernel) -> nn.Module:
        module.norm = _MegaFoldPatchedLayerNorm(module.norm, kernel)
        return module

    def swap_inner_attend(module: nn.Module, kernel) -> nn.Module:
        # The swap point is the attention core inside the module's Attention;
        # projections, gating and output stay the original's, weights intact.
        module.attn.attend = _PatchedAttend(module.attn.attend, kernel)
        return module

    def fusable_layernorm_linear(module: nn.Module) -> bool:
        # Both spellings answer to the same attribute names, so one predicate
        # covers MegaFold's tree and the eager one alike. The contract has a
        # LayerNorm bias and no linear bias.
        return (
            type(module) in (LayernormLinear, _EagerLayernormLinear)
            and module.has_layernorm_bias
            and module.linear_bias is None
        )

    def transition(module: nn.Module) -> bool:
        return type(module) in (FusedTransition, _EagerTransition)

    def inside(match: Callable[[nn.Module], bool]) -> Callable[[nn.Module], bool]:
        return lambda m: id(m) not in outside and match(m)

    return (
        ModulePatch("layer_norm", inside(pre_layer_norm), swap_inner_norm),
        ModulePatch(
            "layernorm_linear",
            inside(fusable_layernorm_linear),
            _PatchedLayernormLinear,
        ),
        ModulePatch("transition", inside(transition), _PatchedSwiGLUTransition),
        ModulePatch(
            "pair_bias_attention",
            inside(lambda m: type(m) is AttentionPairBias and m.window_size is None),
            swap_inner_attend,
        ),
        ModulePatch(
            "triangle_attention",
            inside(lambda m: type(m) is TriangleAttention),
            swap_inner_attend,
        ),
    )


def _deterministic_distogram_loss(loss_module: nn.Module) -> None:
    """``MegaFoldLoss.calculate_distogram_loss``, its mean taken as one sum over the count.

    MegaFold's ``F.cross_entropy(logits, labels, ignore_index=...)`` reduces through
    ``nll_loss2d``'s atomic adds, the one op in the training step PyTorch's
    deterministic mode refuses -- and the verification settings run in that mode.
    The per-pixel losses are the same call with ``reduction="none"``; their sum
    over the number of counted pixels is the mean ``reduction="mean"`` takes, as
    one deterministic reduction, and every per-pixel gradient is the same.
    """
    from megafold.utils.model_utils import to_pairwise_mask

    ignore = loss_module.ignore_index

    def calculate_distogram_loss(distogram_logits, distogram_labels, atom_mask, pairwise_mask):
        mask = to_pairwise_mask(atom_mask) if loss_module.distogram_atom_resolution else pairwise_mask
        labels = torch.where(mask, distogram_labels, ignore)
        losses = F.cross_entropy(distogram_logits, labels, ignore_index=ignore, reduction="none")
        return losses.sum() / (labels != ignore).sum()

    loss_module.calculate_distogram_loss = calculate_distogram_loss


def _apply_surgery(model: nn.Module, patches: tuple[ModulePatch, ...]) -> int:
    """Replace every matching submodule. Returns how many were replaced.

    The unconditional sibling of :func:`patch_modules`: no kernel set, no site
    selection, no provenance. It builds a provider's base tree rather than
    recording what a provider swapped in.
    """
    replaced = 0
    for spec in patches:
        for path, submodule in list(model.named_modules()):
            if not path or not spec.match(submodule):
                continue
            parent_path, _, attribute = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attribute, spec.build(submodule, None))
            replaced += 1
    return replaced


# -- the workload ------------------------------------------------------------


class MegaFoldWorkload:
    """MegaFold training, as a workload the tier-3 harness can drive.

    Implements the :class:`~evograd.evaluation.tier3.model.TrainingWorkload`
    protocol. Patching is by module surgery on the built tree; the paths
    actually replaced are recorded and reported, so a run states what it
    swapped rather than what it meant to.

    Everything this workload pins that MegaFold would otherwise vary -- the
    structure, the recycling count, whether the confidence head is called, the
    attention implementation, the checkpointing flags -- is a constructor
    argument and appears in :meth:`describe`. Nothing is changed silently.
    """

    unit_name = "residues"
    site_registry = MEGAFOLD_SITES

    def __init__(
        self,
        config: AlphaFoldConfig,
        *,
        batch: int,
        residues: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        use_optimized_evo: str | None = None,
        checkpointing: bool = True,
        num_recycling_steps: int = NUM_RECYCLING_STEPS,
        diffusion_num_augmentations: int = DIFFUSION_NUM_AUGMENTATIONS,
        include_confidence_head: bool = False,
        sample: str | None = None,
        pool: str | None = None,
        init_checkpoint: str | None = None,
    ):
        if use_optimized_evo not in (None, "triton", "deepspeed"):
            raise ValueError(
                f"use_optimized_evo must be None, 'triton' or 'deepspeed', "
                f"got {use_optimized_evo!r}"
            )
        self.config = config
        self.batch = batch
        self.residues = residues
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.use_optimized_evo = use_optimized_evo
        self.checkpointing = checkpointing
        self.num_recycling_steps = num_recycling_steps
        self.diffusion_num_augmentations = diffusion_num_augmentations
        self.diffusion_chunk_size = DIFFUSION_CHUNK_SIZE
        self.include_confidence_head = include_confidence_head
        #: A frozen state loaded over the initialization (the verification's warm
        #: start: at the initialization MegaFold's zero-initialized output layers
        #: keep every gradient from the MSA module, the diffusion module and the
        #: input embedder). Its SHA-256 and recipe are part of ``describe()``.
        self.init_checkpoint = init_checkpoint
        if batch != 1:
            raise ValueError("MegaFold trains at batch 1, which is all the frozen pool serves")
        from evograd.benchmark.topdown.megafold import data

        self.sample = sample or data.SAMPLE
        self.pool = pool or str(data.pool_dir())
        self.name = f"{config.name}[{self.sample},residues={residues}]"
        self._features = None
        self._site_calls = None

    @property
    def checkpoint_flags(self) -> dict[str, bool]:
        """The five ``checkpoint_*`` constructor flags, as one dict.

        ``configs/megafold_1x1.yaml`` turns four of them on and
        ``checkpoint_distogram_head`` off. ``checkpointing=False`` turns the
        four off together, which is the only thing a benchmark run has reason
        to vary: activation checkpointing trades recompute for memory and so
        moves step latency without changing any kernel.
        """
        enabled = self.checkpointing
        return {
            "checkpoint_input_embedding": enabled,
            "checkpoint_trunk_pairformer": enabled,
            "checkpoint_diffusion_module": enabled,
            "checkpoint_confidence_head": enabled,
            # Off in the yaml whatever the others do, so it is not a knob here.
            "checkpoint_distogram_head": False,
        }

    def units_per_step(self) -> int:
        return self.batch * self.residues

    def build(
        self, kernels: KernelSet, *, provider: str | None = None
    ) -> nn.Module:
        model, _provenance = self.build_patched(kernels, provider=provider)
        return model

    def build_patched(
        self, kernels: KernelSet, *, provider: str | None = None
    ) -> tuple[nn.Module, PatchProvenance]:
        """One provider's model, and what this build actually replaced.

        ``provider`` is the name the runner is filing this build under, and it
        selects the tree. ``megafold`` gets MegaFold's own, with its fused
        Triton kernels intact. ``compile`` gets the eager tree wrapped in
        ``torch.compile``. Everything else gets the eager tree, and then
        whatever the kernel set patches on top of it, so a candidate provider
        differs from the reference in its kernels and nothing else.

        The name is a parameter rather than something read out of the kernel
        set: these providers are different trees, an identical kernel set
        cannot tell them apart, and encoding a build mode in ``overrides``
        would put this workload's private convention inside a shared type.
        """
        model_module = _require_model_module()
        _prepare_single_process_run()

        if any(source.origin == WHOLE_MODEL_COMPILE_ORIGIN for source in kernels.sources):
            mode = BUILD_COMPILE
        else:
            mode = provider if provider in BUILD_MODES else BUILD_EAGER
        megafold_mode = mode == BUILD_MEGAFOLD

        # compile_clean rebinds MegaFold module globals and flips dynamo config
        # for the life of the process. Any provider built afterwards in the
        # same process would inherit both and measure something other than
        # what its name claims, so that is refused here rather than allowed to
        # produce a number. The tier-3 CLI's one-child-per-provider isolation
        # never reaches this; `--no-isolate` and the smoke script do.
        from . import scaffolding

        if scaffolding.installed() and mode != BUILD_COMPILE_CLEAN:
            raise RuntimeError(
                f"cannot build provider {provider!r}: the compile_clean "
                "scaffolding patches are installed in this process and cannot "
                "be removed. Build compile_clean last, or run each provider in "
                "its own process (the tier-3 CLI default)."
            )
        # MegaFold's own tree runs its Triton EvoAttention, which is what the
        # yaml asks for; the eager reference must not, or the comparison would
        # be against a partly fused model.
        evo = (self.use_optimized_evo or "triton") if megafold_mode else None

        config = self.config
        torch.manual_seed(self.seed)
        model = model_module.MegaFold(
            # Widths, head counts and stack depths come from the registered
            # configuration, because those are what the declared benchmark dims
            # are derived from; a case could otherwise claim an architecture
            # the model does not have.
            dim_atompair_inputs=config.dim_atompair_inputs,
            dim_template_feats=config.dim_template_feats,
            atoms_per_window=config.atoms_per_window,
            dim_single=config.c_s,
            dim_pairwise=config.c_z,
            confidence_head_kwargs=dict(
                pairformer_depth=config.confidence_pairformer_depth
            ),
            template_embedder_kwargs=dict(
                pairformer_stack_depth=config.template_pairformer_depth
            ),
            msa_module_kwargs=dict(depth=config.msa_module_depth, dim_msa=config.c_m),
            pairformer_stack=dict(
                depth=config.pairformer_depth,
                pair_bias_attn_dim_head=config.pair_bias_attn_dim_head,
                pair_bias_attn_heads=config.pair_bias_attn_heads,
                dropout_row_prob=YAML_PAIRFORMER_DROPOUT_ROW_PROB,
            ),
            diffusion_module_kwargs=dict(
                atom_encoder_depth=config.diffusion_atom_encoder_depth,
                token_transformer_depth=config.diffusion_token_transformer_depth,
                atom_decoder_depth=config.diffusion_atom_decoder_depth,
            ),
            # Everything else is the yaml's, verbatim.
            **YAML_MODEL_KWARGS,
            # This run's own pins, each of them reported by describe().
            use_optimized_evo=evo,
            diffusion_num_augmentations=self.diffusion_num_augmentations,
            diffusion_chunk_size=self.diffusion_chunk_size,
            **self.checkpoint_flags,
        )
        if self.init_checkpoint:
            model.load_state_dict(self.initial_state()[0], strict=True)
        model = model.to(device=self.device, dtype=self.dtype)

        # Attention-weight dropout off, for EVERY provider. MegaFold applies
        # its attention dropout post-aggregation precisely so a fused kernel
        # can be dropped in, but no fused-attention contract models it, and a
        # dropout that fires in one provider and not another would also
        # desynchronize their RNG streams. Structured row/column dropout sits
        # outside the attention core and stays on.
        from megafold.model.attention import Attend

        for module in model.modules():
            if isinstance(module, Attend):
                module.dropout = 0.0
                module.attn_dropout.p = 0.0

        if not megafold_mode:
            _apply_surgery(model, _eager_surgery(model_module))
            _deterministic_distogram_loss(model.loss)

        # patch_modules touches only patched sites, records what it replaced,
        # and raises on a requested site that matches nothing -- the case where
        # MegaFold's class names have moved out from under the matchers.
        provenance = patch_modules(model, _module_patches(model), kernels)

        if mode == BUILD_COMPILE_CLEAN:
            scaffolding.install()

        if mode in (BUILD_COMPILE, BUILD_COMPILE_CLEAN):
            # Inductor generates its own Philox random numbers, so a seeded
            # forward consumes a different stream than eager does, and this
            # model is stochastic in three places at once: diffusion noise,
            # structured row/column dropout, and stochastic frame averaging.
            # Left alone the compiled provider trains on different randomness
            # and its loss is not comparable to the reference, which makes its
            # step time unattributable. Falling back to eager RNG is the
            # documented way to keep the two streams identical; it costs some
            # fusion around the random ops, so this provider's timing is
            # conservative and the report says which mode produced it.
            torch._inductor.config.fallback_random = True

            # Deliberately not fullgraph. MegaFold draws its recycling count
            # with random.randint, loops in Python over recycling steps, calls
            # activation checkpointing, and carries beartype decorators on
            # every forward, so a single graph is not reachable and demanding
            # one would fail the provider rather than measure it. Graph breaks
            # are allowed and the compiled regions are the measurement.
            #
            # No surgery: this is the eager tree and nothing else, so the
            # comparison against `eager` isolates what the compiler did.
            model = torch.compile(model)
        # Which provider built this model, recorded where a report will show
        # it: two providers can request the same sites and still be different
        # trees, and "module_surgery" alone could not tell them apart.
        provenance = PatchProvenance(
            method=f"module_surgery:{mode}",
            requested_sites=provenance.requested_sites,
            actual_sites=provenance.actual_sites,
            paths=provenance.paths,
        )
        return model, provenance

    def initial_state(self):
        """``(state_dict, identity)`` of ``init_checkpoint``, read once."""
        if getattr(self, "_initial", None) is None:
            from evograd.evaluation.tier3.workloads.common import init_state

            self._initial = init_state.load(self.init_checkpoint, "megafold")
        return self._initial

    def expected_site_calls(self) -> dict[str, int]:
        """Each site kernel's calls in one forward and backward of the pinned step:
        the training calls, their activation-checkpoint recomputes and the
        confidence head's no-grad mini-rollout. The step's control flow is pinned
        (one structure, one recycling pass, a fixed rollout), so the calls are
        counted once, on the eager tree with every site patched by a counting
        kernel, and kept; the caller's random state is left as it was."""
        if self._site_calls is None:
            from evograd.benchmark import get_task
            from evograd.opdecl.oracle import resolve_runtime_forward

            registry = self.site_registry
            counts = dict.fromkeys(registry.names, 0)

            def counting(site):
                forward = resolve_runtime_forward(get_task(registry.require(site).op))

                def kernel(*args, **kwargs):
                    counts[site] += 1
                    return forward(*args, **kwargs)

                return kernel

            kernels = KernelSet(registry=registry, patched=tuple(registry.names),
                                overrides={site: counting(site) for site in registry.names})
            state = random.getstate()
            with torch.random.fork_rng():
                try:
                    model = self.build(kernels).train()
                    self.loss(model, self.batch_for(seed=0)).backward()
                finally:
                    random.setstate(state)
            del model
            torch.cuda.empty_cache()
            self._site_calls = counts
        return dict(self._site_calls)

    def _sample(self):
        """The pinned structure, loaded once (hash checked) and checked against the case."""
        if self._features is None:
            _require_model_package()
            from evograd.benchmark.topdown.megafold import data
            from megafold.inputs import collate_inputs_to_batched_atom_input

            atom_input, entry = data.load_sample(self.sample, self.pool)
            if entry["tokens"] != self.residues:
                raise ValueError(f"{self.sample} has {entry['tokens']} tokens; the case asks for "
                                 f"{self.residues} residues")
            batch = collate_inputs_to_batched_atom_input(
                [atom_input], atoms_per_window=self.config.atoms_per_window)
            self._features = (batch, entry, data.pool_manifest(self.pool))
        return self._features

    def batch_for(self, *, seed: int) -> Any:
        """The structure on the device, carrying the seed its step's randomness is drawn from.

        The seed is a (host) tensor of the batch: batches of one structure differ
        by it alone, and a batch's content -- what Gate 2 identifies a sample by --
        must tell them apart.
        """
        batch, _entry, _manifest = self._sample()
        return {"seed": torch.tensor(seed, dtype=torch.int64), **batch.to(self.device).dict()}

    def loss(self, model: nn.Module, batch: Any, *,
             call_confidence_head: bool | None = None) -> torch.Tensor:
        """One forward to MegaFold's own combined loss.

        The reseed from the batch's seed is what makes every provider train on
        the same randomness: the diffusion module samples noise and
        augmentations and the pair stack samples structured dropout. It is
        identical work for every provider and inside the timed region for all
        of them alike. The seed is the batch's, not the workload's, so a batch
        means the same step wherever and whenever it is used.

        ``num_recycling_steps`` is passed rather than left to the forward's
        default so the pin is visible here and in ``describe``: MegaFold draws
        ``randint(1, num_recycling_steps)`` in training mode, which is 1.

        ``call_confidence_head`` is the trainer's own switch, which it drives
        from ``steps % confidence_head_interval == 0`` (every tenth step in the
        yaml). Pinning it (``include_confidence_head``) means every step here is
        the same kind of step; the argument overrides the pin for one call, as
        a training loop that follows the trainer's schedule does. The head's
        forward runs either way; called, it also gets its loss (a diffusion
        mini-rollout under ``no_grad`` makes its labels), and without that no
        gradient reaches it.

        ``dtype`` is how MegaFold's trainer requests mixed precision: the model
        opens its own autocast around the whole forward using the dtype it is
        handed.
        """
        seed = int(batch["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        inputs = {key: value.clone() if key in WRITTEN_INPUTS and torch.is_tensor(value) else value
                  for key, value in batch.items() if key != "seed"}
        return model(
            **inputs,
            num_recycling_steps=self.num_recycling_steps,
            call_confidence_head=(self.include_confidence_head if call_confidence_head is None
                                  else call_confidence_head),
            dtype=_trainer_autocast_dtype(),
        )

    #: What :meth:`primary_output` returns beside the loss.
    primary_output_name = "distogram_logits"
    objective_definition = ("MegaFold's own combined training loss (4 x diffusion + 0.03 x distogram "
                            "+ 1e-4 x confidence, megafold_1x1 weights) of one forward at batch 1, one "
                            "recycling pass, confidence head called or not as the workload pins it; "
                            "backward of that scalar. Primary output: the distogram head's logits.")

    def primary_output(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``(loss, distogram logits)``: the head's output, read by a hook on it."""
        seen = []
        head = getattr(model, "_orig_mod", model).distogram_head
        handle = head.register_forward_hook(lambda _m, _a, out: seen.append(out))
        try:
            loss = self.loss(model, batch)
        finally:
            handle.remove()
        if len(seen) != 1:
            raise RuntimeError(f"the distogram head ran {len(seen)} times in one step")
        return loss, seen[0]

    def data_identity(self) -> dict[str, Any]:
        from evograd.benchmark.topdown.megafold import data

        _batch, entry, manifest = self._sample()
        return {"pool": self.pool, "pool_manifest_sha256": data.sha_file(Path(self.pool) / data.POOL_MANIFEST),
                "sample": self.sample, "sample_sha256": entry["sha256"],
                "megafold_commit": manifest["megafold_commit"], "rule": manifest["rule"]}

    def data_recipe(self) -> dict[str, Any]:
        _batch, entry, manifest = self._sample()
        return {"sample": self.sample, "tokens": entry["tokens"], "atoms": entry["atoms"],
                "msa_rows": entry["msa_rows"], "crop": manifest["crop"], "seed": manifest["seed"],
                "batch": 1, "steps_differ_by": "seed (dropout, diffusion noise, augmentations)"}

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "megafold_train_step",
            "model": self.config.name,
            "pairformer_depth": self.config.pairformer_depth,
            "diffusion_depth": self.config.diffusion_token_transformer_depth,
            "batch": self.batch,
            "residues": self.residues,
            "seed": self.seed,
            "dtype": str(self.dtype).removeprefix("torch."),
            "autocast_dtype": str(_trainer_autocast_dtype()).removeprefix("torch."),
            "use_optimized_evo": self.use_optimized_evo,
            "num_recycling_steps": self.num_recycling_steps,
            "diffusion_num_augmentations": self.diffusion_num_augmentations,
            "diffusion_chunk_size": self.diffusion_chunk_size,
            "include_confidence_head": self.include_confidence_head,
            "weights": "warm_start" if self.init_checkpoint else "random",
            **({"init_checkpoint": self.initial_state()[1]} if self.init_checkpoint else {}),
            "sample": self.sample,
            "pool": self.pool,
            "dim_atom_inputs": YAML_MODEL_KWARGS["dim_atom_inputs"],
            **self.checkpoint_flags,
        }


#: ``configs/megafold_1x1.yaml`` ``model.pairformer_stack.dropout_row_prob``.
YAML_PAIRFORMER_DROPOUT_ROW_PROB = 0.25

#: Every remaining ``model:`` key from ``configs/megafold_1x1.yaml``, verbatim.
#:
#: Read from the file, not remembered. What is deliberately *not* here: the
#: widths and depths the registered AlphaFold3 configuration owns, the
#: ``checkpoint_*`` flags and ``use_optimized_evo`` (constructor arguments, so
#: a run can vary them and say that it did). ``dim_atom_inputs`` is the yaml's
#: 3, the width MegaFold's featurizer emits (formal charge, implicit and
#: explicit valence), not the registered configuration's 77.
YAML_MODEL_KWARGS = {
    "dim_atom_inputs": 3,
    "dim_template_model": 64,
    "num_atom_embeds": 47,
    "num_atompair_embeds": 11,
    "dim_atom": 128,
    "dim_atompair": 16,
    "dim_input_embedder_token": 384,
    "dim_token": 768,
    "ignore_index": -1,
    "num_dist_bins": 64,
    "num_plddt_bins": 50,
    "num_pde_bins": 64,
    "num_pae_bins": 64,
    "sigma_data": 16,
    "loss_confidence_weight": 0.0001,
    "loss_distogram_weight": 0.03,
    "loss_diffusion_weight": 4.0,
    "prior_type": "diffusion",
    "multi_chain_permutation_alignment": True,
    "atom_permutation_alignment": True,
    "use_tempo_layernorm": False,
    "stochastic_frame_average": True,
    "disable_distogram_casting": True,
    "disable_edm_casting": True,
    "disable_sampling_casting": True,
    "disable_confidence_casting": True,
    "disable_loss_casting": True,
    "input_independent_baseline": False,
    "constraints": ["pocket", "contact", "docking"],
}


def make_workload(
    case: Workload,
    *,
    device: str = "cuda",
    seed: int = 0,
    config: AlphaFoldConfig | None = None,
    **settings: Any,
) -> MegaFoldWorkload:
    """Build the training workload for one declared benchmark case.

    ``config`` defaults to whichever registered AlphaFold3 configuration the
    case's dims re-derive from -- the full model for declared benchmark cases,
    the reduced ``alphafold3_2l`` for iteration cases built from it -- and the
    dims are checked against the configuration rather than trusted, so a case
    cannot quietly claim an architecture it does not describe.

    ``settings`` are :class:`MegaFoldWorkload`'s own knobs
    (``use_optimized_evo``, ``checkpointing``, ``num_recycling_steps``,
    ``diffusion_num_augmentations``, ``include_confidence_head``, ``init_checkpoint``), forwarded
    unchanged so the adapter can turn command-line flags into them without this
    function growing a branch per flag. An unknown one is a ``TypeError`` here,
    naming itself, rather than a stray keyword MegaFold's constructor swallows
    into its trailing ``**kwargs``.
    """
    # Checked here rather than at build time: the CLI builds a workload before
    # it builds any provider, so this is where a missing MegaFold can still be
    # reported as one clear sentence instead of surfacing later from inside a
    # timed child process.
    _require_model_package()

    dims = case.dims
    if config is None:
        for candidate in (ALPHAFOLD3, ALPHAFOLD3_2L):
            expected = candidate.train_step_dims(
                batch=dims["batch"], residues=dims["residues"]
            )
            if expected == dims:
                config = candidate
                break
        else:
            raise ValueError(
                f"case dims {dims} re-derive from no registered AlphaFold3 "
                "configuration"
            )
    else:
        expected = config.train_step_dims(
            batch=dims["batch"], residues=dims["residues"]
        )
        if expected != dims:
            raise ValueError(
                f"case dims {dims} do not re-derive from {config.name}: "
                f"expected {expected}"
            )
    return MegaFoldWorkload(
        config,
        batch=dims["batch"],
        residues=dims["residues"],
        device=device,
        dtype=getattr(torch, case.dtype),
        seed=seed,
        **settings,
    )
