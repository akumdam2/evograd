"""Typed operator declarations for backward-kernel synthesis.

An :class:`OpDecl` is the single source of truth for one operator: which
arguments are differentiable (``Active``) vs non-differentiable
(``Inactive``), their symbolic shapes, the workloads and tolerances used for
correctness/benchmarking, and the prose semantics shown to LLM pipelines.

Everything downstream derives from it: seed-generation prompts (pipelines
A/C), deterministic wrapper codegen (pipeline B), the autograd oracle, the
correctness verifier, and the OpenEvolve evaluator.

Naming rules made explicit (previously string conventions):

* An input's gradient is named ``"d" + name`` unless ``Active.grad``
  overrides it (e.g. ``pair_bias`` -> ``d_pair_bias``).
* The upstream gradient is the output's gradient name (``y`` -> ``dy``,
  ``c`` -> ``dc``, ``out`` -> ``dout``, ``o`` -> ``do``).
* Backward returns gradients of the ``Active`` args in declaration order
  unless ``grad_order`` overrides it (e.g. layernorm_linear returns
  ``dlinear_weight`` before ``dweight``/``dbias``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Active:
    """Differentiable tensor argument whose gradient is required.

    The backward contract must produce a gradient for every ``Active``
    input; the harness routes it to the matching position of the
    ``torch.autograd.Function`` it builds.
    """

    name: str
    shape: str = "unspecified"
    dtype: str | None = None
    grad: str | None = None  # override when "d" + name is not the legacy grad name
    note: str = ""

    @property
    def grad_name(self) -> str:
        return self.grad if self.grad is not None else f"d{self.name}"


@dataclass(frozen=True)
class Inactive:
    """Non-differentiable argument (the autograd binding emits ``None`` for it).

    A tensor if ``shape`` is given (e.g. an additive attention mask), a scalar
    otherwise (e.g. ``eps``). Scalar defaults are re-passed to the backward as
    keyword arguments.
    """

    name: str
    shape: str | None = None
    dtype: str | None = None
    default: float | int | None = None
    note: str = ""

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None


Arg = Active | Inactive


def format_default(value: float | int) -> str:
    """Render scalar defaults in stable Python syntax (``1e-5``, not ``1e-05``)."""
    if isinstance(value, float):
        text = f"{value:g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            sign = "-" if exponent.startswith("-") else "+" if exponent.startswith("+") else ""
            digits = exponent.lstrip("+-").lstrip("0") or "0"
            text = f"{mantissa}e{sign}{digits}"
        return text
    return str(value)


def bind_shape(shape: str, dims: dict[str, int]) -> tuple[int, ...]:
    """Resolve a symbolic shape string against concrete dims: "[B, 1, N]" -> (2, 1, 128)."""
    text = shape.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"shape must look like '[A, B]', got {shape!r}")
    if not text[1:-1].strip():
        return ()
    resolved = []
    for token in text[1:-1].split(","):
        token = token.strip()
        if token in dims:
            resolved.append(dims[token])
        elif token.isdigit():
            resolved.append(int(token))
        else:
            raise ValueError(f"shape dim {token!r} not bound by {sorted(dims)}")
    return tuple(resolved)


#: Where a benchmark workload's shape came from. ``source`` is the strength of
#: the claim, not a free-text note: ``hf_config`` means every dim is derivable
#: from a frozen :mod:`evograd.opdecl.models` entry (and ``tests/test_provenance``
#: re-derives it), ``paper`` means it is quoted from a published sweep, and
#: ``handpicked`` is an honest admission that a human chose the number.
PROVENANCE_SOURCES = ("hf_config", "paper", "handpicked")

#: Dtypes a reference computation may be promoted to. Only float types: the
#: point is to make the reference more accurate than the candidate, and the
#: integer/bool dtypes only ever appear on Inactive inputs (class labels, masks)
#: which are not promoted at all.
_REFERENCE_DTYPES = frozenset({"float32", "float64"})


@dataclass(frozen=True)
class Provenance:
    """The real workload a benchmark shape is traceable to.

    Without this, a benchmark grid is a pile of plausible-looking constants: the
    pre-v1 declarations carried Llama-3 and GPT-2 dimensions in them, but only
    as comments, so nothing stopped a later edit from drifting away from any
    real model. Recording the model and component makes the claim checkable.

    ``scaled`` marks a shape that follows a real configuration in every
    dimension except a deliberately reduced one (e.g. MegaFold benchmarks
    EvoAttention at batch 4; a single-GPU grid runs batch 1). Say which dim was
    scaled and why in ``note``.
    """

    model: str  # registry key in evograd.opdecl.models, e.g. "llama_3_8b"
    component: str  # "mlp_down_proj", "decoder_layer", "pair_bias_attention"
    #: The dims a model configuration does *not* fix — batch, token count, crop
    #: length. Everything else is derived. This is what makes the claim
    #: mechanically checkable: ``MODELS[model].<component>_dims(**free)`` must
    #: reproduce the workload's dims exactly, which ``tests/test_provenance``
    #: asserts for every benchmark case.
    free: dict[str, int] = field(default_factory=dict)
    source: str = "hf_config"
    scaled: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.source not in PROVENANCE_SOURCES:
            raise ValueError(
                f"provenance source must be one of {PROVENANCE_SOURCES}, "
                f"got {self.source!r}"
            )
        if not self.model or not self.component:
            raise ValueError("provenance requires both a model and a component")
        if self.scaled and not self.note:
            raise ValueError(
                f"{self.model}/{self.component}: a scaled provenance must say in "
                "'note' which dimension was scaled and why"
            )


@dataclass(frozen=True)
class Workload:
    """One concrete binding of the operator's symbolic dims, plus dtype.

    ``atol``/``rtol`` override the op-level per-dtype tolerance for this case
    (e.g. evoattention's Dim=128 register-pressure case is looser).

    ``provenance`` traces the shape back to a real model. It is optional on the
    type so ablation suites (``legacy_*``, ``tb_*``) and externally scaffolded
    declarations stay expressible, but every case in a benchmark task's default
    suite is expected to carry one.
    """

    dims: dict[str, int]
    dtype: str
    atol: float | None = None
    rtol: float | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class OpDecl:
    """Complete declaration of one operator's autograd-pair contract."""

    name: str
    forward: str  # forward reference as "module.path:callable"
    # Optional ``path.py:op`` reference for externally generated declarations.
    # Built-in declarations are discovered by package name and leave this None.
    declaration: str | None
    dims: tuple[str, ...]
    args: tuple[Arg, ...]
    output: Active
    forward_semantics: str
    backward_semantics: str
    # Optional second forward, as "module.path:callable": the same mathematics
    # written the way a production PyTorch user would write it.
    #
    # ``forward`` is deliberately unfused — it spells the definition out in
    # primitives so AtenIR can lower it into an unfused seed, and so the oracle
    # differentiates the definition rather than someone's optimization of it.
    # That makes it the wrong thing to *time* against: a LayerNorm written as
    # mean/sub/square/mean/rsqrt/mul/add launches a dozen kernels and reads HBM
    # a dozen times, where ``F.layer_norm`` is one fused kernel. Timing a
    # candidate against the primitive spelling reports how much faster it is
    # than a strawman, and inflates every speedup.
    #
    # When set, the eager baseline is timed through this instead. Operators
    # PyTorch has no fused implementation for (dyt, poly_norm, sparsemax, jsd,
    # tvd) leave it None, because there the primitive spelling *is* the best
    # available PyTorch, and timing against it is honest.
    #
    # The two must agree numerically — ``verify_runtime_forward`` checks it
    # before any timing is trusted, since otherwise this would silently compare
    # the runtimes of two different functions.
    runtime_forward: str | None = None

    # ── benchmark taxonomy ────────────────────────────────────────────────
    # Task hierarchy level: 1 primitive, 2 fused, 3 architectural block.
    # ``None`` means "not a benchmark task" — scaffolded and user-supplied
    # declarations are perfectly usable without claiming a place in the suite.
    # Every built-in operator sets it; ``tests/test_opdecl`` enforces that.
    level: int | None = None
    # Aggregation group for the suite report. Speedups are pooled per family
    # before families are pooled, so a family with many operators cannot
    # dominate the headline number by weight of numbers alone.
    family: str | None = None
    # Compute the forward reference (and therefore the gradient oracle) in this
    # dtype instead of the workload's, then compare the candidate against it.
    # A same-dtype reference carries its own rounding error, so at low precision
    # the tolerance has to cover both sides and stops meaning "how wrong is the
    # candidate". Composing ~10 ops in a level-3 block makes that untenable;
    # a float32 reference restores a tolerance that bounds the candidate's own
    # error. ``None`` keeps the historical same-dtype behavior.
    reference_dtype: str | None = None
    extra_constraints: str = ""
    grad_order: tuple[str, ...] | None = None
    correctness: tuple[Workload, ...] = ()
    # Untimed compile/runtime coverage gate. These workloads need not be part
    # of the performance objective, but every publishable candidate must run
    # forward and backward on all of them.
    coverage: tuple[Workload, ...] = ()
    benchmark: tuple[Workload, ...] = ()
    tolerances: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Per-result multipliers applied to the workload/dtype tolerance. This
    # preserves cases where one reduction is intrinsically noisier than the
    # other outputs (e.g. evoattention d_pair_bias reduces over N_seq).
    tolerance_multipliers: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Optional named benchmark suites. ``benchmark`` remains the default suite;
    # named suites let one declaration expose legacy/TritonBench profiles
    # without environment-dependent module import side effects.
    benchmark_suites: dict[str, tuple[Workload, ...]] = field(default_factory=dict)
    # Optional post-optimization performance baselines. A hook receives the
    # concrete inputs and timing helpers and returns ``(backward_ms, full_ms)``.
    # PyTorch autograd is always available as the built-in default.
    performance_baselines: dict[str, object] = field(default_factory=dict)
    # Only these tensor inputs define the saved-memory budget. Defaults to all
    # tensor inputs. This excludes non-model state such as integer labels from
    # ratios while still allowing the candidate to save them.
    memory_inputs: tuple[str, ...] | None = None
    # Shape-aware specialization metadata. ``regime_feature`` maps a workload
    # to the scalar used to split the declared "small" and "large" suites;
    # ``case_weight`` optionally weights specialist geomeans.
    regime_feature: object | None = None
    regime_split: float | None = None
    case_weight: object | None = None
    # Optional workload-aware tolerance hook:
    # (workload, result_name, base_atol, base_rtol) -> (atol, rtol).
    tolerance_hook: object | None = None
    # Optional override for input construction, signature
    # (torch, op, workload, device) -> dict of {arg name: tensor/scalar, upstream grad name: tensor}.
    # Takes the torch module as a parameter so op declarations stay importable
    # without torch (dev boxes have no CUDA). Default: randn per Active input,
    # zeros per Inactive tensor — override when an Inactive input has semantics (e.g.
    # evoattention's additive keep/drop mask).
    make_inputs: object | None = None
    # Inputs whose storage the backward is permitted to overwrite with their own
    # gradient. The fair-bench protocol otherwise rejects any implementation
    # that mutates a benchmark input, because a candidate that quietly rewrites
    # its inputs both skews repeated timings and skips work everyone else does.
    # But writing a gradient back over the activation that produced it is a real
    # optimization — it is what Liger's SwiGLU does, and it is safe under
    # autograd, which guarantees the activation is dead once the backward has
    # consumed it. Naming those inputs here keeps the rule public and symmetric:
    # every candidate for this operator may reuse exactly these buffers, and
    # the suite report states which ones were relaxed. It is a property of the
    # operator's contract, never of one implementation.
    backward_may_overwrite: tuple[str, ...] = ()

    # ── derived views ─────────────────────────────────────────────────────

    def active_args(self) -> tuple[Active, ...]:
        return tuple(a for a in self.args if isinstance(a, Active))

    def inactive_args(self) -> tuple[Inactive, ...]:
        return tuple(a for a in self.args if isinstance(a, Inactive))

    def tensor_inactive_args(self) -> tuple[Inactive, ...]:
        return tuple(a for a in self.inactive_args() if a.is_tensor)

    def scalar_inactive_args(self) -> tuple[Inactive, ...]:
        return tuple(a for a in self.inactive_args() if not a.is_tensor)

    def grad_names(self) -> tuple[str, ...]:
        """Backward return names, in the order the contract requires."""
        default = tuple(a.grad_name for a in self.active_args())
        return self.grad_order if self.grad_order is not None else default

    def forward_parameters(self) -> str:
        """Candidate-facing forward parameter list, derived from typed args."""
        parts = []
        for arg in self.args:
            if isinstance(arg, Inactive) and not arg.is_tensor and arg.default is not None:
                parts.append(f"{arg.name}={format_default(arg.default)}")
            else:
                parts.append(arg.name)
        return ", ".join(parts)

    def backward_parameters(self) -> str:
        """Candidate-facing backward parameter list."""
        parts = [self.upstream_grad_name, "saved_tensors"]
        for arg in self.scalar_inactive_args():
            suffix = f"={format_default(arg.default)}" if arg.default is not None else ""
            parts.append(f"{arg.name}{suffix}")
        return ", ".join(parts)

    def backward_returns(self) -> str:
        return ", ".join(self.grad_names())

    @property
    def upstream_grad_name(self) -> str:
        return self.output.grad_name

    @property
    def forward_fn_name(self) -> str:
        return f"{self.name}_forward_with_saved"

    @property
    def backward_fn_name(self) -> str:
        return f"{self.name}_backward_from_saved"

    # ── validation ────────────────────────────────────────────────────────

    def validate(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"operator name must be a Python identifier, got {self.name!r}")
        if ":" not in self.forward:
            raise ValueError(
                f"{self.name}: forward must be 'module.path:callable', got {self.forward!r}"
            )
        if self.declaration is not None and ":" not in self.declaration:
            raise ValueError(
                f"{self.name}: declaration must be 'path.py:attribute', "
                f"got {self.declaration!r}"
            )

        names = [a.name for a in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate argument names in {names}")
        invalid_names = [name for name in (*names, self.output.name) if not name.isidentifier()]
        if invalid_names:
            raise ValueError(f"{self.name}: argument/output names must be identifiers: {invalid_names}")
        if len(self.dims) != len(set(self.dims)) or any(not dim.isidentifier() for dim in self.dims):
            raise ValueError(f"{self.name}: dims must be unique Python identifiers: {self.dims}")

        saw_default = False
        for arg in self.args:
            has_default = isinstance(arg, Inactive) and not arg.is_tensor and arg.default is not None
            if saw_default and not has_default:
                raise ValueError(
                    f"{self.name}: required argument {arg.name!r} cannot follow a defaulted scalar Inactive"
                )
            saw_default = saw_default or has_default

        if not isinstance(self.output, Active):
            raise ValueError(f"{self.name}: output must be Active (it receives an upstream gradient)")

        if (self.level is None) != (self.family is None):
            raise ValueError(
                f"{self.name}: level and family must be set together "
                f"(got level={self.level!r}, family={self.family!r})"
            )
        if self.level is not None:
            if self.level not in (1, 2, 3):
                raise ValueError(
                    f"{self.name}: level must be 1 (primitive), 2 (fused), or "
                    f"3 (block), got {self.level!r}"
                )
            if not self.family.isidentifier():
                raise ValueError(
                    f"{self.name}: family must be a Python identifier, got {self.family!r}"
                )
        if self.reference_dtype is not None and self.reference_dtype not in _REFERENCE_DTYPES:
            raise ValueError(
                f"{self.name}: reference_dtype must be one of "
                f"{sorted(_REFERENCE_DTYPES)}, got {self.reference_dtype!r}"
            )
        # Provenance is checked here, beside every other declaration invariant,
        # rather than left to review: an operator that claims a place in the
        # benchmark must say where its timed shapes come from. Only the default
        # suite is covered — named ablation suites (legacy_*, tb_*) are historical
        # controls and would have to invent an origin they do not have.
        if self.level is not None:
            unsourced = [
                workload.dims
                for workload in self.benchmark
                if workload.provenance is None
            ]
            if unsourced:
                raise ValueError(
                    f"{self.name}: level-{self.level} benchmark workloads without "
                    f"provenance: {unsourced}"
                )

        for arg in (*self.args, self.output):
            shape = getattr(arg, "shape", None)
            if shape is None or shape == "unspecified":
                continue
            self._validate_shape(arg.name, shape)

        if self.grad_order is not None:
            default = tuple(a.grad_name for a in self.active_args())
            if sorted(self.grad_order) != sorted(default):
                raise ValueError(
                    f"{self.name}: grad_order {self.grad_order} is not a "
                    f"permutation of the Active gradients {default}"
                )

        grad_names = self.grad_names()
        if len(grad_names) != len(set(grad_names)) or any(
            not name.isidentifier() for name in grad_names
        ):
            raise ValueError(f"{self.name}: gradient names must be unique identifiers: {grad_names}")

        unknown_tolerances = set(self.tolerance_multipliers) - set(self.grad_names())
        if unknown_tolerances:
            raise ValueError(
                f"{self.name}: tolerance multipliers name unknown gradients "
                f"{sorted(unknown_tolerances)}"
            )

        for name, multipliers in self.tolerance_multipliers.items():
            if len(multipliers) != 2 or any(value <= 0 for value in multipliers):
                raise ValueError(
                    f"{self.name}: tolerance multiplier for {name} must be two positive values"
                )
        invalid_baselines = [
            name
            for name, hook in self.performance_baselines.items()
            if not name.isidentifier() or not callable(hook)
        ]
        if invalid_baselines:
            raise ValueError(
                f"{self.name}: performance baseline names/hooks are invalid: {invalid_baselines}"
            )
        tensor_names = {
            arg.name for arg in self.args if getattr(arg, "shape", None) is not None
        }
        if self.memory_inputs is not None:
            unknown_memory_inputs = set(self.memory_inputs) - tensor_names
            if unknown_memory_inputs:
                raise ValueError(
                    f"{self.name}: memory_inputs names non-tensor/unknown inputs "
                    f"{sorted(unknown_memory_inputs)}"
                )
        regime_values = (self.regime_feature, self.regime_split, self.case_weight)
        if any(value is not None for value in regime_values):
            if not callable(self.regime_feature) or self.regime_split is None:
                raise ValueError(
                    f"{self.name}: shape regimes require callable regime_feature "
                    "and numeric regime_split"
                )
            if self.regime_split <= 0:
                raise ValueError(f"{self.name}: regime_split must be positive")
            if self.case_weight is not None and not callable(self.case_weight):
                raise ValueError(f"{self.name}: case_weight must be callable")
        if self.tolerance_hook is not None and not callable(self.tolerance_hook):
            raise ValueError(f"{self.name}: tolerance_hook must be callable")

        suites = [self.benchmark, *self.benchmark_suites.values()]
        for workload in (
            *self.correctness,
            *self.coverage,
            *(w for suite in suites for w in suite),
        ):
            missing = set(self.dims) - set(workload.dims)
            extra = set(workload.dims) - set(self.dims)
            if missing or extra:
                raise ValueError(
                    f"{self.name}: workload dims {sorted(workload.dims)} do not "
                    f"match declared dims {self.dims} "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )
            if (workload.atol is None) != (workload.rtol is None):
                raise ValueError(
                    f"{self.name}: workload atol and rtol must be specified together"
                )
            has_case_tol = workload.atol is not None and workload.rtol is not None
            if not has_case_tol and workload.dtype not in self.tolerances:
                raise ValueError(
                    f"{self.name}: no tolerance for dtype {workload.dtype!r} "
                    f"(neither per-case atol/rtol nor op-level tolerances)"
                )

    def _validate_shape(self, arg_name: str, shape: str) -> None:
        text = shape.strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError(f"{self.name}.{arg_name}: shape must look like '[A, B]', got {shape!r}")
        if not text[1:-1].strip():
            return
        for token in text[1:-1].split(","):
            token = token.strip()
            if token in self.dims or token.isdigit():
                continue
            raise ValueError(
                f"{self.name}.{arg_name}: shape dim {token!r} is neither a "
                f"declared dim {self.dims} nor an integer literal"
            )

    def tolerance_for(
        self, workload: Workload, result_name: str | None = None
    ) -> tuple[float, float]:
        if workload.atol is not None and workload.rtol is not None:
            base = (workload.atol, workload.rtol)
        else:
            base = self.tolerances[workload.dtype]
        scale = self.tolerance_multipliers.get(result_name or "", (1.0, 1.0))
        atol, rtol = base[0] * scale[0], base[1] * scale[1]
        if self.tolerance_hook is not None:
            atol, rtol = self.tolerance_hook(workload, result_name, atol, rtol)
        return float(atol), float(rtol)

    def memory_input_names(self) -> tuple[str, ...]:
        if self.memory_inputs is not None:
            return self.memory_inputs
        return tuple(
            arg.name for arg in self.args if getattr(arg, "shape", None) is not None
        )

    def workload_weight(self, workload: Workload) -> float:
        if self.case_weight is None:
            return 1.0
        return float(self.case_weight(workload))

    def benchmark_workloads(
        self,
        suite: str | None = None,
        dtypes: tuple[str, ...] | None = None,
    ) -> tuple[Workload, ...]:
        """Select a declared benchmark suite and optional dtype subset."""
        if suite in (None, "default"):
            workloads = self.benchmark
        else:
            try:
                workloads = self.benchmark_suites[suite]
            except KeyError:
                available = ["default", *sorted(self.benchmark_suites)]
                raise KeyError(
                    f"{self.name}: unknown benchmark suite {suite!r}; available: {available}"
                ) from None
        if dtypes is not None:
            wanted = set(dtypes)
            unknown = wanted - {w.dtype for w in workloads}
            if unknown:
                raise ValueError(
                    f"{self.name}: benchmark suite has no cases for dtypes {sorted(unknown)}"
                )
            workloads = tuple(w for w in workloads if w.dtype in wanted)
            if not workloads:
                raise ValueError(
                    f"{self.name}: benchmark selection has no cases for dtypes {sorted(wanted)}"
                )
        return workloads


_DTYPE_SHORT = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int64": "i64",
    "int32": "i32",
    "bool": "bool",
}


def example_input_spec(op: OpDecl, workload: Workload | None = None) -> str:
    """The atenir.extract ``--example-input`` string, derived from the declaration.

    One entry per tensor argument in declared order, e.g.
    ``"[(8,64) f32, (64) f32, (64) f32]"`` for layernorm. Replaces the
    hand-typed strings in the old repo's README commands.
    """
    if workload is None:
        if not op.correctness:
            raise ValueError(f"{op.name}: no correctness workloads to derive an example input from")
        # Symbolic tracing specializes sizes 0/1 (see atenir.extract), so an
        # example input with a size-1 symbolic dim bakes that dim into the
        # extracted graph. Prefer the first workload with every dim >= 2.
        workload = next(
            (w for w in op.correctness if all(int(v) >= 2 for v in w.dims.values())),
            op.correctness[0],
        )
    entries = []
    for arg in op.args:
        shape = getattr(arg, "shape", None)
        if shape is None:  # scalar Inactive (e.g. eps): not a placeholder
            continue
        dims = ",".join(str(d) for d in bind_shape(shape, workload.dims))
        dtype_name = arg.dtype if arg.dtype and "|" not in arg.dtype else workload.dtype
        entries.append(f"({dims}) {_DTYPE_SHORT[dtype_name]}")
    return "[" + ", ".join(entries) + "]"


def declare_op(
    *,
    name: str,
    forward: str,
    declaration: str | None = None,
    dims: tuple[str, ...],
    args: tuple[Arg, ...],
    output: Active,
    forward_semantics: str,
    backward_semantics: str,
    level: int | None = None,
    family: str | None = None,
    reference_dtype: str | None = None,
    extra_constraints: str = "",
    grad_order: tuple[str, ...] | None = None,
    correctness: tuple[Workload, ...] = (),
    coverage: tuple[Workload, ...] = (),
    benchmark: tuple[Workload, ...] = (),
    tolerances: dict[str, tuple[float, float]] | None = None,
    tolerance_multipliers: dict[str, tuple[float, float]] | None = None,
    benchmark_suites: dict[str, tuple[Workload, ...]] | None = None,
    performance_baselines: dict[str, object] | None = None,
    memory_inputs: tuple[str, ...] | None = None,
    regime_feature: object | None = None,
    regime_split: float | None = None,
    case_weight: object | None = None,
    tolerance_hook: object | None = None,
    make_inputs: object | None = None,
    backward_may_overwrite: tuple[str, ...] = (),
    runtime_forward: str | None = None,
) -> OpDecl:
    """Build and validate an :class:`OpDecl`. The only way ops should be made."""
    op = OpDecl(
        name=name,
        forward=forward,
        declaration=declaration,
        dims=tuple(dims),
        args=tuple(args),
        output=output,
        forward_semantics=forward_semantics,
        backward_semantics=backward_semantics,
        level=level,
        family=family,
        reference_dtype=reference_dtype,
        extra_constraints=extra_constraints,
        grad_order=tuple(grad_order) if grad_order is not None else None,
        correctness=tuple(correctness),
        coverage=tuple(coverage),
        benchmark=tuple(benchmark),
        tolerances=dict(tolerances) if tolerances is not None else {},
        tolerance_multipliers=(
            dict(tolerance_multipliers) if tolerance_multipliers is not None else {}
        ),
        benchmark_suites=(
            {name: tuple(cases) for name, cases in benchmark_suites.items()}
            if benchmark_suites is not None
            else {}
        ),
        performance_baselines=(
            dict(performance_baselines) if performance_baselines is not None else {}
        ),
        memory_inputs=tuple(memory_inputs) if memory_inputs is not None else None,
        regime_feature=regime_feature,
        regime_split=regime_split,
        case_weight=case_weight,
        tolerance_hook=tolerance_hook,
        make_inputs=make_inputs,
        backward_may_overwrite=tuple(backward_may_overwrite),
        runtime_forward=runtime_forward,
    )
    op.validate()
    return op
