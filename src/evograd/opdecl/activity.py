"""Typed operator declarations for backward-kernel synthesis.

An :class:`OpDecl` is the single source of truth for one operator: which
arguments are active (``Duplicated``) vs inactive (``Const``), their symbolic
shapes, the workloads and tolerances used for correctness/benchmarking, and
the prose semantics shown to LLM pipelines.

Everything downstream derives from it: seed-generation prompts (pipelines
A/C), deterministic wrapper codegen (pipeline B), the autograd oracle, the
correctness verifier, and the OpenEvolve evaluator.

Naming rules made explicit (previously string conventions):

* An input's gradient is named ``"d" + name`` unless ``Duplicated.grad``
  overrides it (e.g. ``pair_bias`` -> ``d_pair_bias``).
* The upstream gradient is the output's gradient name (``y`` -> ``dy``,
  ``c`` -> ``dc``, ``out`` -> ``dout``, ``o`` -> ``do``).
* Backward returns gradients of the ``Duplicated`` args in declaration order
  unless ``grad_order`` overrides it (e.g. layernorm_linear returns
  ``dlinear_weight`` before ``dweight``/``dbias``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Duplicated:
    """Active tensor argument: primal plus gradient ("duplicated" = primal + shadow).

    The backward contract must produce a gradient for every ``Duplicated``
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
class Const:
    """Inactive argument: carries no gradient (harness emits ``None`` for it).

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


Arg = Duplicated | Const


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


@dataclass(frozen=True)
class Workload:
    """One concrete binding of the operator's symbolic dims, plus dtype.

    ``atol``/``rtol`` override the op-level per-dtype tolerance for this case
    (e.g. evoattention's Dim=128 register-pressure case is looser).
    """

    dims: dict[str, int]
    dtype: str
    atol: float | None = None
    rtol: float | None = None


@dataclass(frozen=True)
class OpDecl:
    """Complete declaration of one operator's autograd-pair contract."""

    name: str
    forward: str  # forward reference as "module.path:callable"
    dims: tuple[str, ...]
    args: tuple[Arg, ...]
    output: Duplicated
    forward_semantics: str
    backward_semantics: str
    extra_constraints: str = ""
    grad_order: tuple[str, ...] | None = None
    correctness: tuple[Workload, ...] = ()
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
    # Optional override for input construction, signature
    # (torch, op, workload, device) -> dict of {arg name: tensor/scalar, upstream grad name: tensor}.
    # Takes the torch module as a parameter so op declarations stay importable
    # without torch (dev boxes have no CUDA). Default: randn per Duplicated,
    # zeros per Const tensor — override when a Const has semantics (e.g.
    # evoattention's additive keep/drop mask).
    make_inputs: object | None = None

    # ── derived views ─────────────────────────────────────────────────────

    def duplicated_args(self) -> tuple[Duplicated, ...]:
        return tuple(a for a in self.args if isinstance(a, Duplicated))

    def const_args(self) -> tuple[Const, ...]:
        return tuple(a for a in self.args if isinstance(a, Const))

    def tensor_const_args(self) -> tuple[Const, ...]:
        return tuple(a for a in self.const_args() if a.is_tensor)

    def scalar_const_args(self) -> tuple[Const, ...]:
        return tuple(a for a in self.const_args() if not a.is_tensor)

    def grad_names(self) -> tuple[str, ...]:
        """Backward return names, in the order the contract requires."""
        default = tuple(a.grad_name for a in self.duplicated_args())
        return self.grad_order if self.grad_order is not None else default

    def forward_parameters(self) -> str:
        """Candidate-facing forward parameter list, derived from typed args."""
        parts = []
        for arg in self.args:
            if isinstance(arg, Const) and not arg.is_tensor and arg.default is not None:
                parts.append(f"{arg.name}={format_default(arg.default)}")
            else:
                parts.append(arg.name)
        return ", ".join(parts)

    def backward_parameters(self) -> str:
        """Candidate-facing backward parameter list."""
        parts = [self.upstream_grad_name, "saved_tensors"]
        for arg in self.scalar_const_args():
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
            has_default = isinstance(arg, Const) and not arg.is_tensor and arg.default is not None
            if saw_default and not has_default:
                raise ValueError(
                    f"{self.name}: required argument {arg.name!r} cannot follow a defaulted scalar Const"
                )
            saw_default = saw_default or has_default

        if not isinstance(self.output, Duplicated):
            raise ValueError(f"{self.name}: output must be Duplicated (it has a gradient)")

        for arg in (*self.args, self.output):
            shape = getattr(arg, "shape", None)
            if shape is None or shape == "unspecified":
                continue
            self._validate_shape(arg.name, shape)

        if self.grad_order is not None:
            default = tuple(a.grad_name for a in self.duplicated_args())
            if sorted(self.grad_order) != sorted(default):
                raise ValueError(
                    f"{self.name}: grad_order {self.grad_order} is not a "
                    f"permutation of the Duplicated gradients {default}"
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

        suites = [self.benchmark, *self.benchmark_suites.values()]
        for workload in (*self.correctness, *(w for suite in suites for w in suite)):
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
        return base[0] * scale[0], base[1] * scale[1]

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


_DTYPE_SHORT = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


def example_input_spec(op: OpDecl, workload: Workload | None = None) -> str:
    """The atenir.extract ``--example-input`` string, derived from the declaration.

    One entry per tensor argument in declared order, e.g.
    ``"[(8,64) f32, (64) f32, (64) f32]"`` for layernorm. Replaces the
    hand-typed strings in the old repo's README commands.
    """
    if workload is None:
        if not op.correctness:
            raise ValueError(f"{op.name}: no correctness workloads to derive an example input from")
        workload = op.correctness[0]
    entries = []
    for arg in op.args:
        shape = getattr(arg, "shape", None)
        if shape is None:  # scalar Const (e.g. eps): not a placeholder
            continue
        dims = ",".join(str(d) for d in bind_shape(shape, workload.dims))
        dtype_name = arg.dtype if arg.dtype and "|" not in arg.dtype else workload.dtype
        entries.append(f"({dims}) {_DTYPE_SHORT[dtype_name]}")
    return "[" + ", ".join(entries) + "]"


def declare_op(
    *,
    name: str,
    forward: str,
    dims: tuple[str, ...],
    args: tuple[Arg, ...],
    output: Duplicated,
    forward_semantics: str,
    backward_semantics: str,
    extra_constraints: str = "",
    grad_order: tuple[str, ...] | None = None,
    correctness: tuple[Workload, ...] = (),
    benchmark: tuple[Workload, ...] = (),
    tolerances: dict[str, tuple[float, float]] | None = None,
    tolerance_multipliers: dict[str, tuple[float, float]] | None = None,
    benchmark_suites: dict[str, tuple[Workload, ...]] | None = None,
    performance_baselines: dict[str, object] | None = None,
    make_inputs: object | None = None,
) -> OpDecl:
    """Build and validate an :class:`OpDecl`. The only way ops should be made."""
    op = OpDecl(
        name=name,
        forward=forward,
        dims=tuple(dims),
        args=tuple(args),
        output=output,
        forward_semantics=forward_semantics,
        backward_semantics=backward_semantics,
        extra_constraints=extra_constraints,
        grad_order=tuple(grad_order) if grad_order is not None else None,
        correctness=tuple(correctness),
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
        make_inputs=make_inputs,
    )
    op.validate()
    return op
