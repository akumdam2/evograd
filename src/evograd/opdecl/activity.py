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
        if ":" not in self.forward:
            raise ValueError(
                f"{self.name}: forward must be 'module.path:callable', got {self.forward!r}"
            )

        names = [a.name for a in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate argument names in {names}")

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

        for workload in (*self.correctness, *self.benchmark):
            missing = set(self.dims) - set(workload.dims)
            extra = set(workload.dims) - set(self.dims)
            if missing or extra:
                raise ValueError(
                    f"{self.name}: workload dims {sorted(workload.dims)} do not "
                    f"match declared dims {self.dims} "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
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

    def tolerance_for(self, workload: Workload) -> tuple[float, float]:
        if workload.atol is not None and workload.rtol is not None:
            return (workload.atol, workload.rtol)
        return self.tolerances[workload.dtype]


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
        make_inputs=make_inputs,
    )
    op.validate()
    return op
