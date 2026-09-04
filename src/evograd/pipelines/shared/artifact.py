"""The generated-artifact contract, owned by EvoGrad and shared by every pipeline.

A generated candidate is not a pair of functions any more; it is a deployment
artifact with four layers:

1. Triton kernels.
2. **The public pair** -- ``<op>_forward_with_saved`` and
   ``<op>_backward_from_saved``. These *are* the implementation: allocations,
   shape dispatch, grid choice, launches and the saved-state decision live in
   their bodies. There is no ``_impl`` beneath them to forward to.
3. A **static** ``torch.autograd.Function``, generated from the declaration
   rather than written by the model, calling the pair directly.
4. A thin deployment entry -- a callable and an ``nn.Module`` -- named by
   ``DEPLOYMENT_ENTRY`` and consumed unchanged by tier 2 and tier 3.

Layers 3 and 4 are generated identically here for Pipeline A and Pipeline B, so
the two cannot drift apart. Only layers 1 and 2 differ between them, and only
because one is written by a model and the other by a graph compiler.

**The ABI belongs to EvoGrad**, not to the generator: argument order, output
order and arity, upstream-gradient order, input-gradient order, what happens to
inactive arguments, and the rule that an unsupported input raises instead of
quietly running something else. A generator may choose kernels, launch
strategy and saved state; it may not choose the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evograd.opdecl.activity import OpDecl

#: Attribute naming the deployment callable. Present means the artifact wired
#: its own autograd and wants to be called directly; absent means it is a
#: legacy pair and needs the generic binder.
DEPLOYMENT_ENTRY_ATTR = "DEPLOYMENT_ENTRY"

#: Bumped when the required public surface changes.
ARTIFACT_CONTRACT_VERSION = "evograd-artifact/1"


def function_name(op: OpDecl) -> str:
    """The generated ``autograd.Function`` class name for a declaration."""
    return "".join(part.title() for part in op.name.split("_")) + "Function"


def module_name(op: OpDecl) -> str:
    return "".join(part.title() for part in op.name.split("_")) + "Module"


def entry_name(op: OpDecl) -> str:
    return f"{op.name}_deployment"


@dataclass(frozen=True)
class ArtifactContract:
    """Every name and order the ABI pins, derived from one declaration.

    ``allowed_primitives`` is the one part that is not derived: it is the
    capability the *run* granted, recorded in the artifact so a report can say
    which Level-1 providers this operator was permitted rather than infer it.
    """

    op: OpDecl
    allowed_primitives: tuple[str, ...] = ()

    @property
    def forward_fn(self) -> str:
        return self.op.forward_fn_name

    @property
    def backward_fn(self) -> str:
        return self.op.backward_fn_name

    @property
    def function_cls(self) -> str:
        return function_name(self.op)

    @property
    def module_cls(self) -> str:
        return module_name(self.op)

    @property
    def entry(self) -> str:
        return entry_name(self.op)

    @property
    def arg_names(self) -> tuple[str, ...]:
        """Forward argument order. The Function's signature follows it exactly."""
        return tuple(arg.name for arg in self.op.args)

    @property
    def scalar_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.op.scalar_inactive_args())

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.op.active_args())

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.op.output_names)

    @property
    def upstream_names(self) -> tuple[str, ...]:
        return tuple(self.op.upstream_grad_names)

    @property
    def grad_names(self) -> tuple[str, ...]:
        return tuple(self.op.grad_names())

    def gradient_return_order(self) -> list[str | None]:
        """One entry per forward argument: its gradient name, or ``None``.

        ``backward_from_saved`` returns gradients in *active-argument* order;
        an ``autograd.Function`` must return one value per *forward parameter*,
        including ``None`` for every inactive one. Conflating the two orders is
        the classic way a wrapper puts a correct gradient in the wrong slot.
        """
        by_arg = {a.name: a.grad_name for a in self.op.active_args()}
        return [by_arg.get(name) for name in self.arg_names]

    def required_symbols(self) -> tuple[str, ...]:
        return (self.forward_fn, self.backward_fn, self.function_cls,
                self.module_cls, self.entry, DEPLOYMENT_ENTRY_ATTR)

    def metadata(self) -> dict[str, Any]:
        return {
            "contract": ARTIFACT_CONTRACT_VERSION,
            "op": self.op.name,
            "arguments": list(self.arg_names),
            "outputs": list(self.output_names),
            "upstream_grads": list(self.upstream_names),
            "input_grads": list(self.grad_names),
            "gradient_return_order": [g or "None" for g in self.gradient_return_order()],
            "scalars": list(self.scalar_names),
            "deployment_entry": self.entry,
            "allowed_primitives": list(self.allowed_primitives),
        }


# ── generation ───────────────────────────────────────────────────────────────


def render_deployment_layer(op: OpDecl, *, allowed_primitives=()) -> str:
    """Layers 3 and 4, generated from the declaration.

    Emitted verbatim by both pipelines and never placed inside an evolve block:
    a model may rewrite the kernels beneath this, but not the interface above
    it. The only part that tracks the generator's choices is which tensors get
    saved, and that is handled by passing the pair's own saved state straight
    through ``save_for_backward``.
    """
    from evograd.pipelines.shared.primitives import normalize, render_primitive_layer

    allowed_primitives = normalize(allowed_primitives)
    c = ArtifactContract(op, allowed_primitives)
    trusted = render_primitive_layer(allowed_primitives)
    args = ", ".join(c.arg_names)
    scalars = c.scalar_names
    outputs = ", ".join(c.output_names)
    upstream = ", ".join(c.upstream_names)

    scalar_store = "\n".join(
        f"        ctx.{name} = {name}" for name in scalars
    ) or "        pass"
    scalar_load = ", ".join(f"ctx.{name}" for name in scalars)
    backward_scalars = f", {scalar_load}" if scalars else ""

    # `backward_from_saved` takes one upstream gradient for a single-output
    # declaration and a tuple for a structured one; the declaration decides.
    upstream_arg = f"({upstream},)" if len(c.upstream_names) > 1 else upstream
    if len(c.upstream_names) > 1:
        upstream_arg = f"({upstream})"

    forward_capture = (
        f"({outputs}), saved" if len(c.output_names) > 1 else f"{outputs}, saved"
    )
    forward_return = f"return {outputs}"

    grad_terms = ", ".join(
        (g if g is not None else "None") for g in c.gradient_return_order()
    )
    if len(c.gradient_return_order()) == 1:
        grad_terms += ","
    grad_unpack = ", ".join(g for g in c.grad_names)
    if len(c.grad_names) == 1:
        grad_unpack += ","

    flat_args, flat_outs, rank = batched_names(op)
    # Adapt only when the leading-dimension story is unambiguous: every declared
    # output shares the arguments' top rank, so flattening the inputs and
    # restoring the outputs is the same operation on both sides. Where inputs
    # and outputs disagree -- attention takes [B,HQ,T,D] and returns [B,T,H] --
    # there is no single leading shape to restore, and guessing one would
    # silently reshape a result. Those declarations are called at their declared
    # rank and get no adaptation.
    unambiguous = (
        rank is not None
        and rank >= 2
        and bool(flat_args)
        and len(flat_outs) == len(c.output_names)
    )
    if not unambiguous:
        rank_adapt = f"    return {c.function_cls}.apply({args})"
    else:
        lead = flat_args[0]
        flatten = "\n".join(
            f"        {n} = {n}.reshape(-1, {n}.shape[-1])" for n in flat_args
        )
        restore = "\n".join(
            f"        {n} = {n}.view(*_leading, {n}.shape[-1])" for n in flat_outs
        )
        outs_tuple = ", ".join(c.output_names)
        capture = outs_tuple if len(c.output_names) > 1 else c.output_names[0]
        rank_adapt = (
            f"    _leading = None\n"
            f"    if {lead}.dim() > {rank}:\n"
            f"        _leading = {lead}.shape[:-1]\n"
            f"{flatten}\n"
            f"    {capture} = {c.function_cls}.apply({args})\n"
            f"    if _leading is not None:\n"
            f"{restore}\n"
            f"    return {capture}"
        )

    module_args = ", ".join(
        n for n in c.arg_names if n not in (op.parameter_args or ()) and n not in scalars
    )
    param_names = tuple(op.parameter_args or ())
    param_init = "\n".join(
        f"        self.{n} = ({n} if isinstance({n}, torch.nn.Parameter)\n"
        f"                  else torch.nn.Parameter({n}))"
        for n in param_names
    ) or "        pass"
    scalar_init = "\n".join(f"        self.{n} = {n}" for n in scalars) or ""
    call_args = ", ".join(
        f"self.{n}" if n in param_names or n in scalars else n for n in c.arg_names
    )
    module_sig = ", ".join(
        list(param_names) + [f"{n}={_default_repr(op, n)}" for n in scalars]
    )

    return trusted + f'''

# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {c.metadata()!r}


class {c.function_cls}(torch.autograd.Function):
    """Static autograd wiring for `{op.name}`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, {args}):
        {forward_capture} = {c.forward_fn}({args})
        ctx.save_for_backward(*saved)
{scalar_store}
        {forward_return}

    @staticmethod
    def backward(ctx, {upstream}):
        {grad_unpack} = {c.backward_fn}(
            {upstream_arg}, ctx.saved_tensors{backward_scalars}
        )
        # One value per forward argument, `None` for every inactive one.
        return {grad_terms}


def {c.entry}({args}):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
{rank_adapt}


class {c.module_cls}(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, {module_sig}):
        super().__init__()
{param_init}
{scalar_init}
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, {module_args}):
        return {c.entry}({call_args})


DEPLOYMENT_ENTRY = "{c.entry}"
'''


def declared_rank(shape: str | None) -> int | None:
    """Rank of a declared shape string like ``"[rows, cols]"``."""
    if not shape:
        return None
    inner = shape.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    inner = inner.strip()
    return len([p for p in inner.split(",") if p.strip()]) if inner else 0


def batched_names(op: OpDecl) -> tuple[tuple[str, ...], tuple[str, ...], int | None]:
    """Which arguments and outputs carry the model's leading dimensions.

    A declaration states rows-by-cols; a model calls the site with
    ``[batch, tokens, hidden]``. The generic binder used to flatten and restore
    those leading dimensions, so a direct artifact has to do it too -- and it
    is the *deployment entry's* job, because the kernel is entitled to assume
    the declared rank.
    """
    ranks = {
        arg.name: declared_rank(getattr(arg, "shape", None)) for arg in op.args
    }
    known = [r for r in ranks.values() if r]
    if not known:
        return (), (), None
    top = max(known)
    args = tuple(n for n, r in ranks.items() if r == top)
    outs = tuple(o.name for o in op.outputs if declared_rank(o.shape) == top)
    return args, outs, top


def _default_repr(op: OpDecl, name: str) -> str:
    for arg in op.scalar_inactive_args():
        if arg.name == name:
            return repr(arg.default)
    return "None"


# ── validation ───────────────────────────────────────────────────────────────


class ArtifactError(RuntimeError):
    """A generated artifact does not meet the contract."""


#: A generated pair must not delegate to a private twin. The public function is
#: the implementation; a forwarding alias is dead weight that hides where the
#: evolvable code actually lives.
_FORWARDING_ALIAS = re.compile(
    r"def\s+(\w+)\s*\([^)]*\)\s*:\s*(?:#[^\n]*\n\s*)*return\s+_\1_impl\s*\(", re.S
)


def find_forwarding_aliases(source: str) -> list[str]:
    """Public functions whose whole body forwards to a private ``_impl`` twin."""
    names = []
    for match in re.finditer(r"def\s+(\w+)\s*\(", source):
        name = match.group(1)
        if name.startswith("_"):
            continue
        body_start = source.find(":", match.end())
        chunk = source[body_start : body_start + 400]
        if re.match(r":\s*(?:\"\"\".*?\"\"\"\s*)?return\s+_\w*_impl\s*\(", chunk, re.S):
            names.append(name)
    return names


def evolvable_region_covers_pair(op: OpDecl, source: str) -> bool:
    """Do the evolve markers span the public pair, not just the kernels?

    Under the artifact contract the pair bodies *are* the host implementation:
    grids, allocations and the saved-state choice live there. A block that ends
    before them lets evolution rewrite the kernels while freezing the code that
    launches them, which is worse than not evolving at all -- the two have to
    change together.
    """
    lines = source.splitlines()
    starts = [i for i, l in enumerate(lines) if "EVOLVE-BLOCK-START" in l]
    ends = [i for i, l in enumerate(lines) if "EVOLVE-BLOCK-END" in l]
    if not starts or not ends:
        return False
    spans = list(zip(starts, ends))
    for name in (op.forward_fn_name, op.backward_fn_name):
        at = next((i for i, l in enumerate(lines) if l.startswith(f"def {name}(")), None)
        if at is None or not any(b < at < e for b, e in spans):
            return False
    return True


def validate_artifact(op: OpDecl, module: Any, *, source: str | None = None,
                      allowed_primitives=None):
    """Check a loaded artifact against the contract. Raises on any breach.

    ``allowed_primitives`` defaults to whatever the artifact's own
    ``ARTIFACT_CONTRACT`` declares, so a candidate loaded later -- by tier 2, by
    tier 3 -- is held to the capability it was generated under rather than to
    whatever the caller happens to know. Pass it explicitly to check a source
    against a *different* grant, which is what generation does.
    """
    import inspect

    from evograd.pipelines.shared.primitives import (
        PrimitiveViolation,
        check_source,
        normalize,
    )

    if allowed_primitives is None:
        declared = getattr(module, "ARTIFACT_CONTRACT", None)
        allowed_primitives = (declared or {}).get("allowed_primitives", ()) \
            if isinstance(declared, dict) else ()
    allowed_primitives = normalize(allowed_primitives)
    c = ArtifactContract(op, allowed_primitives)
    missing = [name for name in c.required_symbols() if not hasattr(module, name)]
    if missing:
        raise ArtifactError(
            f"{op.name}: artifact is missing {missing}; a direct deployment "
            f"artifact must export {list(c.required_symbols())}"
        )

    entry_attr = getattr(module, DEPLOYMENT_ENTRY_ATTR)
    if not isinstance(entry_attr, str):
        raise ArtifactError(
            f"{op.name}: {DEPLOYMENT_ENTRY_ATTR} must name the deployment "
            f"callable as a string, got {type(entry_attr).__name__}"
        )
    entry = getattr(module, entry_attr, None)
    if not callable(entry):
        raise ArtifactError(
            f"{op.name}: {DEPLOYMENT_ENTRY_ATTR}={entry_attr!r} does not resolve "
            "to a callable"
        )

    for name in (c.forward_fn, c.backward_fn, c.entry):
        if not callable(getattr(module, name)):
            raise ArtifactError(f"{op.name}: {name} is not callable")

    entry_params = list(inspect.signature(entry).parameters)
    if entry_params != list(c.arg_names):
        raise ArtifactError(
            f"{op.name}: deployment entry takes {entry_params}, the declared "
            f"argument order is {list(c.arg_names)}"
        )

    fn = getattr(module, c.function_cls)
    if not (isinstance(fn, type) and issubclass(fn, __import__("torch").autograd.Function)):
        raise ArtifactError(f"{op.name}: {c.function_cls} is not a torch.autograd.Function")
    backward_params = [p for p in inspect.signature(fn.backward).parameters][1:]
    if backward_params != list(c.upstream_names):
        raise ArtifactError(
            f"{op.name}: {c.function_cls}.backward takes {backward_params}, the "
            f"declared upstream-gradient order is {list(c.upstream_names)}"
        )

    if source is not None:
        aliases = find_forwarding_aliases(source)
        if aliases:
            raise ArtifactError(
                f"{op.name}: {aliases} only forward to a private _impl twin; the "
                "public pair functions must be the implementation"
            )
        if not evolvable_region_covers_pair(op, source):
            raise ArtifactError(
                f"{op.name}: the EVOLVE-BLOCK does not span "
                f"{op.forward_fn_name} and {op.backward_fn_name}; the pair bodies "
                "are the host implementation and must be evolvable with the "
                "kernels they launch"
            )
        try:
            check_source(source, allowed=allowed_primitives)
        except PrimitiveViolation as exc:
            raise ArtifactError(f"{op.name}: {exc}") from exc
    return c


def deployment_entry(module: Any):
    """The artifact's own differentiable callable, or ``None`` for a legacy pair."""
    name = getattr(module, DEPLOYMENT_ENTRY_ATTR, None)
    if not isinstance(name, str):
        return None
    call = getattr(module, name, None)
    return call if callable(call) else None
