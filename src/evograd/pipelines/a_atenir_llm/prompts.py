"""Prompt templates for Pipeline A (AtenIR-grounded autograd-pair synthesis).

Plan, codegen and repair share one context (:func:`render_context`): the
forward reference source, the pair rules for the :class:`OpDecl`, the library
primitives the run granted (read from the trusted-primitive registry) and the
AtenIR graph summary.
"""

from __future__ import annotations

from evograd.opdecl.activity import OpDecl
from evograd.pipelines.shared.primitives import REGISTRY, normalize

SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair.  The forward may save tensors
that the backward reuses.  Correctness is required; efficiency should balance
backward latency, forward+backward latency, and saved-tensor memory."""

_GUIDANCE = """\
Saved state:
- `saved_tensors` is a tensor, or a tuple/list of tensors and immutable Python
  scalars; evolution may change it later. Only tensors count toward the saved
  memory the evaluator reports.
- Saving original inputs is fine. Save an intermediate only when it clearly
  pays for its memory; prefer small per-row statistics to activation-sized
  tensors the backward can cheaply recompute.

Triton:
- `tl.arange` bounds are compile-time constants (`BLOCK_*: tl.constexpr`). Pass
  dimensions, strides and scalars as arguments; never read Python globals
  inside `@triton.jit`.
- Accumulate reductions in fp32; prefer partial-buffer reductions to contended
  global atomics.
- Never spin-wait on another program or build a grid-wide barrier (CUDA does
  not guarantee co-residency); use separate launches for multi-pass
  reductions. Hanging candidates are killed and rejected."""

_GRAPH_NOTE = """\
The reference's forward and backward as traced by torch.autograd. Placeholders
are the upstream gradients, then the tensor arguments in declared order; the
outputs are the forward outputs, then the input gradients. Scalars are the
values at extraction; the pair takes its scalar arguments explicitly. The graph
fixes the math, not the kernel structure."""


def _render_primitive_rules(allowed) -> str:
    """The library primitives this run granted, read from the registry."""
    names = normalize(allowed)
    if not names:
        return ("## Library primitives\n\nNone granted: write every contraction "
                "as a Triton `tl.dot`.\n")
    sources = "\n\n".join(REGISTRY[name].source.strip() for name in names)
    return f"""## Library primitives granted for this run

EvoGrad defines these functions after `# EVOLVE-BLOCK-END`; call them by name
from the pair, and do not define, import or copy them. Use them for the work
they cover wherever the reference does it, in the forward and in the backward
(input and weight gradients alike), and spend Triton on what they cannot fuse.
They are the only way to reach a vendor contraction or fused attention: every
other contraction is a Triton `tl.dot`.

```python
{sources}
```
"""


def render_pair_rules(op: OpDecl, allowed_primitives=()) -> str:
    no_grad = ""
    inactive = tuple(c.name for c in op.tensor_inactive_args())
    if inactive:
        names = ", ".join(f"`{n}`" for n in inactive)
        no_grad = (f"- {names} carry NO gradient and are not returned by the "
                   "backward; drop their gradients if the AtenIR graph has them.\n")
    extra = f"\n{op.extra_constraints}\n" if op.extra_constraints else ""
    return f"""\
## Autograd-pair rules

Public API to implement:

```python
def {op.forward_fn_name}({op.forward_parameters()}):
    return {op.forward_returns()}

def {op.backward_fn_name}({op.backward_parameters()}):
    return {op.backward_returns()}
```

- {op.forward_semantics}
- {op.backward_semantics}
{no_grad}{extra}
Output contract:
- The module is Python source only: the imports of `torch`, `triton` and
  `triton.language as tl`, then `# EVOLVE-BLOCK-START`, the kernels, launch
  helpers and both public functions, then `# EVOLVE-BLOCK-END`
  after the two public pair functions, as the last line. Nothing but those
  imports may sit outside the markers: only the block is evolved and checked.
- The two public functions ARE the implementation: allocations, shape
  dispatch, grids, launches and the saved-state choice live in their bodies.
  A body that only returns `_..._impl(...)` is rejected.
- The arithmetic runs in Triton kernels and granted primitives; host code
  allocates, views, checks shapes and launches.
- Do not write a `torch.autograd.Function`, an `nn.Module`, a deployment
  entry, `DEPLOYMENT_ENTRY` or `ARTIFACT_CONTRACT`: EvoGrad generates them from
  the declaration and appends them, with the argument, output and gradient
  orders fixed.
- Static rules, checked on every line inside the block: no `@`; no torch
  contraction, softmax, normalization or attention, as a function or a method
  (`torch.matmul`, `x.mm`, `torch.einsum`, `torch.softmax`, ...); nothing from
  `F` or `torch.nn.functional`; no `torch.autograd`, `.backward()`,
  `torch.compile`, `torch.ops`, `torch._C` or other private `torch._*` name.
  Allocations, views and shape checks on the host are fine.
- An unsupported shape, dtype or device raises; never fall back to eager
  PyTorch, another library or the reference.
- Cache nothing that depends on argument values between calls; all of the
  math runs inside the timed forward or backward.

{_GUIDANCE}

{_render_primitive_rules(allowed_primitives)}"""


def render_context(*, reference: str, graph_summary: str, op: OpDecl,
                   allowed_primitives=(), lowering_context: str = "") -> str:
    """What every prompt of one synthesis shows: reference, rules, graph."""
    lowering = (f"\n## Additional lowering context\n\n```text\n{lowering_context}\n```\n"
                if lowering_context else "")
    return f"""## Forward reference

```python
{reference.rstrip()}
```

{render_pair_rules(op, allowed_primitives)}
## AtenIR graph

{_GRAPH_NOTE}

```text
{graph_summary.rstrip()}
```
{lowering}"""


def render_plan_prompt(context: str) -> str:
    return f"""# Autograd-Pair Planning

Plan a forward/backward autograd pair for the forward reference below.

{context}
Return Markdown with:

1. The saved state the backward receives.
2. The forward and backward Triton kernels, and which products (if any) go to
   a granted primitive.
3. The backward formula and reduction strategy.
4. The saved-state memory and why it is worth the latency.
"""


def render_codegen_prompt(context: str, pair_plan: str) -> str:
    return f"""# Autograd-Pair Codegen

Write the complete Python module for the pair, following the plan.

{context}
## Plan

```markdown
{pair_plan}
```

Return only the Python module.
"""


def render_repair_prompt(context: str, pair_plan: str, previous_code: str,
                         verifier_report: str) -> str:
    return f"""# Autograd-Pair Repair

The previous module failed verification. Return the complete repaired module.

{context}
## Plan

```markdown
{pair_plan}
```

## Previous module

```python
{previous_code.rstrip()}
```

## Verifier report

```json
{verifier_report}
```

Return only the repaired Python source.
"""
