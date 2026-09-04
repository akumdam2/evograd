"""Prompt templates for Pipeline A (AtenIR-grounded autograd-pair synthesis).

Ported from the old repo's ``pipeline/autograd_pair_fusion_agent/prompts.py``.
The prompts consume :class:`OpDecl` directly; there are no JSON spec files or
transitional stringly-typed contracts.
"""

from __future__ import annotations

from evograd.opdecl.activity import OpDecl

SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair.  The forward may save tensors
that the backward reuses.  Correctness is required; efficiency should balance
backward latency, forward+backward latency, and saved-tensor memory."""

_SAVED_TENSOR_GUIDANCE = """\
Saved-tensor guidance:
- The saved tensor tuple is part of the evolvable program state.  The initial
  seed may save only original inputs; OpenEvolve may add, remove, or reorder
  saved tensors as long as the forward and backward agree.
- You may save forward intermediates if doing so improves the forward+backward
  tradeoff, but the prompt does not prescribe which intermediates to save.
- Prefer compact saved state such as small per-row/per-block statistics over
  full activation-sized tensors when the backward can cheaply reconstruct the
  larger intermediate.
- Avoid saving tensors with the same shape as a large activation unless the
  latency benefit clearly outweighs the memory cost.
- Do not save excessive large intermediates unless they clearly improve
  forward+backward latency.  The evaluator reports saved memory.
- It is acceptable to save original inputs if the backward needs them."""

_TRITON_PITFALLS = """\
Triton pitfalls:
- `tl.arange` bounds must be compile-time constants. Use `BLOCK_*: tl.constexpr`.
- Do not read Python globals inside `@triton.jit`; pass dimensions, strides, and
  scalar constants as arguments or meta-parameters.
- Use fp32 accumulation for reductions.
- Avoid global atomic contention when a partial-buffer reduction is better.
- Never spin-wait for another program/block or attempt a grid-wide barrier.
  CUDA does not guarantee block co-residency; use separate kernel launches for
  multi-pass reductions. Every benchmark shape is smoke-run and hangs are
  killed and rejected."""


_NO_PRIMITIVES = """
- Every arithmetic operation belongs in a Triton kernel. `torch.matmul`, `@`,
  `torch.mm`, `F.linear`, SDPA and any eager RMSNorm/RoPE spelling are
  forbidden anywhere in the evolvable block, and so are `torch.compile`,
  `torch.autograd` and `.backward()`.
"""


def _render_primitive_rules(allowed) -> str:
    """What Level-1 providers this run granted, spelled out for the generator."""
    from evograd.pipelines.shared.primitives import REGISTRY, normalize

    names = normalize(allowed)
    if not names:
        return _NO_PRIMITIVES
    lines = [
        "",
        "## Trusted Level-1 primitives (granted for this run)",
        "",
        "You may call the following fixed functions from inside the evolvable",
        "pair. EvoGrad defines them itself, below `EVOLVE-BLOCK-END`; do NOT",
        "define, redefine or import them, and do NOT put them inside the block.",
        "",
    ]
    for name in names:
        spec = REGISTRY[name]
        lines.append(f"```python\n{spec.source.strip()}\n```")
        lines.append("")
    lines += [
        "How to use them well:",
        "- Use the vendor GEMM for the large dense contractions -- the",
        "  projections in the forward, and the input-gradient and",
        "  weight-gradient contractions in the backward. Those are what cuBLAS",
        "  is best at and what a hand-written Triton matmul loses to.",
        "- Spend your Triton kernels on what cuBLAS cannot do: the per-head",
        "  RMSNorm, RoPE, their fused backward, layout changes and reductions.",
        "- Reshape to 2-D with `.reshape`/`.view` and pass a transposed weight",
        "  as `w.t()`. `.t()` on a 2-D tensor is a metadata-only view and cuBLAS",
        "  consumes it directly -- never materialise a transposed copy.",
        "- Every GEMM must run inside the timed forward or backward. Do NOT",
        "  pre-pack, pre-transpose or cache anything that depends on parameter",
        "  values: the parameters change between calls and a cached product",
        "  would be wrong as well as dishonest.",
        "",
        "Everything else stays forbidden: `torch.matmul`, `@`, `torch.mm` called",
        "directly, `F.linear`, SDPA, eager RMSNorm/RoPE, `torch.compile`,",
        "`torch.autograd`, `.backward()`, and any whole-operator fallback.",
        "",
    ]
    return "\n".join(lines)


def render_pair_rules(op: OpDecl, allowed_primitives=()) -> str:
    no_grad_lines = ""
    no_grad_inputs = tuple(c.name for c in op.tensor_inactive_args())
    if no_grad_inputs:
        names = ", ".join(f"`{n}`" for n in no_grad_inputs)
        no_grad_lines = (
            f"- The following inputs carry NO gradient and must not appear in the "
            f"backward output: {names}. "
            f"The AtenIR graph may include their gradients as outputs — discard them.\n"
        )
    extra = f"\n{op.extra_constraints}" if op.extra_constraints else ""
    primitives = _render_primitive_rules(allowed_primitives)
    return f"""\
## Autograd-pair rules

Public API to implement:

```python
def {op.forward_fn_name}({op.forward_parameters()}):
    return {op.forward_returns()}

def {op.backward_fn_name}({op.backward_parameters()}):
    return {op.backward_returns()}
```

Hard constraints:
- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Put `# EVOLVE-BLOCK-START` before the kernels and `# EVOLVE-BLOCK-END`
  **after the two public pair functions**, so the block spans the kernels,
  the launch helpers AND the pair bodies. The pair bodies are the host
  implementation now, so evolution has to be able to rewrite them together
  with the kernels they launch; a block that stops before them freezes the
  half that decides grids, allocations and saved state.
- **The two functions above ARE the implementation.** Put the allocations,
  shape dispatch, grid choice, kernel launches and the saved-state decision in
  their bodies. Do NOT write a private `_..._impl` twin and forward to it: a
  public function whose whole body is `return _x_impl(...)` will be rejected.
- Do NOT write a `torch.autograd.Function`, a deployment entry point, or a
  `DEPLOYMENT_ENTRY` constant. EvoGrad generates those from the declaration and
  appends them; the argument order, output order, upstream-gradient order and
  input-gradient order are fixed and not yours to choose.
- Reject what the kernel cannot do by raising. An unsupported shape, dtype or
  device must raise, never silently fall back to eager PyTorch, another
  library, or a reference implementation -- a benchmark that measures a
  fallback measures the wrong function.
- `saved_tensors` may be a tensor or tuple/list mixing tensors with immutable
  Python scalar metadata. Only tensors count toward saved-memory usage.
- {op.forward_semantics}
- {op.backward_semantics}
{no_grad_lines}
{_SAVED_TENSOR_GUIDANCE}

{_TRITON_PITFALLS}
{primitives}{extra}"""


def render_plan_prompt(
    *,
    forward: str,
    graph_summary: str,
    lowering_context: str = "",
    op: OpDecl,
    allowed_primitives=(),
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context
        else ""
    )
    return f"""# Autograd-Pair Planning

Forward reference:

```text
{forward}
```

Declared public contract:

```python
def {op.forward_fn_name}({op.forward_parameters()}):
    return {op.forward_returns()}

def {op.backward_fn_name}({op.backward_parameters()}):
    return {op.backward_returns()}
```

Forward semantics: {op.forward_semantics}

Backward semantics: {op.backward_semantics}

The AtenIR backward graph below describes the reference backward semantics, but
the generated implementation is allowed to change the forward/backward contract
by saving forward intermediates.

{graph_summary}
{lowering_section}

{_render_primitive_rules(allowed_primitives)}

Return Markdown with:

1. Initial saved tensor contract and which parts should remain evolvable.
2. Which contractions go to a granted vendor primitive and which arithmetic
   stays in Triton kernels, with the reason for each.
3. Triton kernels for forward and backward.
4. Backward formula and reduction strategy.
5. Expected memory overhead of saved tensors and why it is worth the latency tradeoff.
"""


def render_codegen_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    lowering_context: str = "",
    op: OpDecl,
    allowed_primitives=(),
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context
        else ""
    )
    return f"""# Autograd-Pair Codegen

Generate a complete Python module for an autograd pair for the provided forward reference.

{render_pair_rules(op, allowed_primitives)}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}"""


def render_repair_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str = "",
    op: OpDecl,
    allowed_primitives=(),
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context
        else ""
    )
    return f"""# Autograd-Pair Repair

The generated autograd-pair program failed verification.

Verifier report:

```json
{verifier_report}
```

{render_pair_rules(op, allowed_primitives)}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}

## Previous code

```python
{previous_code}
```

Return only the repaired Python source. No Markdown.
"""
