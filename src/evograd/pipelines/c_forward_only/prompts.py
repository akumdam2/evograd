"""Prompt templates for Pipeline C (forward-source-only ablation).

Ported from ``pipeline/no_atenir_fusion_agent/prompts.py``. It consumes the
:class:`OpDecl` directly; the LayerNorm-verbatim fallback block is gone.
"""

from __future__ import annotations

from evograd.opdecl.activity import OpDecl

SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair directly from a PyTorch forward
reference, without an AtenIR backward graph. Correctness is more important than
speed."""


_GENERIC_GUIDANCE_AND_PITFALLS = """Saved-tensor guidance:

- The saved tensor tuple is part of the evolvable program state. You may save
  original inputs and/or forward-computed intermediates.
- Prefer compact saved state such as per-row statistics over full
  activation-sized tensors when the backward can cheaply reconstruct them.
- Avoid saving tensors with the same shape as a large activation unless the
  latency benefit clearly outweighs the memory cost.
- The evaluator reports saved memory so OpenEvolve can optimize the tradeoff.

Triton pitfalls:

- `tl.arange` bounds must be compile-time constants. Use a `BLOCK_*:
  tl.constexpr` meta-parameter and mask inactive lanes.
- Do not read Python globals inside `@triton.jit` kernels. Pass eps, dimensions,
  strides, and other scalars as kernel arguments or meta-parameters.
- Do not hard-code traced dimensions such as 64, 128, or 1024 unless they are
  semantic constants. Derive runtime dimensions from input tensors.
- Use fp32 accumulation for reductions.
- Output dtypes must match the public API contract.
- Allocate outputs with `torch.empty_like` / `torch.zeros_like` where possible.
"""


def render_pair_rules(op: OpDecl) -> str:
    no_grad = ""
    no_grad_inputs = tuple(c.name for c in op.tensor_inactive_args())
    if no_grad_inputs:
        names = ", ".join(f"`{n}`" for n in no_grad_inputs)
        no_grad = (
            f"- The following inputs are not trainable; do not return gradients for "
            f"them: {names}. The evaluator wrapper inserts None in their positions.\n"
        )
    extra = f"\n{op.extra_constraints}\n" if op.extra_constraints else ""
    return f"""## Autograd-pair rules

Public API to implement:

```python
def {op.forward_fn_name}({op.forward_parameters()}):
    return y, saved_tensors

def {op.backward_fn_name}({op.backward_parameters()}):
    return {op.backward_returns()}
```

Hard constraints:

- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- Do not call PyTorch autograd or a high-level PyTorch reference of this operator
  in the generated math.
- {op.forward_semantics}
- {op.backward_semantics}
- Backward must consume only the upstream gradient, `saved_tensors`, and the
  declared scalar Inactive arguments (when the forward takes them).
- `saved_tensors` may be a tensor or tuple/list mixing tensors with immutable
  Python scalar metadata. Only tensors count toward saved-memory usage.
{no_grad}
{_GENERIC_GUIDANCE_AND_PITFALLS}{extra}"""


def render_plan_prompt(
    *,
    forward: str,
    forward_source: str,
    op: OpDecl,
) -> str:
    return f"""# No-AtenIR Autograd-Pair Planning

This is an ablation. You must derive the forward saved-tensor contract and
backward semantics directly from the forward reference below. You are not given
an AtenIR backward graph.

Forward reference spec:

```text
{forward}
```

Forward reference source:

```python
{forward_source}
```

Task:

1. Derive the mathematical backward formula from the forward source.
2. Propose a saved tensor contract for `{op.forward_fn_name}`.
3. Propose Triton kernels for forward and backward.
4. Identify reductions, reduction axes, accumulation dtypes, and output dtypes.
5. Explain the memory/speed tradeoff of the saved tensors.

Do not use or assume any AtenIR backward graph. Return Markdown only.
"""


def render_codegen_prompt(
    *,
    forward: str,
    forward_source: str,
    plan: str,
    op: OpDecl,
) -> str:
    return f"""# No-AtenIR Autograd-Pair Codegen

Generate a complete Python module implementing the autograd-pair API.
This is an ablation. You must derive the implementation from the forward source
and the plan only. No AtenIR backward graph is provided.

Requirements:

{render_pair_rules(op)}

Forward reference spec:

```text
{forward}
```

Forward reference source:

```python
{forward_source}
```

Plan:

```markdown
{plan}
```
"""


def render_repair_prompt(
    *,
    forward: str,
    forward_source: str,
    plan: str,
    previous_code: str,
    verifier_report: str,
    op: OpDecl,
) -> str:
    return f"""# No-AtenIR Autograd-Pair Repair

The generated autograd-pair program failed verification. Repair it using only
the forward source, previous plan, and verifier report. No AtenIR backward graph
is available.

Verifier report:

```json
{verifier_report}
```

Forward reference spec:

```text
{forward}
```

Forward reference source:

```python
{forward_source}
```

Plan:

```markdown
{plan}
```

Previous code:

```python
{previous_code}
```

{render_pair_rules(op)}

Before writing the repaired code, internally classify the failure as a formula
error, saved-tensor contract error, reduction-axis error, shape/tiling error,
dtype/casting error, or Triton compile-time error. Then fix the root cause.

Return only repaired Python source code, no Markdown.
"""
