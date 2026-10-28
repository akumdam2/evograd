"""The shared problem statement every agent is prompted with.

Prompt parity is constructed rather than asserted:

* Section 1 is the evograd agent's own OpenEvolve system message for this
  operator, produced by ``evograd.evolve.run.render_config`` -- the function
  ``evograd evolve`` renders it with -- and copied verbatim. Its SHA-256 is
  recorded in ``task.json`` so a run can show it was not edited.
* Section 2 replaces what evograd's seed carries. The evograd LLM never sees
  the forward reference: it sees a working program that computes it. An agent
  without the frontend has no program, so it is given the reference source the
  frontend consumes, the fixed configuration, and the deliverable.
* Section 3 states the rules the evaluator enforces statically on every
  candidate, read from the enforcement code (``pipelines.shared.primitives``)
  so the text cannot drift from what is checked. evograd's seed already obeys
  them; a from-scratch agent would otherwise learn them only by failing.
* Section 4 is how the evaluator scores, which evograd's system message
  summarizes and which this repeats with the configuration's numbers.

Agents embed this text in their own prompt slot (GEPA's ``background``,
KernelAgent's ``problem_description``, GEAK's op spec and task steer,
OpenEvolve's system message). Nothing agent-specific is added here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evograd.adbench_tools.task import AgentTask


def system_message(task: "AgentTask") -> str:
    """evograd's OpenEvolve system message for this task's scoped operator."""
    import yaml

    from evograd.evolve.run import render_config

    config = yaml.safe_load(render_config(task.op()))
    return config["prompt"]["system_message"].strip() + "\n"


def static_rules() -> dict[str, list[str]]:
    """The constructs ``check_source`` rejects, read from its own tables."""
    from evograd.pipelines.shared import primitives

    return {
        "torch_calls": sorted(primitives._TORCH_DENIED),
        "torch_prefixes": sorted(".".join(("torch", *p)) for p in primitives._DENIED_TORCH_PREFIXES),
        "roots": sorted(primitives._DENIED_ROOTS),
        "methods": sorted(primitives._DENIED_METHODS),
    }


def reserved_names(task: "AgentTask") -> list[str]:
    """Names the harness defines around the agent's module."""
    from evograd.pipelines.shared.artifact import ArtifactContract
    from evograd.pipelines.shared.primitives import REGISTRY

    contract = ArtifactContract(task.op())
    names = {contract.function_cls, contract.module_cls, contract.entry,
             "DEPLOYMENT_ENTRY", "ARTIFACT_CONTRACT"}
    for spec in REGISTRY.values():
        names.update(spec.symbols)
    return sorted(names)


def _argument_lines(abi: dict) -> list[str]:
    lines = []
    for entry in abi["arguments"]:
        if entry["kind"] == "scalar":
            lines.append(f"- `{entry['name']}`: Python scalar, default {entry['default']!r} (inactive)")
            continue
        role = (f"active, receives gradient `{entry['grad']}`" if entry["kind"] == "active"
                else "inactive tensor, no gradient")
        size = ", ".join(str(v) for v in entry["size"])
        lines.append(f"- `{entry['name']}`: {entry['shape']} = [{size}], {entry['dtype']} ({role})")
    for entry in abi["outputs"]:
        size = ", ".join(str(v) for v in entry["size"])
        lines.append(f"- output `{entry['name']}`: {entry['shape']} = [{size}], {entry['dtype']}"
                     f" (upstream gradient `{entry['grad']}`)")
    return lines


def _tolerance_text(gate1: dict) -> str:
    """Gate 1's acceptance rule, from ``task.json``'s ``gate1_tolerances``."""
    results = "\n".join(
        f"- `{name}` ({r['dtype']}): atol={r['atol']:.3g}, rtol={r['rtol']:.3g}"
        for name, r in gate1["results"].items()
    )
    gamma = gate1.get("envelope_gamma")
    if gamma is None:
        return f"""Correctness tolerances at this configuration, per output and gradient
(`|candidate - oracle| <= atol + rtol * |oracle|`, elementwise):

{results}"""
    return f"""Acceptance compares every output and gradient with the reference, element by
element in float64, on real recorded training calls held out from you:
`|candidate - reference| <= max(atol + rtol * |reference|, envelope)`.
`atol` and `rtol` are the `torch.testing.assert_close` defaults for the result's
dtype (below); `envelope` is {gamma:g}x the larger of the largest elementwise
deviation of `torch.compile` from the reference on that result, measured on other
recorded calls, and one unit in the last place of the reference's largest
magnitude. So each element may be off by the dtype default or by up to {gamma:g}x
torch.compile's own error or {gamma:g} ulps of the largest value, whichever is
larger.

{results}"""


def _search_text(spec: dict) -> str:
    """How the evaluator checks correctness during search (``search_correctness``)."""
    from evograd.verification.payloads.gate1 import SEARCH_RULE

    search = spec.get("search_correctness") or {}
    if search.get("rule") == SEARCH_RULE:
        return (f" Its correctness check applies the acceptance rule of section 2 to "
                f"{len(search['payload_ids'])} recorded training calls (the first, middle and "
                "last call of this operator in each of two training steps); the calls "
                "acceptance judges are not among them.")
    return ""


def render_statement(task: "AgentTask") -> str:
    spec = task.spec
    abi = spec["abi"]
    rules = static_rules()
    workload = spec["workload"]
    dims = ", ".join(f"{k}={v}" for k, v in workload["dims"].items())
    tolerances = _tolerance_text(spec["gate1_tolerances"])
    granted = spec["allowed_primitives"]
    if granted:
        from evograd.pipelines.shared.primitives import GRANT_SCOPE, REGISTRY, describe_grant

        symbols = [symbol for name in granted for symbol in REGISTRY[name].symbols]
        library = "\n".join(
            f"  - {line}"
            for line in describe_grant(granted, "The harness defines it outside your module")
        ) + (
            f"\n  - {GRANT_SCOPE}"
            "\n  - To run your module on its own, add `from adbench_primitives import "
            f"{', '.join(symbols)}` (the task pack ships that file); the harness removes "
            "that line and supplies its own definition."
        )
    else:
        library = "  - none. Every contraction must be written in Triton (`tl.dot`)."
    reserved = ", ".join(f"`{name}`" for name in reserved_names(task))

    return f"""# Task {spec['task_id']}

Write a training-ready forward/backward pair for `{spec['op']}` that is
faster than PyTorch eager autograd at the configuration below, and correct
within the tolerances in section 2.

## 1. Instructions (identical to the evograd agent's)

{system_message(task)}
## 2. What you are given

There is no working starting program (at most an unimplemented skeleton):
where section 1 says "the code inside the EVOLVE-BLOCK", read "the module you
write" (section 3). The operator is
defined by the PyTorch forward reference below and nothing else; the
gradients you return are checked against `torch.autograd` applied to it.

```python
{task.reference_source.rstrip()}
```

Fixed configuration: {dims}, dtype {workload['dtype']}, one CUDA device.

{chr(10).join(_argument_lines(abi))}

{tolerances}

## 3. Deliverable and rules

Return ONE self-contained Python module (source code only) that defines, at
module level:

```python
{abi['forward_signature']}:
    return {abi['forward_returns']}

{abi['backward_signature']}:
    return {abi['backward_returns']}
```

- `saved_tensors` is a tuple of tensors and/or immutable Python scalars; the
  backward receives exactly what the forward saved. Gradients are returned in
  the order `{', '.join(abi['grad_order'])}`, each with its input's shape and
  dtype; inactive arguments get no gradient.
- Write the math as Triton kernels (`@triton.jit`) launched from these two
  functions and/or PyTorch tensor operations not listed below. Allocation,
  shape inspection, views and launches in plain PyTorch are fine.
- The harness wires autograd and the deployment entry from the declaration.
  Do not write a `torch.autograd.Function` or an `nn.Module`, and do not define
  these reserved names: {reserved}.
- Every line of your module is checked statically. Rejected:
  - calls `torch.<name>` for: {', '.join(rules['torch_calls'])};
  - any private `torch._*` entry point, and anything under {', '.join(rules['torch_prefixes'])};
  - imports of or calls into: {', '.join(rules['roots'])};
  - methods `.<name>()` on any object for: {', '.join(rules['methods'])};
  - the `@` operator.
- Library primitives this task grants:
{library}

## 4. How it is scored

The same evaluator scores every agent. A candidate that fails the static
rules, the import, any output or gradient check, or a run at the timed shape
scores as incorrect.{_search_text(spec)} A correct candidate is timed at the configuration above
against the `{spec['performance_baseline']}` baseline: forward, backward from
saved state, and the forward+backward step, with saved-tensor bytes reported
relative to input bytes. `combined_score` is evograd's `{spec['scoring']}`
policy over those measurements -- higher is better.
"""
