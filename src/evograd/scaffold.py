"""Synthesize a declaration-native operator contract from one PyTorch forward."""

from __future__ import annotations

import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from evograd.opdecl.importing import resolve_callable


@dataclass(frozen=True)
class ScaffoldResult:
    name: str
    forward: str
    declaration: Path


_SYSTEM = """You design typed contracts for differentiable PyTorch operators.
Return exactly one JSON object. You provide semantic facts and valid input
generation; code will mechanically build the oracle and declaration. Never
write a backward implementation."""


def _prompt(name: str, forward: str, source: str, signature: str) -> str:
    return f"""Create an evograd operator contract for `{name}`.

Forward reference: `{forward}`
Signature: `{signature}`

```python
{source}
```

Return this JSON shape:
{{
  "dims": ["symbolic", "dimension", "names"],
  "args": [
    {{"name": "x", "kind": "active_tensor", "shape": "[rows, cols]", "dtype": null}},
    {{"name": "target", "kind": "inactive_tensor", "shape": "[rows]", "dtype": "int64"}},
    {{"name": "eps", "kind": "inactive_scalar", "default": 1e-5}}
  ],
  "output": {{"name": "out", "shape": "[rows, cols]"}},
  "input_body": "Python assignments for every tensor arg using torch, device, dtype and named dims",
  "forward_semantics": "precise forward math",
  "backward_semantics": "which active gradients are required and the backward math",
  "correctness": [
    {{"dims": {{"rows": 8, "cols": 64}}, "dtype": "float32"}},
    {{"dims": {{"rows": 16, "cols": 128}}, "dtype": "float16"}}
  ],
  "benchmark": [
    {{"dims": {{"rows": 256, "cols": 1024}}, "dtype": "bfloat16"}}
  ],
  "regime_dim": "cols or rows or null",
  "regime_split": 4096,
  "reduced_grads": ["names such as dweight whose reduction error grows with a dimension"]
}}

Rules:
- args must exactly match the Python signature and order.
- active_tensor means a gradient is requested. Labels/targets/masks are inactive.
- scalar args must have their exact signature defaults.
- shapes use only declared dims and integer literals; use [] for a scalar tensor/output.
- input_body assigns tensors only and has no return statement. Generate semantically valid
  probabilities, log-probabilities, labels, masks, etc. Use torch (not torch_module).
- provide at least two correctness dtypes when the operation supports them and 12-16 realistic,
  bounded benchmark cases with non-power-of-two coverage.
- regime_dim describes the backward's structural split. Use null for a single regime. If non-null,
  put at least four benchmark shapes on each side of regime_split.
- do not import modules or implement an oracle."""


def _json_object(response: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
    raw = match.group(1) if match else response[response.find("{") : response.rfind("}") + 1]
    return json.loads(raw)


def _validate_spec(name: str, signature: inspect.Signature, spec: dict) -> None:
    dims = tuple(spec["dims"])
    if not dims or any(not dim.isidentifier() for dim in dims):
        raise ValueError("dims must be non-empty Python identifiers")
    args = spec["args"]
    expected = list(signature.parameters)
    actual = [arg["name"] for arg in args]
    if actual != expected:
        raise ValueError(f"args {actual} do not match signature {expected}")
    allowed = {"active_tensor", "inactive_tensor", "inactive_scalar"}
    for arg in args:
        if arg["kind"] not in allowed:
            raise ValueError(f"unsupported arg kind {arg['kind']!r}")
        if arg["kind"] == "inactive_scalar":
            parameter = signature.parameters[arg["name"]]
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(
                    f"scalar {arg['name']} needs a signature default"
                )
            arg["default"] = parameter.default
        elif "shape" not in arg:
            raise ValueError(f"tensor {arg['name']} has no shape")
    for collection in ("correctness", "benchmark"):
        if not spec.get(collection):
            raise ValueError(f"{collection} workloads are empty")
        for case in spec[collection]:
            if set(case["dims"]) != set(dims):
                raise ValueError(
                    f"{collection} dims {sorted(case['dims'])} != {sorted(dims)}"
                )
            if case["dtype"] not in ("float32", "float16", "bfloat16"):
                raise ValueError(f"unsupported workload dtype {case['dtype']}")
    regime_dim = spec.get("regime_dim")
    if regime_dim is not None:
        if regime_dim not in dims or float(spec.get("regime_split", 0)) <= 0:
            raise ValueError("regime_dim/split is invalid")


def _source(name: str, forward: str, spec: dict) -> str:
    dims = tuple(spec["dims"])
    arg_lines = []
    tensor_names = []
    active_names = []
    scalar_defaults = {}
    for arg in spec["args"]:
        kind = arg["kind"]
        if kind == "active_tensor":
            active_names.append(arg["name"])
            tensor_names.append(arg["name"])
            arg_lines.append(
                f"        Active({arg['name']!r}, {arg['shape']!r}, "
                f"dtype={arg.get('dtype')!r}),"
            )
        elif kind == "inactive_tensor":
            tensor_names.append(arg["name"])
            arg_lines.append(
                f"        Inactive({arg['name']!r}, {arg['shape']!r}, "
                f"dtype={arg.get('dtype')!r}),"
            )
        else:
            scalar_defaults[arg["name"]] = arg["default"]
            arg_lines.append(
                f"        Inactive({arg['name']!r}, default={arg['default']!r}),"
            )

    workload_lines = {}
    for collection in ("correctness", "benchmark"):
        workload_lines[collection] = "\n".join(
            f"        Workload(dims={case['dims']!r}, dtype={case['dtype']!r}),"
            for case in spec[collection]
        )
    dim_bindings = "\n".join(
        f"    {dim} = workload.dims[{dim!r}]" for dim in dims
    )
    input_body = textwrap.indent(spec["input_body"].strip(), "    ")
    returned = [f"{name!r}: {name}" for name in tensor_names]
    returned.extend(f"{name!r}: {value!r}" for name, value in scalar_defaults.items())
    output_name = spec["output"]["name"]
    returned.append(f"{('d' + output_name)!r}: dout")
    memory_inputs = tuple(active_names)

    regime_dim = spec.get("regime_dim")
    regime_source = ""
    regime_args = ""
    if regime_dim is not None:
        split = float(spec["regime_split"])
        regime_source = f"""
def _regime_feature(workload):
    return float(workload.dims[{regime_dim!r}])


def _case_weight(workload):
    import math
    distance = abs(math.log2(_regime_feature(workload)) - math.log2({split!r}))
    return float(max(1.0, round(distance)))


_SMALL = tuple(case for case in _BENCHMARK if _regime_feature(case) < {split!r})
_LARGE = tuple(case for case in _BENCHMARK if _regime_feature(case) >= {split!r})
"""
        regime_args = f"""
    benchmark_suites={{"full": _BENCHMARK, "small": _SMALL, "large": _LARGE}},
    regime_feature=_regime_feature,
    regime_split={split!r},
    case_weight=_case_weight,"""

    reduced = tuple(spec.get("reduced_grads") or ())
    tolerance_source = ""
    tolerance_arg = ""
    if reduced:
        tolerance_source = f"""
_REDUCED_GRADS = {reduced!r}


def _tolerance(workload, result_name, atol, rtol):
    if result_name in _REDUCED_GRADS:
        scale = max(1.0, (max(workload.dims.values()) / 64.0) ** 0.5)
        return atol * scale, rtol
    return atol, rtol
"""
        tolerance_arg = "\n    tolerance_hook=_tolerance,"

    return f'''"""Generated declaration for {name}; review before publishing."""

from evograd.opdecl import Active, Inactive, Workload, bind_shape, declare_op


def _make_inputs(torch, op, workload, device="cuda"):
    dtype = getattr(torch, workload.dtype)
{dim_bindings}
{input_body}
    output_shape = bind_shape({spec["output"]["shape"]!r}, workload.dims)
    dout = torch.randn(output_shape, device=device, dtype=dtype)
    return {{{", ".join(returned)}}}


_CORRECTNESS = (
{workload_lines["correctness"]}
)
_BENCHMARK = (
{workload_lines["benchmark"]}
)
{regime_source}
{tolerance_source}
op = declare_op(
    name={name!r},
    forward={forward!r},
    declaration=f"{{__file__}}:op",
    dims={dims!r},
    args=(
{chr(10).join(arg_lines)}
    ),
    output=Active({output_name!r}, {spec["output"]["shape"]!r}),
    forward_semantics={spec["forward_semantics"]!r},
    backward_semantics={spec["backward_semantics"]!r},
    correctness=_CORRECTNESS,
    benchmark=_BENCHMARK,{regime_args}
    tolerances={{
        "float32": (2e-5, 2e-5),
        "float16": (5e-2, 5e-2),
        "bfloat16": (8e-2, 8e-2),
    }},
    memory_inputs={memory_inputs!r},{tolerance_arg}
    make_inputs=_make_inputs,
)
'''


def synthesize_declaration(
    name: str,
    forward: str,
    *,
    output_dir: Path,
    model: str = "gpt-5.5",
    api_base: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    max_attempts: int = 4,
) -> ScaffoldResult:
    """Generate, import, validate, and smoke the new declaration."""
    fn = resolve_callable(forward)
    signature = inspect.signature(fn)
    source = textwrap.dedent(inspect.getsource(fn))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(name, forward, source, f"{fn.__name__}{signature}")

    from evograd.pipelines.shared.llm_client import (
        generate_with_openai_compatible_api,
    )

    failure = ""
    declaration = (output_dir / "declaration.py").resolve()
    for attempt in range(1, max_attempts + 1):
        attempt_dir = output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        current_prompt = prompt
        if failure:
            current_prompt += (
                "\n\nThe previous contract failed validation. Correct this failure:\n"
                + failure[-3000:]
            )
        (attempt_dir / "prompt.md").write_text(current_prompt, encoding="utf-8")
        response = generate_with_openai_compatible_api(
            prompt=current_prompt,
            system_message=_SYSTEM,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=12000,
            temperature=0.2,
            timeout=240,
        )
        (attempt_dir / "response.txt").write_text(response, encoding="utf-8")
        try:
            spec = _json_object(response)
            _validate_spec(name, signature, spec)
            declaration.write_text(_source(name, forward, spec), encoding="utf-8")
            from evograd.ops import load_op

            op = load_op(f"{declaration}:op")
            if op.name != name:
                raise ValueError(f"generated name {op.name!r} != {name!r}")
            try:
                import torch

                from evograd.opdecl.inputs import make_case_inputs
                from evograd.opdecl.oracle import oracle

                device = "cuda" if torch.cuda.is_available() else "cpu"
                values = make_case_inputs(op, op.correctness[0], device=device)
                oracle(op, values)
            except ImportError:
                pass
            (output_dir / "contract.json").write_text(
                json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
            )
            return ScaffoldResult(name=name, forward=forward, declaration=declaration)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            (attempt_dir / "failure.txt").write_text(failure, encoding="utf-8")
    raise RuntimeError(
        f"could not synthesize a valid declaration for {name} after "
        f"{max_attempts} attempts: {failure}"
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--forward", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args(argv)
    from evograd.api import _file_forward_spec

    forward = _file_forward_spec(args.forward)
    result = synthesize_declaration(
        args.op,
        forward,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_attempts=args.max_attempts,
    )
    print(result.declaration)
    return 0
