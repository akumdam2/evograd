import ast
from pathlib import Path
import pytest
import torch
from evograd.benchmark import TASKS
from evograd.opdecl.inputs import as_output_tuple, make_case_inputs
from evograd.opdecl.oracle import resolve_runtime_forward
from evograd.evaluation.tier3.patch import eager_pair_for, kernel_from_pair
from tests._small_cases import SMALL


@pytest.mark.parametrize(
    "name",
    [
        "convnextv2_tiny_convnext_mlp",
        "convnextv2_tiny_dwconv_layernorm",
        "qwen2_vl_2b_text_attention_block",
        "qwen2_vl_2b_text_mlp",
        "qwen2_vl_2b_text_residual_rmsnorm",
        "qwen2_vl_2b_vision_attention",
        "qwen2_vl_2b_vision_mlp",
        "qwen2_vl_2b_vision_residual_layernorm",
        "spatial_conv2d",
        "grn",
        "gelu",
        "quick_gelu",
    ],
)
def test_pair_binding_all_gradients(name):
    op = TASKS[name]
    active = {a.name for a in op.active_args()}
    for case in SMALL[name]:
        if case.dtype != "float32":
            continue
        values = make_case_inputs(op, case, device="cpu")
        args = [
            (
                values[a.name].detach().requires_grad_(True)
                if a.name in active
                else values[a.name]
            )
            for a in op.args
        ]
        fn = resolve_runtime_forward(op)
        expected = as_output_tuple(op, fn(*args))
        upstream = [values[n] for n in op.upstream_grad_names]
        leaves = [a for a, d in zip(args, op.args) if d.name in active]
        ref = torch.autograd.grad(expected, leaves, upstream)
        actual = as_output_tuple(op, kernel_from_pair(op, eager_pair_for(op))(*args))
        got = torch.autograd.grad(actual, leaves, upstream)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)
        for a, b in zip(ref, got):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", ["convnextv2_tiny_dwconv_layernorm",
                                  "qwen2_vl_2b_text_attention_block",
                                  "qwen2_vl_2b_text_residual_rmsnorm",
                                  "qwen2_vl_2b_vision_attention",
                                  "qwen2_vl_2b_vision_residual_layernorm"])
def test_the_reference_an_agent_reads_is_the_declared_forward(name):
    """The standalone reference closure (what a task pack shows) computes the same."""
    from evograd.opdecl.importing import reference_closure

    op = TASKS[name]
    source, _ = reference_closure(op.forward)
    namespace: dict = {}
    exec(compile(source, "reference", "exec"), namespace)
    for case in SMALL[name]:
        values = make_case_inputs(op, case, device="cpu")
        args = [values[a.name] for a in op.args]
        ours = as_output_tuple(op, namespace["reference_forward"](*args))
        declared = as_output_tuple(op, resolve_runtime_forward(op)(*args))
        assert all(torch.equal(a, b) for a, b in zip(ours, declared))


def test_no_reference_dependency():
    roots = [
        Path("src/evograd/benchmark/topdown/common/vision"),
        Path("src/evograd/evaluation/tier3/workloads/common/vision"),
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    assert not any(n.name.startswith("references") for n in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("references")


def test_observed_cases_reproduce_strides(monkeypatch):
    from .test_models import small_spec
    from evograd.benchmark.topdown.common.vision.harvest import collect, snapshot
    from evograd.benchmark.topdown.common.vision import contracts

    payload = snapshot(collect(small_spec("convnextv2_tiny")))
    monkeypatch.setattr(contracts, "has_snapshot", lambda _: True)
    monkeypatch.setattr(contracts, "load_snapshot", lambda _: payload)
    op = contracts.declaration("convnextv2_tiny", "convnext_mlp")
    assert len(op.benchmark) == 4
    for case in op.benchmark:
        inputs = make_case_inputs(op, case, device="cpu")
        for name, stride in case.provenance.strides:
            assert inputs[name].stride() == stride
        assert case.provenance.source == "execution"


@pytest.mark.parametrize("model", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_primitive_cases_have_valid_shapes_and_eps(monkeypatch, model):
    from .test_models import small_spec
    from evograd.benchmark.topdown.common.vision.harvest import collect, snapshot
    from evograd.benchmark.cases import observed_workloads
    from evograd.benchmark import topdown
    from evograd.ops import PRIMITIVES

    payload = snapshot(collect(small_spec(model)))
    monkeypatch.setattr(topdown, "load_snapshot", lambda _: payload)
    for name, entry in payload["level1"].items():
        op = PRIMITIVES[name]
        cases = observed_workloads(model, name, op=op)
        for case in cases:
            values = make_case_inputs(op, case, device="cpu")
            args = [values[a.name] for a in op.args]
            result = resolve_runtime_forward(op)(*args)
            assert result.shape == values[op.outputs[0].grad_name].shape
            if name in ("layernorm", "rmsnorm"):
                assert values["eps"] == case.provenance.free["eps"]
