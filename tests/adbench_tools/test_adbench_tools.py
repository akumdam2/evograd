"""The agent-neutral comparison harness, without a GPU, an LLM or an agent.

What these pin is what makes the comparison fair: the task pack is a model's
observed case under the model's reference, the statement carries evograd's
system message verbatim, the reference an agent reads computes exactly the
declared forward, every line an agent writes is under the evaluator's static
rules, search feedback is Gate 1 on the case's payloads, and nothing outside
the evolve block survives into the final measurement.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from evograd.adbench_tools.artifact import (  # noqa: E402
    END,
    PackagingError,
    extract_code,
    package,
    unpackage,
)
from evograd.adbench_tools.evaluate import fixed_code_intact, prepare_program  # noqa: E402
from evograd.adbench_tools.score import compare  # noqa: E402
from evograd.adbench_tools.statement import system_message  # noqa: E402
from evograd.adbench_tools.task import build_task, load, reference_closure  # noqa: E402
from tests._small_cases import SMALL

REPO = Path(__file__).resolve().parents[2]

#: A correct PyTorch pair for the linear task (torchrec_dlrm's dense layer):
#: tensor ops plus the granted vendor_gemm, which is all the static rules allow
#: besides Triton.
LINEAR = '''
import torch


def linear_forward_with_saved(x, weight, bias):
    return vendor_gemm(x, weight.t()) + bias, (x, weight)


def linear_backward_from_saved(dy, saved_tensors):
    x, weight = saved_tensors
    return vendor_gemm(dy, weight), vendor_gemm(dy.t(), x), dy.sum(0)
'''

#: The case behind ``small_task``: a dense layer small enough for a CPU.
SMALL_MODEL, SMALL_CASE = "torchrec_dlrm", "linear/dense_layers"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear_payloads(seed: int = 0):
    """Recorded-call-like payloads of :data:`SMALL_CASE`: three calls in each of the
    two calibration orders, at the case's shapes."""
    g = torch.Generator().manual_seed(seed)
    return [(call, order, {"x": torch.randn(32, 13, generator=g),
                           "weight": 0.3 * torch.randn(512, 13, generator=g),
                           "bias": torch.randn(512, generator=g)},
             {"dy": torch.randn(32, 512, generator=g)})
            for order in (101, 202) for call in "abc"]


@pytest.fixture(scope="module")
def small_task(tmp_path_factory):
    return payload_rule_task(tmp_path_factory.mktemp("small") / "task", SMALL_MODEL, SMALL_CASE,
                             payloads=linear_payloads())


# ── the task pack ────────────────────────────────────────────────────────────


def test_the_statement_carries_evograds_system_message_verbatim(small_task):
    message = system_message(small_task)
    statement = small_task.statement
    assert message in statement
    assert small_task.spec["system_message_sha256"] == hashlib.sha256(message.encode()).hexdigest()
    assert small_task.spec["statement_sha256"] == hashlib.sha256(statement.encode()).hexdigest()
    assert small_task.reference_source.strip() in statement


def test_a_task_pack_is_never_overwritten(small_task):
    with pytest.raises(FileExistsError):
        build_task(small_task.root, model=SMALL_MODEL, case_id=SMALL_CASE)


@pytest.mark.parametrize("name", ["llama3_swiglu_mlp", "llama3_attention_block", "layernorm",
                                  "rmsnorm", "qwen3_attention_block", "qwen3_lm_head_ce"])
def test_the_reference_an_agent_reads_is_the_declared_forward(name):
    from evograd.benchmark import get_task
    from evograd.opdecl.importing import resolve_callable
    from evograd.opdecl.inputs import make_case_inputs

    op = get_task(name)
    source, _ = reference_closure(op.forward)
    namespace: dict = {}
    exec(compile(source, "reference", "exec"), namespace)
    workload = next(w for w in SMALL[name] if w.dtype == "float32")
    values = make_case_inputs(op, workload, device="cpu")
    args = [values.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
    ours, declared = namespace["reference_forward"](*args), resolve_callable(op.forward)(*args)
    ours = ours if isinstance(ours, tuple) else (ours,)
    declared = declared if isinstance(declared, tuple) else (declared,)
    assert all(torch.equal(a, b) for a, b in zip(ours, declared))


# ── packaging ────────────────────────────────────────────────────────────────


def test_packaging_puts_the_whole_module_under_the_static_rules(small_task, tmp_path):
    from evograd.pipelines.shared.primitives import check_source

    artifact = package(small_task, extract_code(f"Sure:\n```python\n{LINEAR}```\n", small_task))
    usage = check_source(artifact, allowed=small_task.allowed_primitives)
    assert usage["primitive_call_sites"]["vendor_gemm"] == 3
    assert unpackage(artifact).strip() == LINEAR.strip()
    with pytest.raises(Exception, match="matrix-multiply"):
        check_source(package(small_task, LINEAR + "\nZ = 1 @ 2\n"),
                     allowed=small_task.allowed_primitives)


@pytest.mark.parametrize("bad, message", [
    ("def ARTIFACT_CONTRACT():\n    pass\n", "reserves"),
    ("def vendor_gemm(a, b):\n    return a\n", "reserves"),
    ("# EVOLVE-BLOCK-START\n", "markers"),
    ("from adbench_primitives import vendor_gemm as gemm\n", "only"),
    ("import adbench_primitives\n", "may only be imported"),
])
def test_packaging_refuses_what_the_harness_owns(small_task, bad, message):
    with pytest.raises(PackagingError, match=message):
        package(small_task, LINEAR + "\n" + bad)


def test_packaging_requires_the_pair(small_task):
    with pytest.raises(PackagingError, match="must define"):
        package(small_task, "import torch\n")


def test_the_standalone_primitive_import_is_replaced(small_task):
    artifact = package(small_task, "from __future__ import annotations\n"
                                   "from adbench_primitives import vendor_gemm\n" + LINEAR)
    assert "adbench_primitives" not in artifact
    assert artifact.index("from __future__ import annotations") < artifact.index("import torch")


def test_code_outside_the_block_never_reaches_the_final_measurement(small_task, tmp_path):
    artifact = package(small_task, LINEAR, origin="openevolve")
    assert fixed_code_intact(small_task, artifact) is True
    tampered = artifact.replace("DEPLOYMENT_ENTRY = ", "BACKDOOR = 1\nDEPLOYMENT_ENTRY = ")
    assert tampered.split(END, 1)[1] != artifact.split(END, 1)[1]
    assert fixed_code_intact(small_task, tampered) is False
    rebuilt = prepare_program(small_task, tampered, tmp_path, origin="final", repackage=True)
    assert "BACKDOOR" not in rebuilt.read_text()


def test_a_packaged_pair_passes_the_packs_search_rule(small_task, tmp_path, monkeypatch):
    """Gate 1 on the case's six calibration payloads, through the evaluator's own check."""
    pytest.importorskip("triton")
    from evograd.evolve.evaluator import _load_module, _run_correctness
    from evograd.verification.payloads.gate1 import SEARCH_CORRECTNESS_ENV

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setenv(SEARCH_CORRECTNESS_ENV, str(small_task.root / "task.json"))
    path = tmp_path / "artifact.py"
    path.write_text(package(small_task, LINEAR))
    report = _run_correctness(small_task.op(), _load_module(str(path)), device="cpu")
    assert report["status"] == "passed"
    assert report["passed"] == report["total"] == 6, report["reports"]
    wrong = tmp_path / "wrong.py"
    wrong.write_text(package(small_task, LINEAR.replace("dy.sum(0)", "1.01 * dy.sum(0)")))
    report = _run_correctness(small_task.op(), _load_module(str(wrong)), device="cpu")
    assert report["status"] == "failed" and report["dbias_correctness"] == 0.0


# ── results ──────────────────────────────────────────────────────────────────


def test_compare_refuses_results_from_different_tasks(tmp_path):
    def result(name, task_id):
        path = tmp_path / name / "final.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"agent": name, "task_id": task_id, "statement_sha256": "s",
                                    "accepted": True, "correct": True, "combined_score": 1.0,
                                    "metrics": {}, "run": {}}))
        return path

    table = compare([result("a", "t"), result("b", "t")])
    assert "| a |" in table and "| b |" in table
    with pytest.raises(ValueError, match="different tasks"):
        compare([result("c", "t"), result("d", "u")])


# ── agent glue (candidates/) ─────────────────────────────────────────────────


def test_kernelagent_and_geak_time_the_same_step_as_the_pair(small_task, tmp_path):
    sys.path.insert(0, str(REPO / "candidates"))
    try:
        kernelagent = _module(REPO / "candidates/kernelagent/run.py", "adbench_tools_kernelagent")
        geak = _module(REPO / "candidates/geak/prepare.py", "adbench_tools_geak_prepare")
    finally:
        sys.path.remove(str(REPO / "candidates"))
    from evograd.opdecl.inputs import make_case_inputs

    (tmp_path / "problem.py").write_text(kernelagent.render_problem(small_task))
    compile(kernelagent.render_test(small_task, tmp_path), "test", "exec")
    problem = _module(tmp_path / "problem.py", "adbench_tools_problem")
    namespace: dict = {}
    sys.path.insert(0, str(small_task.root))
    try:
        exec(compile("from adbench_primitives import vendor_gemm\n" + LINEAR
                     + kernelagent.kernel_function_source(small_task), "kernel", "exec"), namespace)
    finally:
        sys.path.remove(str(small_task.root))
    values = make_case_inputs(small_task.op(), small_task.op().benchmark[0], device="cpu")
    inputs = [values[n] for n in problem._ARGS] + [values[n] for n in problem._UPSTREAM]
    eager = problem.Model()(*inputs)
    assert torch.allclose(namespace["kernel_function"](*inputs), eager, atol=1e-5, rtol=1e-5)

    op_task = geak.prepare(small_task, tmp_path / "run", tmp_path / "op_task")
    meta = json.loads((op_task / "meta.json").read_text())
    assert meta["baseline_callable"] == "baseline_src/reference_step.py:reference_step"
    compile((op_task / "unittest.py").read_text(), "unittest", "exec")
    step = _module(op_task / "baseline_src/reference_step.py", "adbench_tools_geak_baseline")
    frozen = torch.cat([t.float().reshape(-1) for t in step.reference_step(*inputs)])
    assert torch.allclose(frozen, eager)


# ── a model's observed case ──────────────────────────────────────────────────


def payload_rule_task(output_dir: Path, model: str, case_id: str, payloads=None, **build):
    """A model case's pack without a GPU: a tiny real corpus holds its calibration calls
    (placeholders unless ``payloads`` gives them at the case's interface); only the
    operator Gate 2 calibration (the envelope's GPU run) is stubbed."""
    from types import SimpleNamespace
    from unittest import mock

    from evograd.verification.integrations import base, gates
    from tests.verification.test_real_payloads import _write_corpus

    if payloads is None:
        payloads = [(call, order, {"x": torch.ones(1)}, {"dy": torch.ones(1)})
                    for order in (101, 202) for call in "abc"]
    manifest = _write_corpus(output_dir.parent / f"{output_dir.name}_corpus", payloads,
                             case_id=case_id, task=case_id.split("/")[0])

    def calibration(model_obj, case, corpus, **_):
        policy = SimpleNamespace(gamma=4.0, thresholds={})  # a calibration with no envelope
        return SimpleNamespace(op=model_obj.reference_op(case.task)), SimpleNamespace(policy=policy)

    with mock.patch.object(gates, "operator_calibration", calibration), \
            mock.patch.object(base, "configure", lambda: None):
        return build_task(output_dir, model=model, case_id=case_id, corpus=manifest, **build)


def test_a_model_task_is_the_observed_case_under_the_model_reference(tmp_path):
    """The pack `evograd evaluate` judges: the case's configuration and the
    model's own reference computation, both read from the verification side."""
    from evograd.adbench_tools.task import observed_case

    model, case = observed_case("qwen3_0_6b", "rmsnorm/rmsnorm")
    task = payload_rule_task(tmp_path / "task", "qwen3_0_6b", "rmsnorm/rmsnorm")
    reference = model.reference_op("rmsnorm")
    op = task.op()
    assert task.task_id == "qwen3_0_6b.rmsnorm.rmsnorm"
    assert task.spec["model"] == "qwen3_0_6b" and task.spec["case_id"] == "rmsnorm/rmsnorm"
    assert task.spec["level"] == case.level and task.spec["frequency"] == case.frequency
    assert task.workload == {"dims": dict(case.workload.dims), "dtype": case.workload.dtype}
    assert op.forward == reference.forward != get_task_forward("rmsnorm")
    assert op.reference_contract == reference.reference_contract
    assert op.benchmark == op.coverage
    # The observed case itself, provenance included (its layout, its eps).
    assert op.benchmark[0] == case.workload
    assert task.spec["base_declaration"] == f"{task.root / 'declaration.py'}:op"
    assert f"Source: {reference.forward}" in task.reference_source
    # The pack states the tolerance Gate 1 judges by, not the declaration's.
    from evograd.verification.gate1.tolerances import model_policy_identity

    gate1 = task.spec["gate1_tolerances"]
    defaults = model_policy_identity()["defaults"]
    assert "tolerances" not in task.spec
    assert gate1["policy"] == model_policy_identity() and gate1["envelope_gamma"] == 4.0
    assert list(gate1["results"]) == sorted([*op.output_names, *op.grad_names()])
    assert all({"atol": r["atol"], "rtol": r["rtol"]} == defaults[r["dtype"]]
               for r in gate1["results"].values())
    assert "4x the larger of the largest elementwise" in task.statement
    assert "4 ulps of the largest value" in task.statement
    # Search feedback is Gate 1 on payloads; no model case is judged by the declaration.
    assert task.spec["search_correctness"]["rule"] == "gate1_calibration_payloads"
    assert "the declaration's own tolerances" not in task.statement
    with pytest.raises(ValueError, match="needs the model's payload corpus"):
        build_task(tmp_path / "nocorpus", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm")
    with pytest.raises(TypeError, match="search_correctness"):
        build_task(tmp_path / "declared", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm",
                   search_correctness="declared_tolerances")
    with pytest.raises(ValueError, match="no observed case"):
        build_task(tmp_path / "missing", model="qwen3_0_6b", case_id="rmsnorm/nope")
    # The case fixes the declaration and configuration; there is nothing to choose.
    with pytest.raises(TypeError, match="dtype"):
        build_task(tmp_path / "mixed", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm",
                   dtype="float32")
    with pytest.raises(TypeError, match="model"):
        build_task(tmp_path / "operator", case_id="rmsnorm/rmsnorm")


def test_a_model_task_judges_search_feedback_by_gate1_on_calibration_payloads(tmp_path,
                                                                              monkeypatch):
    """The default rule: Gate 1 on the first, middle and last calibration call
    of each order, with the envelope operator Gate 2's calibration gives the
    case; every evaluator the pack starts is pointed at it, and only by the pack."""
    from types import SimpleNamespace

    from evograd.adbench_tools.evaluate import evaluator_env
    from evograd.verification.integrations import base, gates
    from evograd.verification.payloads.gate1 import SEARCH_CORRECTNESS_ENV, SEARCH_RULE
    from tests.verification.test_real_payloads import _write_corpus

    payloads = [(call, order, {"x": torch.ones(2, 4), "weight": torch.ones(4)},
                 {"dy": torch.ones(2, 4)})
                for order in (101, 202, 303, 404) for call in "abcde"]
    manifest = _write_corpus(tmp_path / "corpus", payloads, case_id="rmsnorm/rmsnorm",
                             task="rmsnorm")
    policy = SimpleNamespace(gamma=2.0, thresholds={"out:y": {"envelope": 0.5},
                                                    "grad:x": {"envelope": 0.25},
                                                    "grad:weight": {"rms": 1.0}})
    calibrated = []

    def calibration(model, case, corpus, **options):
        calibrated.append((case.case_id, corpus.corpus_id, options["state"]))
        return SimpleNamespace(op=model.reference_op(case.task)), SimpleNamespace(policy=policy)

    monkeypatch.setattr(gates, "operator_calibration", calibration)
    monkeypatch.setattr(base, "configure", lambda: calibrated.append("configured"))
    task = build_task(tmp_path / "task", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm",
                      corpus=manifest, state=tmp_path / "state")
    search = task.spec["search_correctness"]
    assert search["rule"] == SEARCH_RULE and search["case_id"] == "rmsnorm/rmsnorm"
    assert search["payload_ids"] == [f"o{o}/rmsnorm/rmsnorm/{c}" for o in (101, 202) for c in "ace"]
    assert search["corpus"]["manifest"] == str(Path(manifest).resolve())
    assert search["envelope"] == {"out:y": 1.0, "grad_in:dx": 0.5} and search["calibrated"]
    assert calibrated == ["configured",
                          ("rmsnorm/rmsnorm", search["corpus"]["corpus_id"], tmp_path / "state")]
    assert "applies the acceptance rule of section 2 to 6 recorded training calls" in task.statement
    monkeypatch.setenv(SEARCH_CORRECTNESS_ENV, "/somewhere/else.json")
    assert evaluator_env(task)[SEARCH_CORRECTNESS_ENV] == str(task.root / "task.json")
    # No policy, no pack: the failure says why and nothing is written.
    from evograd.adbench_tools.task import CalibrationFailed

    failure = SimpleNamespace(code="insufficient_calibration", message="out of memory")
    monkeypatch.setattr(gates, "operator_calibration", lambda model, case, corpus, **options: (
        SimpleNamespace(op=model.reference_op(case.task)),
        SimpleNamespace(policy=None, status="incomplete", findings=[failure])))
    with pytest.raises(CalibrationFailed, match="incomplete .insufficient_calibration: out of memory"):
        build_task(tmp_path / "uncalibrated", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm",
                   corpus=manifest, state=tmp_path / "state")
    assert not (tmp_path / "uncalibrated" / "task.json").exists()
    # A pack judged by the declaration's tolerances -- an operator pack, or a model pack
    # built before that rule was removed -- is refused, never judged, and still loads.
    from evograd.adbench_tools.evaluate import search_rule
    from evograd.adbench_tools.task import AgentTask

    stale = AgentTask(task.root, {**task.spec,
                                  "search_correctness": {"rule": "declared_tolerances"}})
    operator = AgentTask(task.root, {k: v for k, v in task.spec.items()
                                     if k not in ("model", "case_id", "search_correctness")})
    for pack in (stale, operator):
        with pytest.raises(ValueError, match="rebuild the pack from a model case"):
            evaluator_env(pack)
        with pytest.raises(ValueError, match="rebuild the pack from a model case"):
            search_rule(pack)
        assert pack.op().name == "rmsnorm"


def test_the_evaluator_judges_correctness_only_by_the_packs_rule(tmp_path, monkeypatch):
    """With the pack's ``task.json`` in the environment, the evolution evaluator's
    correctness check is Gate 1 on its payloads, not generated inputs, for every
    program. Without it the check fails; nothing else is tried."""
    from evograd.evolve import evaluator
    from evograd.verification.payloads import gate1 as payload_gate1

    spec = {"rule": payload_gate1.SEARCH_RULE, "case_id": "c/x"}
    (tmp_path / "task.json").write_text(json.dumps({"search_correctness": spec}))
    seen = []
    monkeypatch.setattr(payload_gate1, "search_correctness",
                        lambda op, module, s, device: seen.append(s) or {"passed": 1, "total": 1})
    monkeypatch.setattr(evaluator, "make_case_inputs",
                        lambda *a, **k: pytest.fail("generated inputs used"))
    monkeypatch.delenv(payload_gate1.SEARCH_CORRECTNESS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="is not set"):
        evaluator._run_correctness(object(), object())
    assert seen == []
    monkeypatch.setenv(payload_gate1.SEARCH_CORRECTNESS_ENV, str(tmp_path / "task.json"))
    assert evaluator._run_correctness(object(), object()) == {"passed": 1, "total": 1}
    assert seen == [spec]


#: Scalars the recorded training calls pass (payload corpora of 2026-10-21) where
#: the task's declared default is another model's or the generic grid's.
MODEL_SCALARS = (
    ("llama_3_2_1b", "llama3_residual_rmsnorm/residual_rmsnorm", {"eps": 1e-5}),
    ("qwen3_0_6b", "rmsnorm/rmsnorm", {"eps": 1e-6}),
    ("qwen2_vl_2b", "layernorm/model.visual.blocks.norm1", {"eps": 1e-6}),
    ("qwen2_vl_2b", "rmsnorm/model.language_model.layers.input_layernorm", {"eps": 1e-6}),
    ("convnextv2_tiny", "layernorm/convnextv2.embeddings.layernorm", {"eps": 1e-6}),
    ("convnextv2_tiny", "spatial_conv2d/convnextv2.embeddings.patch_embeddings",
     {"stride": 4, "padding": 0, "groups": 1}),
    ("dit_s_2", "spatial_conv2d/patch", {"stride": 2, "padding": 0, "groups": 1}),
)


@pytest.mark.parametrize("model,case_id,scalars", MODEL_SCALARS)
def test_a_model_task_states_and_generates_the_models_scalars(tmp_path, model, case_id, scalars):
    """Signature defaults, argument list, skeleton and generated inputs all carry
    the value the model's calls pass, not the declaration's default."""
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import format_default
    from evograd.opdecl.inputs import make_case_inputs

    task = payload_rule_task(tmp_path / "task", model, case_id)
    op = task.op()
    declared = {a.name: a.default for a in get_task(op.name).scalar_inactive_args()}
    assert declared != scalars  # the case this pins
    assert {a.name: a.default for a in op.scalar_inactive_args()} == scalars
    generated = make_case_inputs(op, op.benchmark[0], device="meta")
    assert {name: generated[name] for name in scalars} == scalars
    abi = {e["name"]: e["default"] for e in task.spec["abi"]["arguments"] if e["kind"] == "scalar"}
    assert abi == scalars
    skeleton = (task.root / "skeleton.py").read_text()
    for name, value in scalars.items():
        assert f"- `{name}`: Python scalar, default {value!r} (inactive)" in task.statement
        assert f"{name}={format_default(value)}" in task.spec["abi"]["forward_signature"]
        assert f"{name}={format_default(value)}" in task.spec["abi"]["backward_signature"]
    assert task.spec["abi"]["forward_signature"] in skeleton
    assert task.spec["abi"]["forward_signature"] in task.statement


def get_task_forward(name: str) -> str:
    from evograd.benchmark import get_task

    return get_task(name).forward


def test_auto_library_follows_what_the_frontend_preserves(small_task):
    from evograd.adbench_tools.task import resolve_library

    assert resolve_library("auto", ["linear", "view"])[0] == ("vendor_gemm",)
    assert resolve_library("auto", ["_grouped_mm"])[0] == ("vendor_gemm",)
    # A convolution is cuDNN's (Inductor's extern convolution), not a GEMM.
    assert resolve_library("auto", ["conv2d"])[0] == ("vendor_conv",)
    assert resolve_library("auto", ["conv1d", "linear", "silu"])[0] == ("vendor_conv", "vendor_gemm")
    assert resolve_library("auto", ["scaled_dot_product_attention"])[0] == ("vendor_sdpa",)
    assert resolve_library("auto", ["matmul", "scaled_dot_product_attention"])[0] == (
        "vendor_gemm", "vendor_sdpa")
    assert resolve_library("auto", ["mul", "rsqrt"])[0] == ()
    assert resolve_library("vendor_gemm,vendor_sdpa", [])[0] == ("vendor_gemm", "vendor_sdpa")
    assert resolve_library("vendor_conv", ["linear"])[0] == ("vendor_conv",)
    assert resolve_library("none", ["linear", "scaled_dot_product_attention"])[0] == ()
    with pytest.raises(ValueError):
        resolve_library("vendor_blas", [])
    # The pack records what the trace of its reference reached.
    assert small_task.spec["library_policy"]["observed_calls"] == ["linear"]
    assert small_task.allowed_primitives == ("vendor_gemm",)


#: Cases whose contraction the reference reaches only through code ``reference.py``
#: imports (a shared helper, a Hugging Face function, ``transformers.integrations``):
#: a scan of the reference source granted them nothing. Fused attention stays
#: ``scaled_dot_product_attention`` at the function level (ATen on the meta device
#: would see its math decomposition, ``bmm``).
REACHED = (
    ("qwen3_5_4b", "qwen3_5_attention_block/attention_block", "scaled_dot_product_attention",
     ("vendor_gemm", "vendor_sdpa")),
    ("qwen3_5_4b", "qwen3_5_gated_delta_rule/gated_delta_rule", "matmul", ("vendor_gemm",)),
    ("olmoe_1b_7b", "olmoe_moe_experts/moe_experts", "_grouped_mm", ("vendor_gemm",)),
    ("qwen3_0_6b", "qwen3_attention_block/attention_block", "scaled_dot_product_attention",
     ("vendor_gemm", "vendor_sdpa")),
    ("qwen3_0_6b", "qwen3_lm_head_ce/flce", "linear", ("vendor_gemm",)),
    # Convolutions: the L1 primitives and the L2 cases that contain one.
    ("qwen3_5_4b", "causal_conv1d/gdn_conv1d", "conv1d", ("vendor_conv",)),
    ("qwen3_5_4b", "qwen3_5_gdn_in_proj/gdn_in_proj", "conv1d", ("vendor_conv", "vendor_gemm")),
    ("convnextv2_tiny", "spatial_conv2d/convnextv2.encoder.stages.2.layers.dwconv", "conv2d",
     ("vendor_conv",)),
    ("convnextv2_tiny", "convnextv2_tiny_dwconv_layernorm/convnextv2.encoder.stages.2.layers",
     "conv2d", ("vendor_conv",)),
    ("dit_s_2", "spatial_conv2d/patch", "conv2d", ("vendor_conv",)),
    # float32 attention (eager runs memory-efficient attention): vendor_sdpa covers it.
    ("dit_s_2", "dit_s_2_dit_attention/blocks.attention", "scaled_dot_product_attention",
     ("vendor_gemm", "vendor_sdpa")),
)


@pytest.mark.parametrize("model,case_id,reached,granted", REACHED)
def test_the_auto_grant_follows_what_one_reference_call_reaches(model, case_id, reached, granted):
    from evograd.adbench_tools.task import observed_calls, observed_case, resolve_library

    model_obj, case = observed_case(model, case_id)
    calls = observed_calls(model_obj.reference_op(case.task), case.workload, device="meta")
    assert reached in calls
    assert resolve_library("auto", calls)[0] == granted


def test_only_the_auto_policy_traces_the_reference(tmp_path, monkeypatch):
    """A requested grant never depends on the reference running anywhere."""
    from evograd.adbench_tools import task as task_module

    monkeypatch.setattr(task_module, "observed_calls",
                        lambda *a, **k: pytest.fail("the reference was traced"))
    task = payload_rule_task(tmp_path / "task", SMALL_MODEL, SMALL_CASE, library="none")
    assert task.allowed_primitives == ()
    assert task.spec["library_policy"] == {"requested": "none",
                                           "decision": "requested: no library primitives"}


def test_a_reference_that_needs_values_is_traced_on_the_build_device(monkeypatch):
    import evograd.opdecl.oracle  # noqa: F401 - the module, not the function it exports
    from evograd.adbench_tools.task import observed_calls, observed_case

    def forward(x, weight, bias):
        if x.is_meta:
            raise NotImplementedError("needs values")
        return torch.nn.functional.linear(x, weight, bias)

    monkeypatch.setattr(sys.modules["evograd.opdecl.oracle"], "resolve_forward", lambda op: forward)
    model_obj, case = observed_case(SMALL_MODEL, SMALL_CASE)
    op = model_obj.reference_op(case.task)
    assert "linear" in observed_calls(op, case.workload, device="cpu")
    with pytest.raises(NotImplementedError):
        observed_calls(op, case.workload, device="meta")


def test_the_statement_states_the_rules_and_what_acceptance_judges(small_task):
    from evograd.verification.integrations.gates import MODEL_ORDERS, PAYLOAD_SPLITS

    rules = small_task.statement.split("## 3.")[1].split("## 4.")[0]
    for text in ("conv1d/2d/3d", "embedding", "batch_norm", "loss", "lstm", "torch.nn,",
                 "torch.func", "torch.utils.cpp_extension", "`torch._*`, at any depth",
                 "`from torch.X import Y` counts as `torch.X.Y`",
                 "after `import torch.X as Y`", "`Y = torch.X`, `Y.Z` counts as `torch.X.Z`",
                 "`from torch import *`", "ctypes", "importlib", "builtins", "`getattr`",
                 "`__import__`", "`eval`", "`exec`", "__matmul__", "`@=`"):
        assert text in rules, text
    # Pipeline A's prompt states the same rules, from the same tables.
    from evograd.pipelines.a_atenir_llm.prompts import render_pair_rules
    from evograd.pipelines.shared.primitives import describe_rules

    assert describe_rules("Every line of your module") in rules
    shared = describe_rules("Every line inside the block")
    assert shared in render_pair_rules(small_task.op(), small_task.allowed_primitives)
    # Acceptance (Gate 1, operator Gate 2) judges every payload of both splits --
    # all four orders, the search calls included -- and calibrates the envelope on them.
    assert PAYLOAD_SPLITS == ("calibration", "acceptance") and len(MODEL_ORDERS) == 4
    scored = small_task.statement.split("## 4.")[1]
    assert ("Acceptance then judges every recorded call of this operator in all four "
            "training steps, these calls included; the envelope is measured on those same "
            "calls.") in scored
    assert "not among them" not in small_task.statement
    assert "held out" not in small_task.statement


# ── coding-agent glue (candidates/claude_code, candidates/codex) ─────────────


@pytest.fixture(scope="module")
def coding_agents():
    sys.path.insert(0, str(REPO / "candidates"))
    try:
        claude = _module(REPO / "candidates/claude_code/run.py", "adbench_tools_claude_code")
        codex = _module(REPO / "candidates/codex/run.py", "adbench_tools_codex")
    finally:
        sys.path.remove(str(REPO / "candidates"))
    return claude, codex


def test_the_coding_agents_work_against_a_metered_evaluator_that_hides_the_repository(
        small_task, coding_agents, tmp_path, monkeypatch):
    from evograd.adbench_tools.ledger import RunLedger

    shared = coding_agents[1]._coding_agent
    ledger = RunLedger(tmp_path / "run", agent="codex", task=small_task, config={})
    args = argparse.Namespace(workspace=tmp_path / "workspace", budget=10, eval_budget=3,
                              eval_timeout=300)
    workspace = shared.Workspace(small_task, ledger, "codex", args)
    hidden = (str(REPO), str(ledger.root), str(small_task.root))
    assert sorted(p.name for p in workspace.root.iterdir()) == [
        "TASK.md", "adbench_primitives.py", "candidate.py", "evaluate.py"]
    for path in workspace.root.iterdir():
        assert not any(h in path.read_text() for h in hidden), path.name
    prompt = (ledger.root / shared.PROMPT).read_text()
    assert prompt.endswith(small_task.statement) and "at most 3 evaluations" in prompt
    assert "at most 10 model API requests" in prompt

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO / "src"), "/elsewhere"]))
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(REPO))
    env = workspace.environment("0")
    assert env["PYTHONPATH"] == "/elsewhere" and "SLURM_SUBMIT_DIR" not in env
    assert env["PWD"] == str(workspace.root)
    assert not any(h in value for value in env.values() for h in hidden)

    def evaluate(*args, cwd=workspace.root):
        return subprocess.run([sys.executable, "evaluate.py", *args], cwd=cwd, env=env,
                              capture_output=True, text=True)

    for origin in ("codex", "gepa", "codex"):
        ledger.log_evaluation({"kind": "full", "origin": origin, "correct": False})
    once = evaluate("--correctness")  # the skeleton: incorrect, on a GPU or not
    assert once.returncode == 1 and "INCORRECT" in once.stdout, once.stdout + once.stderr
    assert "evaluations remaining: 0 of 3" in once.stdout
    assert not any(h in once.stdout for h in hidden)
    refused = evaluate()
    assert refused.returncode == 3 and "BUDGET EXHAUSTED" in refused.stdout
    traceback = f'File "{ledger.root}/programs/a.py" from "{REPO}/src/b.py", {small_task.root}'
    assert workspace.evaluator.redact(traceback) == \
        'File "<run>/programs/a.py" from "<evograd>/src/b.py", <task>'
    best, extra = workspace.collect()
    assert best is None and extra["deliverable"] == "none" and extra["oracle_intact"]
    gone = evaluate(cwd=ledger.root / "workspace")
    assert gone.returncode == 1 and "not reachable" in gone.stdout


def test_the_coding_agents_deliverable_is_their_own_verified_choice(small_task, coding_agents,
                                                                     tmp_path):
    from evograd.adbench_tools.ledger import RunLedger

    choose = coding_agents[0]._coding_agent.choose
    ledger = RunLedger(tmp_path / "run", agent="claude_code", task=small_task, config={})
    final = tmp_path / "candidate.py"

    def evaluated(source, score, correct=True, kind="full"):
        program = prepare_program(small_task, source, ledger.root / "programs", origin="claude_code")
        ledger.log_evaluation({"kind": kind, "origin": "claude_code", "correct": correct,
                               "combined_score": score, "program": str(program)})
        return program

    assert choose(small_task, ledger, "claude_code", final) == (None, "none")
    final.write_text((small_task.root / "skeleton.py").read_text())
    assert choose(small_task, ledger, "claude_code", final) == (None, "none")
    final.write_text(LINEAR)
    assert choose(small_task, ledger, "claude_code", final) == (final, "final_unverified")
    faster = evaluated(LINEAR + "\nFASTER = 1\n", 2.0)
    evaluated(LINEAR + "\nWRONG = 1\n", 9.0, correct=False)
    evaluated(LINEAR + "\nCHECKED = 1\n", None, kind="correctness")
    assert choose(small_task, ledger, "claude_code", final) == (faster, "best_evaluated")
    evaluated(LINEAR, 1.0, kind="correctness")
    assert choose(small_task, ledger, "claude_code", final) == (final, "final")


def test_codex_runs_on_the_shared_model_and_endpoint(coding_agents, tmp_path):
    claude, codex = coding_agents
    command = codex.command(argparse.Namespace(model="gpt-x", api_base="https://gw.example/v1",
                                               effort="high"), tmp_path)
    assert command[:2] == ["codex", "exec"] and command[-1] == "-" and "-o" not in command
    assert command[command.index("-m") + 1] == "gpt-x"
    assert 'model_providers.adbench.base_url="https://gw.example/v1"' in command
    assert 'model_provider="adbench"' in command
    assert 'model_providers.adbench.supports_websockets=false' in command
    assert 'model_providers.adbench.requires_openai_auth=false' in command
    assert 'model_providers.adbench.stream_max_retries=0' in command
    assert 'model_reasoning_effort="high"' in command
    default = codex.command(argparse.Namespace(model="gpt-x", api_base=codex.OPENAI_API_BASE,
                                               effort=None), tmp_path)
    assert not any(part.startswith("model_reasoning") for part in default)
    assert f'model_providers.adbench.base_url="{codex.OPENAI_API_BASE}"' in default
    command = claude.command(argparse.Namespace(claude_model="opus", effort=None), "PROMPT")
    assert command[:3] == ["claude", "-p", "PROMPT"]
    assert command[command.index("--model") + 1] == "opus" and "--effort" not in command
    assert codex._coding_agent.cli_version(sys.executable).startswith("Python 3")
    assert codex._coding_agent.cli_version(str(tmp_path / "missing")) is None
    log = tmp_path / "events.jsonl"
    log.write_text('{"type":"item.completed","item":{"type":"agent_message","text":"first"}}\n'
                   'not json\n{"type":"turn.completed","usage":{"input_tokens":3}}\n'
                   '{"type":"item.completed","item":{"type":"agent_message","text":"last"}}\n')
    usage, reply = codex.read_events(log)
    assert reply == "last" and [e["usage"]["input_tokens"] for e in usage] == [3]


def test_a_coding_agent_session_leaves_nothing_running(coding_agents, tmp_path):
    """Whatever the agent started is killed before the final measurement times anything."""
    launch = coding_agents[0]._coding_agent.launch
    started = time.time()
    status = launch(["bash", "-c", "(trap '' INT; exec sleep 600) & echo $! > pid; wait"],
                    tmp_path, dict(os.environ), tmp_path / "log.jsonl", time_limit=2)
    assert status == "time_limit" and time.time() - started < 30
    pid = int((tmp_path / "pid").read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"the agent's background process {pid} survived the session")
