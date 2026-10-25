"""The agent-neutral comparison harness, without a GPU, an LLM or an agent.

What these pin is what makes the comparison fair: the task pack is the
frontend's own scoped declaration, the statement carries evograd's system
message verbatim, the reference an agent reads computes exactly the declared
forward, every line an agent writes is under the evaluator's static rules, and
nothing outside the evolve block survives into the final measurement.
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

REPO = Path(__file__).resolve().parents[2]

#: A correct PyTorch pair for the SwiGLU task: elementwise torch plus the
#: granted vendor_gemm, which is all the static rules allow besides Triton.
SWIGLU = '''
import torch


def llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    rows = x.reshape(-1, x.shape[-1])
    gate = vendor_gemm(rows, gate_weight.t()).float()
    up = vendor_gemm(rows, up_weight.t()).float()
    hidden = (gate * torch.sigmoid(gate) * up).to(x.dtype)
    out = vendor_gemm(hidden, down_weight.t()).to(x.dtype)
    return out.view(*x.shape[:-1], -1), (x, gate_weight, up_weight, down_weight, gate, up, hidden)


def llama3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
    x, gate_weight, up_weight, down_weight, gate, up, hidden = saved_tensors
    rows = x.reshape(-1, x.shape[-1])
    d = dout.reshape(-1, dout.shape[-1])
    ddown = vendor_gemm(d.t(), hidden).to(down_weight.dtype)
    dh = vendor_gemm(d, down_weight).float()
    s = torch.sigmoid(gate)
    dup = (dh * gate * s).to(x.dtype)
    dgate = (dh * up * s * (1 + gate * (1 - s))).to(x.dtype)
    dx = vendor_gemm(dgate, gate_weight) + vendor_gemm(dup, up_weight)
    return (dx.to(x.dtype).view_as(x), vendor_gemm(dgate.t(), rows).to(gate_weight.dtype),
            vendor_gemm(dup.t(), rows).to(up_weight.dtype), ddown)
'''


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def timed_task(tmp_path_factory):
    return build_task("llama3_swiglu_mlp", tmp_path_factory.mktemp("timed") / "task")


@pytest.fixture(scope="module")
def small_task(tmp_path_factory):
    return build_task("llama3_swiglu_mlp", tmp_path_factory.mktemp("small") / "task",
                      case="correctness", dtype="float32")


# ── the task pack ────────────────────────────────────────────────────────────


def test_the_default_task_is_the_timed_case(timed_task):
    op = timed_task.op()
    assert timed_task.workload == {"dims": {"B": 2, "T": 2048, "H": 2048, "I": 8192},
                                   "dtype": "bfloat16"}
    assert op.correctness == op.benchmark == op.coverage
    assert timed_task.allowed_primitives == ("vendor_gemm",)
    assert (timed_task.root / "adbench_primitives.py").is_file()


def test_the_declaration_is_the_frontends_scoped_declaration(small_task, tmp_path):
    """Byte-identical to the declaration.py `evograd seed unified` writes."""
    from evograd.benchmark import get_task
    from evograd.pipelines.unified.synthesize import _write_declaration, select_workload

    base = get_task("llama3_swiglu_mlp")
    _write_declaration(tmp_path / "declaration.py", base, select_workload(base, dtype="float32"))
    assert (small_task.root / "declaration.py").read_text() == \
        (tmp_path / "declaration.py").read_text()
    op = small_task.op()
    assert op.correctness == op.benchmark == op.coverage
    assert op.forward == base.forward


def test_the_statement_carries_evograds_system_message_verbatim(small_task):
    message = system_message(small_task)
    statement = small_task.statement
    assert message in statement
    assert small_task.spec["system_message_sha256"] == hashlib.sha256(message.encode()).hexdigest()
    assert small_task.spec["statement_sha256"] == hashlib.sha256(statement.encode()).hexdigest()
    assert small_task.reference_source.strip() in statement


def test_a_task_pack_is_never_overwritten(small_task):
    with pytest.raises(FileExistsError):
        build_task("llama3_swiglu_mlp", small_task.root)


@pytest.mark.parametrize("name", ["llama3_swiglu_mlp", "llama3_qkv_rope", "layernorm",
                                  "poly_norm", "qwen3_qkv_norm_rope"])
def test_the_reference_an_agent_reads_is_the_declared_forward(name):
    from evograd.benchmark import get_task
    from evograd.opdecl.importing import resolve_callable
    from evograd.opdecl.inputs import make_case_inputs

    op = get_task(name)
    source, _ = reference_closure(op.forward)
    namespace: dict = {}
    exec(compile(source, "reference", "exec"), namespace)
    workload = next(w for w in op.correctness if w.dtype == "float32")
    values = make_case_inputs(op, workload, device="cpu")
    args = [values.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
    ours, declared = namespace["reference_forward"](*args), resolve_callable(op.forward)(*args)
    ours = ours if isinstance(ours, tuple) else (ours,)
    declared = declared if isinstance(declared, tuple) else (declared,)
    assert all(torch.equal(a, b) for a, b in zip(ours, declared))


# ── packaging ────────────────────────────────────────────────────────────────


def test_packaging_puts_the_whole_module_under_the_static_rules(small_task, tmp_path):
    from evograd.pipelines.shared.primitives import check_source

    artifact = package(small_task, extract_code(f"Sure:\n```python\n{SWIGLU}```\n", small_task))
    usage = check_source(artifact, allowed=small_task.allowed_primitives)
    assert usage["primitive_call_sites"]["vendor_gemm"] == 9
    assert unpackage(artifact).strip() == SWIGLU.strip()
    with pytest.raises(Exception, match="matrix-multiply"):
        check_source(package(small_task, SWIGLU + "\nZ = 1 @ 2\n"),
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
        package(small_task, SWIGLU + "\n" + bad)


def test_packaging_requires_the_pair(small_task):
    with pytest.raises(PackagingError, match="must define"):
        package(small_task, "import torch\n")


def test_the_standalone_primitive_import_is_replaced(small_task):
    artifact = package(small_task, "from __future__ import annotations\n"
                                   "from adbench_primitives import vendor_gemm\n" + SWIGLU)
    assert "adbench_primitives" not in artifact
    assert artifact.index("from __future__ import annotations") < artifact.index("import torch")


def test_code_outside_the_block_never_reaches_the_final_measurement(small_task, tmp_path):
    artifact = package(small_task, SWIGLU, origin="openevolve")
    assert fixed_code_intact(small_task, artifact) is True
    tampered = artifact.replace("DEPLOYMENT_ENTRY = ", "BACKDOOR = 1\nDEPLOYMENT_ENTRY = ")
    assert tampered.split(END, 1)[1] != artifact.split(END, 1)[1]
    assert fixed_code_intact(small_task, tampered) is False
    rebuilt = prepare_program(small_task, tampered, tmp_path, origin="final", repackage=True)
    assert "BACKDOOR" not in rebuilt.read_text()


def test_a_packaged_pair_is_correct_against_the_oracle(small_task, tmp_path, monkeypatch):
    pytest.importorskip("triton")
    from evograd.evolve.evaluator import _load_module, _run_correctness

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    path = tmp_path / "artifact.py"
    path.write_text(package(small_task, SWIGLU))
    report = _run_correctness(small_task.op(), _load_module(str(path)), device="cpu")
    assert report["passed"] == report["total"] == 1, report["reports"]


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
        exec(compile("from adbench_primitives import vendor_gemm\n" + SWIGLU
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


def test_a_model_task_is_the_observed_case_under_the_model_reference(tmp_path):
    """The pack `evograd evaluate` judges: the case's configuration and the
    model's own reference computation, both read from the verification side."""
    from evograd.adbench_tools.task import observed_case

    model, case = observed_case("qwen3_0_6b", "rmsnorm/rmsnorm")
    task = build_task(None, tmp_path / "task", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm")
    reference = model.reference_op("rmsnorm")
    op = task.op()
    assert task.task_id == "qwen3_0_6b.rmsnorm.rmsnorm"
    assert task.spec["model"] == "qwen3_0_6b" and task.spec["case_id"] == "rmsnorm/rmsnorm"
    assert task.spec["level"] == case.level and task.spec["frequency"] == case.frequency
    assert task.workload == {"dims": dict(case.workload.dims), "dtype": case.workload.dtype}
    assert op.forward == reference.forward != get_task_forward("rmsnorm")
    assert op.reference_contract == reference.reference_contract
    assert op.correctness == op.benchmark == op.coverage
    assert (op.correctness[0].dims, op.correctness[0].dtype) == (case.workload.dims,
                                                                 case.workload.dtype)
    assert task.spec["base_declaration"] == f"{task.root / 'declaration.py'}:op"
    assert f"Source: {reference.forward}" in task.reference_source
    with pytest.raises(ValueError, match="no observed case"):
        build_task(None, tmp_path / "missing", model="qwen3_0_6b", case_id="rmsnorm/nope")
    with pytest.raises(ValueError, match="fixes the declaration"):
        build_task(None, tmp_path / "mixed", model="qwen3_0_6b", case_id="rmsnorm/rmsnorm",
                   dtype="float32")


def get_task_forward(name: str) -> str:
    from evograd.benchmark import get_task

    return get_task(name).forward


def test_auto_library_follows_what_the_frontend_preserves():
    from evograd.adbench_tools.task import resolve_library

    gemm = "def f(x, w):\n    return torch.nn.functional.linear(x, w)\n"
    sdpa = "def f(q, k, v):\n    return F.scaled_dot_product_attention(q, k, v, is_causal=True)\n"
    both = gemm + sdpa
    assert resolve_library("auto", gemm)[0] == ("vendor_gemm",)
    assert resolve_library("auto", sdpa)[0] == ("vendor_sdpa",)
    assert resolve_library("auto", both)[0] == ("vendor_gemm", "vendor_sdpa")
    assert resolve_library("auto", "def f(x):\n    return x * 2\n")[0] == ()
    assert resolve_library("vendor_gemm,vendor_sdpa", "")[0] == ("vendor_gemm", "vendor_sdpa")
    assert resolve_library("none", both)[0] == ()


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
    final.write_text(SWIGLU)
    assert choose(small_task, ledger, "claude_code", final) == (final, "final_unverified")
    faster = evaluated(SWIGLU + "\nFASTER = 1\n", 2.0)
    evaluated(SWIGLU + "\nWRONG = 1\n", 9.0, correct=False)
    evaluated(SWIGLU + "\nCHECKED = 1\n", None, kind="correctness")
    assert choose(small_task, ledger, "claude_code", final) == (faster, "best_evaluated")
    evaluated(SWIGLU, 1.0, kind="correctness")
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
