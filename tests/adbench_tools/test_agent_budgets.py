"""Driver budget wiring and recoverable submissions, without a GPU or model API."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = "https://example.invalid/v1"


def load_driver(agent):
    path = REPO / "candidates" / agent / "run.py" if agent else REPO / "candidates" / "bench.py"
    spec = importlib.util.spec_from_file_location(f"budget_test_{agent or 'bench'}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    from tests.adbench_tools.test_adbench_tools import payload_rule_task

    return payload_rule_task(tmp_path_factory.mktemp("budget_task") / "task",
                             "qwen3_0_6b", "rmsnorm/rmsnorm")


@pytest.fixture
def metered(monkeypatch):
    """Exercise real driver start/finish, with a non-network model endpoint."""
    from evograd.adbench_tools import llm_budget

    monkeypatch.syspath_prepend(str(REPO / "candidates"))
    common = importlib.import_module("_common")
    monkeypatch.setattr(common.atexit, "register", lambda *args: None)
    environment = {
        "OPENAI_API_KEY": "test-upstream-key",
        "CODEX_API_KEY": "test-codex-key",
        "OPENAI_BASE_URL": UPSTREAM,
        "OPENAI_API_BASE": UPSTREAM,
        "ADBENCH_API_BASE": UPSTREAM,
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    for name in ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"):
        monkeypatch.delenv(name, raising=False)
    instances = []

    class Proxy:
        def __init__(self, upstream, limit, path, *, api_key, api="openai"):
            self.upstream, self.limit, self.budget, self.path = upstream, limit, limit, path
            self.api_key, self.api = api_key, api
            self.base_url = "http://127.0.0.1:54321" + ("/v1" if api == "openai" else "")
            self.calls_used = self.rejected_requests = 0
            self.started = self.closed = False
            instances.append(self)

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

        def exhaust(self):
            self.calls_used, self.rejected_requests = self.limit, 1

    monkeypatch.setattr(llm_budget, "LLMBudgetProxy", Proxy)
    yield SimpleNamespace(common=common, instances=instances, environment=environment)
    # `start` points every evaluator at the pack's payload rule through the environment.
    os.environ.pop("EVOGRAD_SEARCH_CORRECTNESS", None)


def arguments(task, run_dir, *extra, api_base=UPSTREAM):
    return ["--task", str(task.root), "--run-dir", str(run_dir), "--budget", "10",
            "--model", "gpt-5.5", *(["--api-base", api_base] if api_base else []),
            "--no-score", *extra]


def assert_submission(task, run_dir, agent, marker):
    """A final driver artifact must remain consumable by the benchmark."""
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "budget_exhausted"
    assert record["llm_budget"] == {"limit": 10, "used": 10, "rejected": 1,
                                    "unit": "api_requests"}
    assert marker in Path(record["best_program"]).read_text()
    submission = load_driver("").submit(task, run_dir, agent)
    assert submission["submission"] == str(run_dir / "submission.py")
    assert marker in (run_dir / "submission.py").read_text()
    return record


def test_shared_budget_changes_only_the_runtime_endpoint_and_restores_environment(
        task, tmp_path, metered):
    common = metered.common
    args = common.parser("test").parse_args(arguments(task, tmp_path / "run"))
    loaded, ledger = common.start(args, "gepa", {})
    proxy = metered.instances[-1]
    assert proxy.started and proxy.limit == 10 and proxy.upstream == UPSTREAM
    assert proxy.api_key == "test-upstream-key" and proxy.api == "openai"
    assert "ANTHROPIC_BASE_URL" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
    assert proxy.path == ledger.root / "api_calls.jsonl"
    assert args.api_base == proxy.base_url
    assert os.environ["OPENAI_API_KEY"] == os.environ["CODEX_API_KEY"] == "adbench-local"
    assert os.environ["OPENAI_BASE_URL"] == os.environ["OPENAI_API_BASE"] == proxy.base_url
    assert ledger.record["config"]["api_base"] == UPSTREAM
    assert ledger.record["config"]["model"] == "gpt-5.5"
    assert ledger.record["config"]["budget_unit"] == "api_requests"
    # Using the last allowed request is not itself a failed run.
    proxy.calls_used = 10
    assert not common.budget_exhausted(args)
    best = ledger.root / "best.py"
    best.write_text("# saved candidate\n")
    assert common.finish(args, loaded, ledger, best) == 0
    assert ledger.record["status"] == "completed"
    assert proxy.closed and args.api_base == UPSTREAM
    assert {name: os.environ.get(name) for name in metered.environment} == metered.environment
    with pytest.raises(SystemExit, match="already holds a run"):
        common.start(args, "gepa", {})
    assert len(metered.instances) == 1


def test_codex_separates_model_and_evaluator_budgets_and_keeps_its_submission(
        task, tmp_path, monkeypatch, metered):
    driver = load_driver("codex")
    shared = driver._coding_agent
    run_dir = tmp_path / "codex"
    source = (task.root / "skeleton.py").read_text() + "\n# codex-deliverable\n"
    observed = {}

    class Workspace:
        def __init__(self, task_, ledger, origin, args):
            self.root = ledger.root / "workspace"
            self.root.mkdir()
            self.best = self.root / "candidate.py"
            self.best.write_text(source)
            observed["evaluation_budget"] = args.eval_budget
            (ledger.root / shared.PROMPT).write_text(shared.prompt(
                task_, args.budget, evaluation_budget=args.eval_budget))

        def environment(self, gpu):
            return dict(os.environ)

        def collect(self):
            return self.best, {"deliverable": "final_unverified"}

    def launch(command, cwd, env, log, time_limit, *, input_text):
        assert command[command.index("-m") + 1] == "gpt-5.5"
        assert 'model_providers.adbench.wire_api="responses"' in command
        assert f'model_providers.adbench.base_url="{metered.instances[-1].base_url}"' in command
        assert env["OPENAI_API_KEY"] == "adbench-local"
        assert "at most 10 model API requests" in input_text
        assert "at most 3 evaluations" in input_text
        log.write_text("")
        metered.instances[-1].exhaust()
        return "exit_1"

    monkeypatch.setattr(driver.shutil, "which", lambda _: "/test/codex")
    monkeypatch.setattr(shared, "cli_version", lambda _: "test-version")
    monkeypatch.setattr(shared, "Workspace", Workspace)
    monkeypatch.setattr(shared, "launch", launch)
    assert driver.main(arguments(task, run_dir, "--eval-budget", "3")) == 0
    record = assert_submission(task, run_dir, "codex", "codex-deliverable")
    assert observed["evaluation_budget"] == record["config"]["evaluations"] == 3
    assert record["config"]["budget"] == 10
    assert metered.instances[-1].closed


def test_claude_code_caps_anthropic_requests_with_the_key_kept_in_the_proxy(
        task, tmp_path, monkeypatch, metered):
    driver = load_driver("claude_code")
    shared = driver._coding_agent
    run_dir = tmp_path / "claude_code"
    # Set by a launching Claude Code session; the CLI would then ignore the key.
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-vscode")

    def launch(command, cwd, env, log, time_limit, input_text=None):
        proxy = metered.instances[-1]
        assert (proxy.api, proxy.limit, proxy.api_key) == ("anthropic", 10, "test-anthropic-key")
        assert proxy.upstream == metered.common.DEFAULT_ANTHROPIC_API_BASE
        # The real workspace environment: the proxy's endpoint, never its key.
        assert env["ANTHROPIC_BASE_URL"] == proxy.base_url == "http://127.0.0.1:54321"
        assert env["ANTHROPIC_API_KEY"] == "adbench-local"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert "CLAUDE_CODE_ENTRYPOINT" not in env
        assert not any("test-anthropic-key" in value for value in env.values())
        # Nor the other provider's real key.
        assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env
        assert command[command.index("--effort") + 1] == "high"
        prompt = command[command.index("-p") + 1]
        assert "at most 10 model API requests" in prompt
        assert "recorded separately from model API requests" in prompt
        (cwd / "candidate.py").write_text((task.root / "skeleton.py").read_text()
                                          + "\n# claude-deliverable\n")
        log.write_text(json.dumps({"type": "result", "result": "done", "num_turns": 3}) + "\n")
        proxy.exhaust()
        return "exit_1"

    monkeypatch.setattr(driver.shutil, "which", lambda _: "/test/claude")
    monkeypatch.setattr(shared, "cli_version", lambda _: "test-version")
    monkeypatch.setattr(shared, "launch", launch)
    assert driver.main(arguments(task, run_dir, "--effort", "high", "--workspace",
                                 str(tmp_path / "workspace"), api_base=None)) == 0
    record = assert_submission(task, run_dir, "claude_code", "claude-deliverable")
    config = record["config"]
    assert (config["budget"], config["budget_unit"]) == (10, "api_requests")
    assert config["evaluations"] is None
    assert config["api_base"] == metered.common.DEFAULT_ANTHROPIC_API_BASE
    assert metered.instances[-1].closed
    assert os.environ["ANTHROPIC_API_KEY"] == "test-anthropic-key"
    assert os.environ["CLAUDE_CODE_ENTRYPOINT"] == "claude-vscode"
    assert "ANTHROPIC_BASE_URL" not in os.environ


def test_geak_counts_anthropic_requests_without_a_cap_and_passes_effort(
        task, tmp_path, monkeypatch, metered):
    driver = load_driver("geak")
    geak_root = tmp_path / "GEAK"
    (geak_root / "kernel_workflow").mkdir(parents=True)
    (geak_root / "kernel_workflow" / "kernel_workflow.js").write_text("// fake workflow\n")
    prepare = ModuleType("prepare")
    prepare.prepare = lambda task_, run_root, out: out
    monkeypatch.setitem(sys.modules, "prepare", prepare)
    skeleton = (task.root / "skeleton.py").read_text()

    def run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="NVIDIA GH200\n")
        proxy = metered.instances[-1]
        assert (proxy.api, proxy.limit, proxy.api_key) == ("anthropic", None, "test-anthropic-key")
        env = kwargs["env"]
        assert env["ANTHROPIC_BASE_URL"] == proxy.base_url
        assert env["ANTHROPIC_API_KEY"] == "adbench-local"
        assert not any("test-anthropic-key" in value for value in env.values())
        # Nor the other provider's real key.
        assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env
        assert command[command.index("--effort") + 1] == "max"
        candidate = kwargs["cwd"] / "exp" / "round1" / "workspace" / "kernel_src" / "candidate.py"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(skeleton + "\n# geak-deliverable\n")
        proxy.calls_used = 37  # counted, never refused
        return SimpleNamespace(stdout=json.dumps({"result": "{}", "num_turns": 9}), stderr="")

    monkeypatch.setattr(driver.shutil, "which", lambda _: "/test/claude")
    monkeypatch.setattr(driver.subprocess, "run", run)
    run_dir = tmp_path / "geak"
    assert driver.main(arguments(task, run_dir, "--geak-root", str(geak_root),
                                 "--backend", "claude", "--effort", "max", api_base=None)) == 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "completed"
    config = record["config"]
    assert (config["budget"], config["budget_unit"]) == (2, "optimization_directions")
    assert config["effort"] == "max" and "geak_budget" not in config
    assert record["llm_budget"] == {"limit": None, "used": 37, "rejected": 0,
                                    "unit": "api_requests"}
    assert "geak-deliverable" in Path(record["best_program"]).read_text()
    assert metered.instances[-1].closed and "ANTHROPIC_BASE_URL" not in os.environ


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_geak_codex_uses_openai_proxy_and_workflow_result(
        task, tmp_path, monkeypatch, metered, outcome):
    driver = load_driver("geak")
    root = tmp_path / "GEAK"
    for rel in ("kernel_workflow/kernel_workflow.js", "interface/runtime/engine/run_workflow.mjs"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fake runtime")
    prepare = ModuleType("prepare")
    prepare.prepare = lambda task_, run_root, out: out
    monkeypatch.setitem(sys.modules, "prepare", prepare)
    monkeypatch.setattr(driver.shutil, "which", lambda name: f"/test/{name}")
    monkeypatch.setenv("GEAK_MODEL", "wrong-model")
    monkeypatch.setenv("GEAK_AGENT_PROFILE", "wrong-profile")
    monkeypatch.setenv("GEAK_CODEX_EXTRA_ARGS", "-c model_provider=wrong")
    # Codex must work without any Anthropic credential.
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    def run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="NVIDIA GH200")
        proxy = metered.instances[-1]
        assert (proxy.api, proxy.limit, proxy.api_key) == ("openai", None, "test-upstream-key")
        assert command[0] == "node" and command[command.index("--agent") + 1] == "codex"
        env = kwargs["env"]
        assert env["GEAK_CODEX_MODEL"] == "gpt-5.5"
        assert env["GEAK_CODEX_EFFORT"] == "high"
        assert env["OPENAI_BASE_URL"] == proxy.base_url
        assert env["OPENAI_API_KEY"] == "adbench-local"
        assert "GEAK_MODEL" not in env and "GEAK_AGENT_PROFILE" not in env
        assert "model_provider=wrong" not in env["GEAK_CODEX_EXTRA_ARGS"]
        assert "--skip-git-repo-check" in env["GEAK_CODEX_EXTRA_ARGS"]
        assert not any("test-upstream-key" in value for value in env.values())
        wf = json.loads(Path(command[command.index("--args-file") + 1]).read_text())
        assert wf["mode"] == "author" and wf["budget"] == 2
        candidate = kwargs["cwd"] / "exp" / "round1" / "workspace/kernel_src/candidate.py"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("# codex-geak-deliverable")
        proxy.calls_used = 12
        if outcome == "timeout":
            raise driver.subprocess.TimeoutExpired(command, 1)
        Path(command[command.index("--result-file") + 1]).write_text(
            json.dumps({"eval_dir": str(candidate.parents[2])}))
        return SimpleNamespace(stdout="WORKFLOW_RESULT {}", stderr="", returncode=int(outcome == "failure"))

    monkeypatch.setattr(driver.subprocess, "run", run)
    run_dir = tmp_path / "run"
    assert driver.main(arguments(task, run_dir, "--geak-root", str(root), "--effort", "high")) == 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["config"]["backend"] == "codex"
    assert record["status"] == {"success": "completed", "failure": "workflow_failed",
                                "timeout": "time_limit"}[outcome]
    assert record["llm_budget"]["used"] == 12
    assert "codex-geak-deliverable" in Path(record["best_program"]).read_text()
    assert metered.instances[-1].closed


@pytest.mark.parametrize("agent", ["claude_code", "geak"])
def test_anthropic_agents_need_the_upstream_key_before_a_run_starts(
        task, tmp_path, monkeypatch, metered, agent):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    args = metered.common.parser("test").parse_args(arguments(task, tmp_path / "run"))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        metered.common.start(args, agent, {})
    assert not (tmp_path / "run").exists() and not metered.instances


@pytest.mark.parametrize("has_correct, chosen", [(True, "best-correct"), (False, "last-attempt")])
def test_gepa_exhaustion_returns_the_best_correct_or_last_evaluated_source(
        task, tmp_path, monkeypatch, metered, has_correct, chosen):
    from evograd.adbench_tools import evaluate

    driver = load_driver("gepa")
    skeleton = (task.root / "skeleton.py").read_text()
    sources = [skeleton + f"\n# {name}\n" for name in
               ("best-correct", "slower-correct", "last-attempt")]
    fake_gepa = ModuleType("gepa")
    fake_api = ModuleType("gepa.optimize_anything")
    fake_api.OptimizeAnythingConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    def optimize(seed, *, evaluator, objective, background, config):
        assert config.max_evals == 11
        reflection = config.engine_config["reflection"]
        assert reflection["reflection_lm"] == "openai/gpt-5.5"
        assert reflection["reflection_lm_kwargs"]["api_base"] == metered.instances[-1].base_url
        assert background == task.statement
        for source in sources:
            evaluator(source)
        metered.instances[-1].exhaust()
        raise RuntimeError("model request budget exhausted")

    def evaluate_source(task_, source, **kwargs):
        correct = has_correct and "last-attempt" not in source
        score = 3.0 if "best-correct" in source else 1.0
        return SimpleNamespace(correct=correct, combined_score=score,
                               agent_score=lambda: score if correct else 0.0,
                               feedback=lambda: "fake evaluator feedback")

    fake_api.optimize_anything = optimize
    monkeypatch.setitem(sys.modules, "gepa", fake_gepa)
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", fake_api)
    monkeypatch.setattr(evaluate, "evaluate_source", evaluate_source)
    run_dir = tmp_path / "gepa"
    assert driver.main(arguments(task, run_dir)) == 0
    assert_submission(task, run_dir, "gepa", chosen)
    assert metered.instances[-1].closed


@pytest.mark.parametrize("agent, archive_kind", [
    ("evograd", "correct"), ("openevolve", "correct"),
    ("evograd", "none"), ("openevolve", "incorrect"),
])
def test_evolution_exhaustion_recovers_archive_or_frontend_seed(
        task, tmp_path, monkeypatch, metered, agent, archive_kind):
    from evograd.evolve import run as evolution

    driver = load_driver(agent)
    skeleton = (task.root / "skeleton.py").read_text()

    def seed(task_, ledger, **kwargs):
        program = ledger.root / "frontend_seed.py"
        program.write_text(skeleton + "\n# frontend-seed\n")
        return program

    def evolve(op, **kwargs):
        assert kwargs["iterations"] == 10
        assert kwargs["primary_model"] == kwargs["secondary_model"] == "gpt-5.5"
        assert kwargs["api_base"] == metered.instances[-1].base_url
        if archive_kind != "none":
            archive = kwargs["output_dir"] / "programs"
            archive.mkdir(parents=True)
            entries = []
            for name, correct, score in (("best-archive", archive_kind == "correct", 2.0),
                                         ("wrong-faster", False, 100.0)):
                (archive / f"{name}.py").write_text(skeleton + f"\n# {name}\n")
                entries.append({"file": f"{name}.py", "metrics": {
                    "correct": correct, "combined_score": score}})
            (archive / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
        metered.instances[-1].exhaust()
        raise RuntimeError("model request budget exhausted")

    monkeypatch.setattr(metered.common, "unified_seed", seed)
    monkeypatch.setattr(evolution, "run_evolve", evolve)
    run_dir = tmp_path / agent
    assert driver.main(arguments(task, run_dir)) == 0
    chosen = {"correct": "best-archive", "none": "frontend-seed", "incorrect": "wrong-faster"}
    assert_submission(task, run_dir, agent, chosen[archive_kind])
    assert metered.instances[-1].closed


def test_an_a_seed_spends_requests_that_evolution_then_does_not_get(
        task, tmp_path, monkeypatch, metered):
    from evograd.evolve import run as evolution

    driver = load_driver("evograd")
    skeleton = (task.root / "skeleton.py").read_text()

    def seed(task_, ledger, args):
        assert args.frontend == "a"
        metered.instances[-1].calls_used = 3
        program = ledger.root / "a_seed.py"
        program.write_text(skeleton + "\n# a-seed\n")
        return program

    def evolve(op, **kwargs):
        assert kwargs["iterations"] == 7 and kwargs["pin_fixed_code"]
        assert kwargs["allowed_primitives"] == task.allowed_primitives
        metered.instances[-1].exhaust()
        raise RuntimeError("model request budget exhausted")

    monkeypatch.setattr(driver, "pipeline_seed", seed)
    monkeypatch.setattr(evolution, "run_evolve", evolve)
    run_dir = tmp_path / "evograd-a"
    assert driver.main(arguments(task, run_dir, "--frontend", "a")) == 0
    record = assert_submission(task, run_dir, "evograd", "a-seed")
    assert (record["config"]["frontend"], record["seed_requests"], record["iterations"]) == ("a", 3, 7)


@pytest.mark.parametrize("returncode", [0, 1])
def test_a_pipeline_seed_counts_only_when_its_pipeline_exits_zero(
        task, tmp_path, monkeypatch, returncode):
    driver = load_driver("evograd")
    events = []
    ledger = SimpleNamespace(root=tmp_path, log_evaluation=events.append)
    published = tmp_path / "seed" / "initial_program_autograd_pair.py"

    def run(command, **kwargs):
        assert command[3:5] == ["seed", "d"] and "--static-shapes" in command
        published.parent.mkdir(parents=True)
        published.write_text("# D publishes its seed before the static rules and the oracle\n")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(driver.subprocess, "run", run)
    seed = driver.pipeline_seed(task, ledger, SimpleNamespace(frontend="d"))
    assert seed == (published if returncode == 0 else None)
    assert len(events) == (1 if returncode == 0 else 0)


@pytest.mark.parametrize("agent", ["gepa", "claude_code"])
@pytest.mark.parametrize("config", [{"budget": 10, "budget_unit": "legacy_native"},
                                      {"budget": 30, "budget_unit": "api_requests"}])
def test_bench_refuses_to_reuse_finished_runs_with_an_incompatible_budget(
        task, tmp_path, monkeypatch, metered, config, agent):
    bench = load_driver("")
    root = tmp_path / "benchmark"
    cid = task.spec["case_id"]
    shutil.copytree(task.root, root / "tasks" / bench.case_dir(cid))
    run_dir = root / "runs" / agent / bench.case_dir(cid)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": "completed", "config": config}))
    before = (run_dir / "run.json").read_text()
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: pytest.fail("old run was relaunched"))
    args = argparse.Namespace(root=root, case_id=[cid], model="qwen3_0_6b", agent=agent,
                              budget=10, redo=False, agent_args=[], time_limit=None)
    with pytest.raises(SystemExit, match="requested 10-request budget"):
        bench.cmd_run(args)
    assert (run_dir / "run.json").read_text() == before


def test_bench_passes_the_requested_api_cap_and_packages_an_exhausted_run(
        task, tmp_path, monkeypatch, metered):
    bench = load_driver("")
    root = tmp_path / "benchmark"
    cid = task.spec["case_id"]
    shutil.copytree(task.root, root / "tasks" / bench.case_dir(cid))
    launched = []

    def run(command, **kwargs):
        launched.append(command)
        assert command.count("--budget") == 1
        assert command[command.index("--budget") + 1] == "4"
        assert command[command.index("--eval-budget") + 1] == "3"
        assert command[command.index("--model") + 1] == "gpt-5.5"
        assert "--no-score" in command
        run_dir = Path(command[command.index("--run-dir") + 1])
        run_dir.mkdir()
        best = run_dir / "best.py"
        best.write_text((task.root / "skeleton.py").read_text() + "\n# retained-by-bench\n")
        (run_dir / "run.json").write_text(json.dumps({
            "status": "budget_exhausted", "best_program": str(best),
            "config": {"budget": 4, "budget_unit": "api_requests"}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench.subprocess, "run", run)
    args = argparse.Namespace(root=root, case_id=[cid], model="qwen3_0_6b", agent="codex",
                              budget=4, redo=False, time_limit=None,
                              agent_args=["--model", "gpt-5.5", "--eval-budget", "3"])
    assert bench.cmd_run(args) == 0
    submission = root / "runs" / "codex" / bench.case_dir(cid) / "submission.py"
    assert "retained-by-bench" in submission.read_text()
    # A compatible finished run is kept, including a budget-exhausted deliverable.
    assert bench.cmd_run(args) == 0
    assert len(launched) == 1


def test_bench_preserves_an_interrupted_run_and_its_spent_api_budget(
        task, tmp_path, monkeypatch, metered):
    bench = load_driver("")
    root = tmp_path / "benchmark"
    cid = task.spec["case_id"]
    shutil.copytree(task.root, root / "tasks" / bench.case_dir(cid))
    run_dir = root / "runs" / "gepa" / bench.case_dir(cid)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "status": "running", "config": {"budget": 10, "budget_unit": "api_requests"}}))
    (run_dir / "api_calls.jsonl").write_text(
        json.dumps({"call": 1, "method": "POST", "path": "/responses"}) + "\n")
    (run_dir / "candidate.py").write_text("# work from the interrupted run\n")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: pytest.fail("run was relaunched"))
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: pytest.fail("run was deleted"))
    args = argparse.Namespace(root=root, case_id=[cid], model="qwen3_0_6b", agent="gepa",
                              budget=10, redo=False, agent_args=[], time_limit=None)
    with pytest.raises(SystemExit, match="--redo"):
        bench.cmd_run(args)
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before
    assert not run_dir.with_suffix(".log").exists()


@pytest.mark.parametrize("agent, budget_unit, expected_calls", [
    ("claude_code", "legacy_native", None),  # a run from before Claude Code was metered
    ("geak", "optimization_directions", 3),  # counted, not capped
    ("seed", "none", 0),
    ("gepa", "api_requests", 3),
])
def test_run_summary_distinguishes_api_requests_from_cli_usage_events(
        task, tmp_path, agent, budget_unit, expected_calls):
    from evograd.adbench_tools.ledger import API_CALLS_FILE, RunLedger, append_line

    ledger = RunLedger(tmp_path / agent, agent=agent, task=task,
                       config={"budget_unit": budget_unit})
    if agent != "seed":
        ledger.log_llm({"event": "turn.completed", "usage": {"input_tokens": 5}})
        ledger.log_llm({"event": "turn.completed", "usage": {"input_tokens": 7}})
    if expected_calls:
        for index in range(1, 4):
            append_line(ledger.root / API_CALLS_FILE,
                        {"call": index, "method": "POST", "path": "/responses"})
    summary = ledger.summary()
    assert summary["llm_calls_recorded"] == expected_calls
    assert summary["llm_usage_events"] == (0 if agent == "seed" else 2)
    if agent != "seed":
        assert summary["llm_tokens_recorded"] == {"input_tokens": 12}
