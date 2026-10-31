"""CPU-only tests for the Slurm launcher's allocation and resume decisions."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("batch", REPO / "candidates/adbench_search.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(batch, "SHARED", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.ids = ["rmsnorm/rmsnorm", "qwen3_swiglu_mlp/swiglu_mlp"] + [f"task{i}/case" for i in range(6)]
        write_json(self.root / "cases.json", {"qwen3_0_6b": {"corpus": "/shared/manifest.json", "cases": self.ids}})
        for cid in self.ids:
            write_json(self.root / "openevolve/qwen3_0_6b/tasks" / cid.replace("/", "__") / "task.json",
                       {"model": "qwen3_0_6b", "case_id": cid, "level": 1})

    def finished(self, path, state="completed", has_program=True):
        write_json(path / "run.json", {"status": state, "wall_seconds": 600,
                   "config": {"model": "gpt-5.5", "budget": 10, "budget_unit": "api_requests"}})
        write_json(path / "submission.json", {"submission": str(path / "submission.py") if has_program else None})
        if has_program:
            (path / "submission.py").write_text("pass\n")
        (path / "api_calls.jsonl").write_text('{}\n' * 10)

    def smoke(self):
        for cid in batch.SMOKE_CASES:
            self.finished(self.root / "smoke/openevolve/qwen3_0_6b/runs/openevolve" / cid.replace("/", "__"))
        result = self.root / "smoke/openevolve/qwen3_0_6b/evaluation/openevolve/evaluation.json"
        write_json(result, {"summary": {"model": "qwen3_0_6b"},
                            "cases": {cid: {"submitted": True} for cid in batch.SMOKE_CASES}})
        return result

    def test_model_filter_and_manifest_mismatch(self):
        self.assertEqual(len(list(batch.cases("openevolve", "qwen3_0_6b"))), 8)
        write_json(self.root / "cases.json", {"qwen3_0_6b": {"cases": self.ids[:-1]}})
        with self.assertRaises(ValueError):
            list(batch.cases("openevolve", "qwen3_0_6b"))

    def test_running_and_unpackaged_are_not_done(self):
        path = batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])
        self.finished(path, state="running")
        self.assertEqual(batch.status(path), "unfinished")
        self.finished(path)
        (path / "submission.json").unlink()
        self.assertEqual(batch.status(path), "unfinished")

    def test_log_only_failure_is_not_retried(self):
        path = batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])
        path.parent.mkdir(parents=True)
        path.with_suffix('.log').write_text('failed before ledger\n')
        self.assertEqual(batch.status(path), "unfinished")

    def test_terminal_failure_counts_as_done_but_not_kernel(self):
        for cid in self.ids:
            self.finished(batch.run_dir("openevolve", "qwen3_0_6b", cid), "no_program", False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.plan("openevolve", "qwen3_0_6b"), 3)
        self.assertFalse(batch.packaged(batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])))

    def test_pending_preflight_and_suspended_smoke(self):
        result = self.smoke()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.plan("openevolve", "qwen3_0_6b"), 0)
        write_json(result, {"suspended": "deadline", "summary": {"old": True}})
        with self.assertRaises(ValueError):
            batch.smoke_longest("openevolve")

    def test_skip_smoke_plans_and_uses_minimum_time_guard(self):
        with patch.object(batch, "smoke_longest", side_effect=AssertionError("smoke checked")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(batch.plan("openevolve", "qwen3_0_6b", skip_smoke=True), 0)
                self.assertEqual(batch.run_locked("openevolve", "qwen3_0_6b",
                                                  batch.time.time() + 1200,
                                                  skip_smoke=True), 0)
        self.assertIn("smoke check skipped", output.getvalue())
        self.assertIn("Time cutoff reached", output.getvalue())
        self.assertFalse((self.root / "smoke").exists())

    def test_skip_smoke_starts_with_one_hour_remaining(self):
        with patch.object(batch, "smoke_longest", side_effect=AssertionError("smoke checked")), \
             patch.object(batch.subprocess, "Popen") as popen, \
             patch.object(batch, "kill_group"), \
             contextlib.redirect_stdout(io.StringIO()):
            popen.return_value.wait.return_value = 1
            self.assertEqual(batch.run_locked("openevolve", "qwen3_0_6b",
                                              batch.time.time() + 3600,
                                              skip_smoke=True), 2)
        popen.assert_called_once()
        self.assertGreater(popen.return_value.wait.call_args.kwargs["timeout"], 3000)

    def test_codex_worker_uses_request_budget_and_skips_smoke(self):
        shutil.copytree(self.root / 'openevolve', self.root / 'codex')
        with patch.object(batch, 'smoke_longest', side_effect=AssertionError('smoke checked')), \
             patch.object(batch.subprocess, 'Popen') as popen, \
             patch.object(batch, 'kill_group'), \
             contextlib.redirect_stdout(io.StringIO()):
            popen.return_value.wait.return_value = 1
            self.assertEqual(batch.run_locked('codex', 'qwen3_0_6b',
                                              batch.time.time() + 7200,
                                              skip_smoke=True), 2)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('--agent') + 1], 'codex')
        self.assertEqual(command[command.index('--budget') + 1], '10')
        self.assertEqual(command[command.index('--') + 1:], batch.agent_flags('codex'))

    def test_running_only_is_not_complete(self):
        for cid in self.ids:
            self.finished(batch.run_dir("openevolve", "qwen3_0_6b", cid), "running")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.plan("openevolve", "qwen3_0_6b"), 4)

    def test_mixed_pending_and_unfinished_do_not_request_gpu(self):
        path = batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".log").write_text("interrupted\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.plan("openevolve", "qwen3_0_6b",
                                        skip_smoke=True), 4)

    def test_smoke_model_and_budget_are_checked(self):
        self.smoke()
        record = self.root / 'smoke/openevolve/qwen3_0_6b/runs/openevolve/rmsnorm__rmsnorm/run.json'
        data = json.loads(record.read_text())
        data['config']['model'] = 'wrong-model'
        write_json(record, data)
        with self.assertRaises(ValueError):
            batch.smoke_longest('openevolve')
        data['config']['model'] = 'gpt-5.5'
        data['config']['budget'] = 20
        write_json(record, data)
        with self.assertRaises(ValueError):
            batch.smoke_longest('openevolve')

    def test_plan_flags_pin_all_agents_and_geak_backend(self):
        for agent in batch.AGENTS:
            flags = batch.agent_flags(agent)
            self.assertEqual(flags[flags.index("--model") + 1], "gpt-5.5")
            self.assertNotIn("--claude-model", flags)
        flags = batch.agent_flags("geak")
        self.assertEqual(flags[flags.index("--backend") + 1], "codex")

    def test_geak_smoke_requires_codex_even_if_model_metadata_says_gpt(self):
        self.smoke()
        shutil.copytree(self.root / "smoke/openevolve", self.root / "smoke/geak")
        model = self.root / "smoke/geak/qwen3_0_6b"
        (model / "runs/openevolve").rename(model / "runs/geak")
        (model / "evaluation/openevolve").rename(model / "evaluation/geak")
        for cid in batch.SMOKE_CASES:
            path = model / "runs/geak" / cid.replace("/", "__") / "run.json"
            data = batch.read_json(path)
            data["config"].update(budget=2, budget_unit="optimization_directions",
                                  backend="codex", api="openai")
            write_json(path, data)
        self.assertEqual(batch.smoke_longest("geak"), 600)
        data["config"].update(backend="claude", api="anthropic", claude_model="claude-opus-4-8")
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Model/backend mismatch"):
            batch.smoke_longest("geak")

    def test_finished_runs_from_another_model_are_not_silently_reused(self):
        path = batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])
        self.finished(path)
        data = batch.read_json(path / "run.json")
        data["config"]["model"] = "gpt-5.6-sol"
        write_json(path / "run.json", data)
        with self.assertRaisesRegex(ValueError, "Model/backend mismatch"):
            batch.plan("openevolve", "qwen3_0_6b")
        self.assertTrue((path / "submission.py").is_file())

    def test_deadline_comes_from_slurm(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(batch.subprocess, 'check_output', return_value='JobId=123 EndTime=2026-10-28T14:30:00'):
            self.assertEqual(batch.job_end_time(), batch.datetime(2026, 10, 28, 14, 30).timestamp())

    def test_missing_key_stops_before_allocation_or_search(self):
        for agent, key in (("openevolve", "OPENAI_API_KEY"),
                           ("gepa", "OPENAI_API_KEY"),
                           ("geak", "OPENAI_API_KEY")):
            with patch.dict(os.environ, {key: ""}), patch.object(batch, "job_end_time") as deadline:
                with self.assertRaisesRegex(ValueError, key):
                    batch.run(agent, "qwen3_0_6b")
                deadline.assert_not_called()

    def test_lock_rejects_overlapping_agent(self):
        with batch.agent_lock("openevolve"):
            with self.assertRaises(ValueError):
                with batch.agent_lock("openevolve"):
                    pass

    def test_insufficient_allocation_starts_nothing(self):
        self.smoke()
        with patch.object(batch.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            batch.run_locked("openevolve", "qwen3_0_6b", batch.time.time() + 100)
        popen.assert_not_called()

    def test_unfinished_case_stops_before_next_case(self):
        self.smoke()
        path = batch.run_dir("openevolve", "qwen3_0_6b", self.ids[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".log").write_text("interrupted\n")
        with patch.object(batch.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_locked("openevolve", "qwen3_0_6b", batch.time.time() + 7200), 2)
        popen.assert_not_called()

    def test_bench_failure_cleans_group_and_stops(self):
        self.smoke()
        with patch.object(batch.subprocess, "Popen") as popen, patch.object(batch, "kill_group") as cleanup:
            popen.return_value.wait.return_value = 1
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run_locked("openevolve", "qwen3_0_6b", batch.time.time() + 7200), 2)
            self.assertEqual(popen.call_count, 1)
            cleanup.assert_called_once_with(popen.return_value)

    def test_parallel_cases_are_unique_and_bounded(self):
        launched = []
        active_count = [0, 0]
        outer = self

        class Process:
            def __init__(self, model, cid):
                self.model, self.cid = model, cid
                self.done = False
                active_count[0] += 1
                active_count[1] = max(active_count)

            def poll(self):
                if not self.done:
                    outer.finished(batch.run_dir("openevolve", self.model, self.cid))
                    active_count[0] -= 1
                    self.done = True
                return 0

        def launch(agent, model, cid, remaining, lock):
            launched.append(cid)
            self.assertEqual(lock, "/shared/gpu.lock")
            return Process(model, cid)

        with patch.object(batch, "launch_search", side_effect=launch), \
             patch.object(batch, "kill_group"), patch.object(batch.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_parallel("openevolve", list(batch.cases("openevolve", "qwen3_0_6b")),
                                                 batch.time.time() + 7200, 1200, 2,
                                                 "/shared/gpu.lock"), 0)
        self.assertEqual(len(launched), len(self.ids))
        self.assertEqual(len(set(launched)), len(self.ids))
        self.assertEqual(active_count[1], 2)

    def test_workers_reject_unsupported_agents(self):
        for agent in ("geak", "codex"):
            with self.assertRaisesRegex(ValueError, "only for GEPA and OpenEvolve"):
                batch.plan(agent, None, skip_smoke=True, workers=2)

    def test_parallel_failure_drains_without_starting_replacements(self):
        launched = []
        outer = self

        class Process:
            def __init__(self, model, cid, rc):
                self.model, self.cid, self.rc = model, cid, rc

            def poll(self):
                outer.finished(batch.run_dir("openevolve", self.model, self.cid),
                               "completed" if self.rc == 0 else "no_program",
                               has_program=self.rc == 0)
                return self.rc

        def launch(agent, model, cid, remaining, lock):
            launched.append(cid)
            return Process(model, cid, 1 if len(launched) == 1 else 0)

        with patch.object(batch, "launch_search", side_effect=launch), \
             patch.object(batch, "kill_group"), patch.object(batch.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_parallel("openevolve", list(batch.cases("openevolve", "qwen3_0_6b")),
                                                 batch.time.time() + 7200, 1200, 2,
                                                 "/shared/gpu.lock"), 2)
        self.assertEqual(len(launched), 2)

    def test_allocation_expiry_cleans_all_parallel_searches(self):
        from unittest.mock import Mock
        processes = [Mock(), Mock()]
        for proc in processes:
            proc.poll.return_value = None
        with patch.object(batch, "launch_search", side_effect=processes), \
             patch.object(batch, "kill_group") as cleanup, \
             patch.object(batch.time, "sleep"), \
             patch.object(batch.time, "time", side_effect=[0, 0, 0, 7100]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_parallel("openevolve", list(batch.cases("openevolve", "qwen3_0_6b")),
                                                 7200, 1200, 2, "/shared/gpu.lock"), 2)
        self.assertEqual(cleanup.call_count, 2)

    def test_launch_keeps_individual_budget_and_shared_gpu_lock(self):
        with patch.object(batch.subprocess, "Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.launch_search("gepa", "qwen3_0_6b", self.ids[0], 3600, "/shared/gpu.lock")
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--budget") + 1], "10")
        self.assertEqual(popen.call_args.kwargs["env"]["ADBENCH_GPU_LOCK"], "/shared/gpu.lock")

    def test_openevolve_watchdog_allows_lock_waiting(self):
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location("test_openevolve_shared", REPO / "candidates/openevolve/run.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        yaml = SimpleNamespace(safe_load=lambda text: {"prompt": {}, "evaluator": {"timeout": 900}},
                               safe_dump=lambda data, **kwargs: json.dumps(data))
        render = SimpleNamespace(render_config=lambda *args, **kwargs: "config")
        task = SimpleNamespace(op=lambda: None, statement="task")
        args = SimpleNamespace(budget=10, model="gpt-5.5", api_base="example", prompt="statement", seed=0)
        path = self.root / "config.json"
        with patch.dict(sys.modules, {"yaml": yaml, "evograd.evolve.run": render}):
            with patch.dict(os.environ, {"ADBENCH_GPU_LOCK": ""}):
                driver.write_config(task, args, path)
                self.assertEqual(json.loads(path.read_text())["evaluator"]["timeout"], 900)
            with patch.dict(os.environ, {"ADBENCH_GPU_LOCK": "/shared/gpu.lock"}):
                driver.write_config(task, args, path)
                self.assertEqual(json.loads(path.read_text())["evaluator"]["timeout"], 6000)

    def test_parent_exit_does_not_leave_group_alive(self):
        marker = self.root / "child-finished"
        child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
        parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1]])"
        proc = subprocess.Popen([sys.executable, "-c", parent, child], start_new_session=True)
        proc.wait(timeout=5)
        batch.kill_group(proc)
        self.assertFalse(marker.exists())


class LauncherTests(unittest.TestCase):
    def test_codex_status_and_run_dispatch(self):
        with patch.object(sys, 'argv', ['adbench_search.py', 'status', '--agent', 'codex']), \
             patch.object(batch, 'report', return_value=0) as report:
            self.assertEqual(batch.main(), 0)
            report.assert_called_once_with('codex', None)
        with patch.object(sys, 'argv', ['adbench_search.py', 'run', '--agent', 'codex',
                                       '--skip-smoke']), \
             patch.object(batch, 'run', return_value=0) as run, \
             patch.object(batch.signal, 'signal'):
            self.assertEqual(batch.main(), 0)
            run.assert_called_once_with('codex', None, True, 1)
        self.assertEqual(batch.agent_flags('codex'),
                         ['--model', 'gpt-5.5', '--effort', 'high',
                          '--time-limit', '4800', '--eval-timeout', '900'])

    def test_submit_only_when_pending_and_no_overlapping_job(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            candidates = root / 'repo/candidates'
            candidates.mkdir(parents=True)
            shutil.copy(REPO/'candidates/submit_adbench_search.sh', candidates)
            home = root/'home'
            home.mkdir()
            (home/'.adbench_keys.txt').touch()
            binaries = root/'bin'
            binaries.mkdir()
            for name, source in {
                'python': '#!/bin/bash\necho preflight\nexit "$PREFLIGHT_RC"\n',
                'flock': '#!/bin/bash\nexit 0\n',
                'squeue': '#!/bin/bash\nprintf "%s\\n" "$MOCK_JOBS"\n',
                'sbatch': '#!/bin/bash\nprintf "%s\\n" "$@" > "$SUBMITTED"\n',
            }.items():
                path = binaries/name
                path.write_text(source)
                path.chmod(0o755)
            submitted = root/'submitted'
            env = {**os.environ, 'HOME':str(home), 'ADBENCH_ENV':str(root),
                   'PATH':str(binaries)+os.pathsep+os.environ['PATH'], 'PREFLIGHT_RC':'3',
                   'MOCK_JOBS':'', 'SUBMITTED':str(submitted), 'USER':'test'}
            command = ['bash',str(candidates/'submit_adbench_search.sh'),'openevolve','qwen3_0_6b']
            subprocess.run(command, env=env, check=True, capture_output=True)
            self.assertFalse(submitted.exists())
            env['PREFLIGHT_RC']='0'
            subprocess.run(command, env=env, check=True, capture_output=True)
            self.assertIn('--model\nqwen3_0_6b', submitted.read_text())
            self.assertIn('candidates/adbench_search.sbatch', submitted.read_text())
            submitted.unlink()
            subprocess.run(command + ['--skip-smoke'], env=env, check=True,
                           capture_output=True)
            self.assertIn('--skip-smoke', submitted.read_text())
            submitted.unlink()
            subprocess.run(command + ['--skip-smoke', '--workers', '2'], env=env,
                           check=True, capture_output=True)
            self.assertIn('--workers\n2', submitted.read_text())
            submitted.unlink()
            subprocess.run(command[:3] + ['--skip-smoke'], env=env, check=True,
                           capture_output=True)
            self.assertIn('--skip-smoke', submitted.read_text())
            submitted.unlink()
            subprocess.run(command + ['--partition', 'ghx4-interactive', '--skip-smoke'],
                           env=env, check=True, capture_output=True)
            self.assertIn('candidates/adbench_search_interactive.sbatch', submitted.read_text())
            self.assertIn('--model\nqwen3_0_6b', submitted.read_text())
            self.assertIn('--skip-smoke', submitted.read_text())
            self.assertNotIn('--partition', submitted.read_text())
            submitted.unlink()
            for options in (['--partition'], ['--partition', 'bad-queue'],
                            ['--partition', 'ghx4', '--partition', 'ghx4-interactive'],
                            ['--workers', '0'], ['--workers', 'invalid']):
                result = subprocess.run(command + options, env=env, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(submitted.exists())
            # Exercise the Python entry point through the real shell launcher.
            shutil.copy(REPO/'candidates/adbench_search.py', candidates)
            python_command = [sys.executable, str(candidates/'adbench_search.py'),
                              'submit', '--agent', 'openevolve', '--model', 'qwen3_0_6b',
                              '--skip-smoke']
            for partition, script in [('ghx4', 'adbench_search.sbatch'),
                                      ('ghx4-interactive', 'adbench_search_interactive.sbatch')]:
                subprocess.run(python_command + ['--partition', partition, '--workers', '2'], env=env,
                               check=True, capture_output=True)
                self.assertIn(f'candidates/{script}', submitted.read_text())
                self.assertIn('--workers\n2', submitted.read_text())
                submitted.unlink()
            subprocess.run(python_command, env=env, check=True, capture_output=True)
            self.assertIn('candidates/adbench_search.sbatch', submitted.read_text())
            submitted.unlink()
            env['MOCK_JOBS']='123|adb-openevolve-all|RUNNING'
            subprocess.run(python_command + ['--partition', 'ghx4-interactive'], env=env,
                           check=True, capture_output=True)
            self.assertFalse(submitted.exists())

    def test_partition_flag_rejected_outside_submission(self):
        for action in ('run', 'plan', 'status', 'check-smoke'):
            result = subprocess.run(
                [sys.executable, str(REPO/'candidates/adbench_search.py'), action,
                 '--agent', 'openevolve', '--partition', 'ghx4-interactive'],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn('--partition applies only to submit', result.stderr)


if __name__ == '__main__':
    unittest.main()
