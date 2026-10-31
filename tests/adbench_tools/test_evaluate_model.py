"""CPU preflight and dispatch tests; the wrapper never evaluates candidate code."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluate_model", REPO / "candidates/evaluate_model.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.shared = self.root / "shared"
        self.out = self.root / "results"
        self.ids = ["mlp/stage0", "mlp/stage1"]
        self.corpus = self.root / "corpus" / "manifest.json"
        write(self.corpus, {})
        write(self.shared / "cases.json", {"convnextv2_tiny": {
            "corpus": str(self.corpus.parent), "cases": self.ids}})
        for cid in self.ids:
            directory = self.shared / "openevolve/convnextv2_tiny"
            write(directory / "tasks" / cid.replace("/", "__") / "task.json", {
                "model": "convnextv2_tiny", "case_id": cid, "level": 2})
            run = self.run_dir(cid)
            write(run / "run.json", {"status": "completed"})
            program = (b"# EVOLVE-BLOCK-START\nraise RuntimeError()\n# EVOLVE-BLOCK-END\n"
                       b"# original deployment\n")
            (run / "submission.py").write_bytes(program)
            write(run / "submission.json", {
                "case_id": cid, "agent": "openevolve", "submission": "old/path/submission.py",
                "program_sha256": hashlib.sha256(program).hexdigest()})
        self.argv = ["convnextv2_tiny", "--shared-root", str(self.shared),
                     "--submission-root", str(self.shared),
                     "--out-root", str(self.out)]

    def run_dir(self, cid):
        return self.shared / "openevolve/convnextv2_tiny/runs/openevolve" / cid.replace("/", "__")

    def invoke(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main(self.argv + list(extra))

    def test_dry_run_checks_every_submission_without_writing_or_launching(self):
        with patch.object(runner.subprocess, "run") as launch:
            self.assertEqual(self.invoke(["--dry-run"]), 0)
            launch.assert_not_called()
        self.assertFalse(self.out.exists())
        (self.run_dir(self.ids[1]) / "submission.py").write_text("tampered")
        self.assertEqual(self.invoke(["--dry-run"]), 1)

    def test_missing_and_running_cases_stop_before_launch(self):
        ledger = self.run_dir(self.ids[1]) / "run.json"
        write(ledger, {"status": "running"})
        self.assertEqual(self.invoke(["--dry-run"]), 1)
        ledger.unlink()
        self.assertEqual(self.invoke(["--dry-run"]), 1)

    def test_dispatches_all_cases_and_propagates_resume_exit(self):
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)) as launch:
            self.assertEqual(self.invoke(), 3)
            command = launch.call_args.args[0]
            self.assertEqual(command.count("--candidate"), 2)
            self.assertIn("evograd.evaluation.protocol", command)
            self.assertEqual(launch.call_args.kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(self.invoke(), 3)
        path = self.out / "convnextv2_tiny/evaluation/openevolve/selection.json"
        self.assertEqual(set(json.loads(path.read_text())["candidates"]), set(self.ids))

    def test_reuse_flags_forward_to_single_agent_and_prevent_mixing_results(self):
        flag = "--share-gate3-native"
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke([flag]), 0)
            self.assertIn(flag, launch.call_args.args[0])
            launch.reset_mock()
            write(self.out / "convnextv2_tiny/evaluation/openevolve/evaluation.json", {
                "settings": {"calibration_reuse": {"share_gate3_native": True}}})
            self.assertEqual(self.invoke(), 1)
            launch.assert_not_called()

    def test_terminal_failure_is_recorded_without_fabricated_candidate(self):
        write(self.run_dir(self.ids[1]) / "submission.json", {
            "case_id": self.ids[1], "agent": "openevolve", "submission": None})
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke(), 0)
            self.assertEqual(launch.call_args.args[0].count("--candidate"), 1)
        saved = json.loads((self.out / "convnextv2_tiny/evaluation/openevolve/selection.json").read_text())
        self.assertIn(self.ids[1], saved["search_failures"])

    def test_changed_frozen_selection_requires_new_output_root(self):
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke(), 0)
            launch.reset_mock()
            write(self.run_dir(self.ids[1]) / "submission.json", {
                "case_id": self.ids[1], "agent": "openevolve", "submission": None})
            self.assertEqual(self.invoke(), 1)
            launch.assert_not_called()

    def redeploy(self):
        root = self.root / "redeploy"
        for cid in self.ids:
            target = root / self.run_dir(cid).relative_to(self.shared)
            shutil.copytree(self.run_dir(cid), target)
            program = target / "submission.py"
            program.write_bytes(program.read_bytes().replace(b"original deployment", b"fixed deployment"))
            record = json.loads((target / "submission.json").read_text())
            record["program_sha256"] = hashlib.sha256(program.read_bytes()).hexdigest()
            write(target / "submission.json", record)
        return root

    def test_redeploy_changes_only_deployment_and_uses_repaired_hash(self):
        root = self.redeploy()
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke(["--submission-root", str(root)]), 0)
            command = launch.call_args.args[0]
            paths = [command[i + 1].split("=", 1)[1] for i, arg in enumerate(command) if arg == "--candidate"]
            self.assertTrue(all(Path(path).is_relative_to(root.resolve()) for path in paths))
        saved = json.loads((self.out / "convnextv2_tiny/evaluation/openevolve/selection.json").read_text())
        entry = saved["candidates"][self.ids[0]]
        self.assertNotEqual(entry["sha256"], entry["original_sha256"])

    def test_missing_redeploy_never_falls_back_to_original(self):
        self.assertEqual(self.invoke(["--submission-root", str(self.root / "missing"), "--dry-run"]), 1)

    def test_redeploy_rejects_changed_kernel_even_with_updated_hash(self):
        root = self.redeploy()
        target = root / self.run_dir(self.ids[0]).relative_to(self.shared)
        program = target / "submission.py"
        program.write_bytes(program.read_bytes().replace(b"raise RuntimeError()", b"pass"))
        record = json.loads((target / "submission.json").read_text())
        record["program_sha256"] = hashlib.sha256(program.read_bytes()).hexdigest()
        write(target / "submission.json", record)
        self.assertEqual(self.invoke(["--submission-root", str(root), "--dry-run"]), 1)

    def test_all_agents_get_one_group_job_with_separate_selections(self):
        source = self.shared / "openevolve"
        for agent in ("gepa", "codex", "seed"):
            target = self.shared / agent
            shutil.copytree(source, target)
            (target / "convnextv2_tiny/runs/openevolve").rename(target / f"convnextv2_tiny/runs/{agent}")
            for record_path in target.glob("*/runs/*/*/submission.json"):
                record = json.loads(record_path.read_text())
                record["agent"] = agent
                write(record_path, record)
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke(["--agent", "all"]), 0)
            self.assertEqual(launch.call_count, 1)
            self.assertIn("evograd.evaluation.model_group", launch.call_args.args[0])
        plan = json.loads((self.out / "convnextv2_tiny/all_agents.json").read_text())
        self.assertEqual([job["selection"]["agent"] for job in plan["jobs"]], ["openevolve", "gepa", "codex"])
        self.assertNotEqual(plan["jobs"][0]["out"], plan["jobs"][1]["out"])
        self.assertNotIn("calibration_reuse", plan)
        # A rejected attempt can leave a descriptor with different reuse
        # settings. Existing reports, not that descriptor, govern compatibility.
        plan_path = self.out / "convnextv2_tiny/all_agents.json"
        plan["calibration_reuse"] = {name: True for name in runner.REUSE_FLAGS}
        write(plan_path, plan)
        for job in plan["jobs"]:
            write(Path(job["out"]) / "evaluation.json", {"settings": {"calibration_reuse": {}}})
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as launch:
            self.assertEqual(self.invoke(["--agent", "all"]), 0)
            launch.assert_called_once()
        self.assertNotIn("calibration_reuse", json.loads(plan_path.read_text()))
        flags = ["--" + name.replace("_", "-") for name in runner.REUSE_FLAGS]
        with patch.object(runner.subprocess, "run") as launch:
            self.assertEqual(self.invoke(["--agent", "all", *flags]), 1)
            launch.assert_not_called()
        plan = json.loads(plan_path.read_text())
        plan["weight_cache"] = False
        write(plan_path, plan)
        with patch.object(runner.subprocess, "run") as launch:
            self.assertEqual(self.invoke(["--agent", "all"]), 1)
            launch.assert_not_called()
        flags = ["--" + name.replace("_", "-") for name in runner.REUSE_FLAGS]
        self.out = self.root / "reuse-results"
        self.argv[self.argv.index("--out-root") + 1] = str(self.out)
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(self.invoke(["--agent", "all", *flags]), 0)
        plan = json.loads((self.out / "convnextv2_tiny/all_agents.json").read_text())
        self.assertEqual(plan["calibration_reuse"], {name: True for name in runner.REUSE_FLAGS})

    def test_all_requires_each_of_the_three_selected_agents(self):
        with patch.object(runner.subprocess, "run") as launch:
            self.assertEqual(self.invoke(["--agent", "all"]), 1)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
