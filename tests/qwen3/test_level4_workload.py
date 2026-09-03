"""Level-4 Qwen3 workload: identity, contract, reporting, and one real step.

Everything except the final integration case runs on CPU without Transformers,
because the properties that matter most here -- that the canonical workload
cannot drift, and that a debug variant cannot be mistaken for it -- are
properties of plain data.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.bench.workloads.qwen3.levels.level4.report import SCHEMA_VERSION, SmokeReport
from evograd.bench.workloads.qwen3.levels.level4.spec import (
    CANONICAL,
    QWEN3_0_6B,
    WorkloadSpec,
    WorkloadSpecError,
    analytic_parameter_count,
)

try:
    import transformers  # noqa: F401

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_TRANSFORMERS = False

REPO_ROOT = Path(__file__).resolve().parents[2]


def tiny_spec(**overrides) -> WorkloadSpec:
    """A two-layer Qwen3 that runs in a second on CPU. Deliberately *not*
    canonical: the point of the identity tests is that this can never be
    reported as the reference."""
    base = dict(
        device="cpu",
        dtype="float32",
        batch_size=2,
        seq_len=16,
        arch={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 64,
            # keep the special ids inside the shrunken vocabulary so the
            # config validator has nothing to warn about
            "bos_token_id": 0,
            "eos_token_id": 1,
        },
    )
    base.update(overrides)
    return CANONICAL.replace(**base)


class TestCanonicalWorkload(unittest.TestCase):
    """The canonical workload is exactly the one this milestone specifies."""

    def test_canonical_fields(self):
        self.assertEqual(CANONICAL.model_name, "Qwen3-0.6B")
        self.assertEqual(CANONICAL.batch_size, 2)
        self.assertEqual(CANONICAL.seq_len, 2048)
        self.assertEqual(CANONICAL.token_count, 4096)
        self.assertEqual(CANONICAL.dtype, "bfloat16")
        self.assertEqual(CANONICAL.device, "cuda")
        self.assertEqual(CANONICAL.attn_implementation, "sdpa")
        self.assertFalse(CANONICAL.use_cache)
        self.assertFalse(CANONICAL.gradient_checkpointing)
        self.assertTrue(CANONICAL.training)
        self.assertTrue(CANONICAL.is_canonical)

    def test_architecture_is_qwen3_0_6b(self):
        """The pinned config really is Qwen3-0.6B: 0.6B total, 0.44B
        non-embedding, as published on the model card. Derived from the config
        numbers, so a typo in any of them is caught without a GPU."""
        counts = analytic_parameter_count(QWEN3_0_6B)
        # Exact, not approximate: these are the numbers the built model reports
        # (``result.trainable_elements`` of the canonical GH200 run), so this
        # CPU test really does stand in for instantiating 0.6B parameters.
        self.assertEqual(counts["total"], 596_049_920)
        self.assertEqual(counts["non_embedding"], 440_467_456)
        self.assertEqual(counts["lm_head"], 0, "Qwen3-0.6B ties its embeddings")
        self.assertEqual(QWEN3_0_6B["num_hidden_layers"], 28)
        self.assertEqual(QWEN3_0_6B["num_attention_heads"], 16)
        self.assertEqual(QWEN3_0_6B["num_key_value_heads"], 8)
        self.assertEqual(QWEN3_0_6B["head_dim"], 128)
        self.assertEqual(QWEN3_0_6B["vocab_size"], 151936)
        self.assertTrue(QWEN3_0_6B["tie_word_embeddings"])

    def test_canonical_sequence_fits_the_context_window(self):
        self.assertLessEqual(CANONICAL.seq_len, QWEN3_0_6B["max_position_embeddings"])


class TestSpecContract(unittest.TestCase):
    """The three settings that define the workload are enforced, not defaulted."""

    def test_cache_is_rejected(self):
        with self.assertRaises(WorkloadSpecError) as ctx:
            CANONICAL.replace(use_cache=True)
        self.assertIn("use_cache", str(ctx.exception))
        self.assertIn("decode-time", str(ctx.exception))

    def test_gradient_checkpointing_is_rejected(self):
        with self.assertRaises(WorkloadSpecError) as ctx:
            CANONICAL.replace(gradient_checkpointing=True)
        self.assertIn("checkpointing", str(ctx.exception).lower())

    def test_eval_mode_is_rejected(self):
        with self.assertRaises(WorkloadSpecError):
            CANONICAL.replace(training=False)

    def test_unsupported_attention_backend_is_rejected_by_name(self):
        with self.assertRaises(WorkloadSpecError) as ctx:
            CANONICAL.replace(attn_implementation="flash_attention_2")
        message = str(ctx.exception)
        self.assertIn("flash_attention_2", message)
        self.assertIn("sdpa", message)

    def test_supported_alternative_backend_is_allowed_but_not_canonical(self):
        spec = CANONICAL.replace(attn_implementation="eager")
        self.assertFalse(spec.is_canonical)

    def test_sequence_longer_than_the_context_window_is_rejected(self):
        with self.assertRaises(WorkloadSpecError) as ctx:
            CANONICAL.replace(seq_len=QWEN3_0_6B["max_position_embeddings"] + 1)
        self.assertIn("max_position_embeddings", str(ctx.exception))

    def test_degenerate_sizes_are_rejected(self):
        for override in ({"batch_size": 0}, {"seq_len": 0}, {"dtype": "int8"}):
            with self.subTest(**override), self.assertRaises(WorkloadSpecError):
                CANONICAL.replace(**override)

    def test_grouped_query_heads_must_divide(self):
        with self.assertRaises(WorkloadSpecError) as ctx:
            CANONICAL.replace(arch={"num_key_value_heads": 7})
        self.assertIn("divisible", str(ctx.exception))


class TestWorkloadIdentity(unittest.TestCase):
    """A run's identity is a function of what it executes, and only that."""

    #: Pinned. A failure here means the canonical workload changed; that is a
    #: decision to make deliberately, not a test to update reflexively.
    CANONICAL_CONFIG_HASH = "6e254de2e1abf1a0"
    CANONICAL_WORKLOAD_HASH = "6e7919ad5cf60ba7"

    def test_hashes_are_pinned(self):
        self.assertEqual(CANONICAL.config_hash, self.CANONICAL_CONFIG_HASH)
        self.assertEqual(CANONICAL.workload_hash, self.CANONICAL_WORKLOAD_HASH)

    def test_id_is_stable_across_constructions(self):
        self.assertEqual(WorkloadSpec().workload_id, CANONICAL.workload_id)
        self.assertEqual(WorkloadSpec(), CANONICAL)

    def test_id_is_readable_and_carries_the_settings(self):
        self.assertEqual(
            CANONICAL.workload_id,
            f"qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa."
            f"{self.CANONICAL_WORKLOAD_HASH[:8]}",
        )

    def test_arch_key_order_does_not_change_identity(self):
        shuffled = dict(reversed(list(QWEN3_0_6B.items())))
        self.assertEqual(
            WorkloadSpec(arch_items=tuple(sorted(shuffled.items()))).config_hash,
            CANONICAL.config_hash,
        )

    def test_every_run_setting_changes_the_workload_hash(self):
        for override in (
            {"batch_size": 4},
            {"seq_len": 1024},
            {"dtype": "float32"},
            {"device": "cpu"},
            {"attn_implementation": "eager"},
            {"seed": 1},
        ):
            with self.subTest(**override):
                other = CANONICAL.replace(**override)
                self.assertNotEqual(other.workload_hash, CANONICAL.workload_hash)
                self.assertFalse(other.is_canonical)

    def test_architecture_changes_the_config_hash(self):
        other = CANONICAL.replace(arch={"num_hidden_layers": 2})
        self.assertNotEqual(other.config_hash, CANONICAL.config_hash)

    def test_run_settings_do_not_change_the_config_hash(self):
        """The architecture hash identifies the model, not the execution."""
        self.assertEqual(CANONICAL.replace(batch_size=8).config_hash, CANONICAL.config_hash)

    def test_dict_round_trip_preserves_identity(self):
        restored = WorkloadSpec.from_dict(json.loads(json.dumps(CANONICAL.to_dict())))
        self.assertEqual(restored.workload_hash, CANONICAL.workload_hash)
        self.assertEqual(restored, CANONICAL)


class TestReportSerialization(unittest.TestCase):
    def _report(self) -> SmokeReport:
        return SmokeReport(
            workload={"workload_id": CANONICAL.workload_id, "canonical": True},
            environment={"torch": torch.__version__, "transformers": "5.16.1"},
            effective={"attn_implementation": "sdpa"},
            result={
                "loss": 11.93,
                "loss_is_finite": True,
                "trainable_params": 310,
                "params_with_grad": 310,
                "missing_grad_params": [],
                "grads_all_finite": True,
            },
            diagnostics={"peak_allocated_bytes": 2**30},
        )

    def test_json_round_trip(self):
        report = self._report()
        restored = SmokeReport.from_dict(json.loads(report.to_json()))
        self.assertEqual(restored.to_dict(), report.to_dict())
        self.assertEqual(restored.schema_version, SCHEMA_VERSION)

    def test_required_keys_are_present(self):
        payload = self._report().to_dict()
        for key in (
            "schema_version",
            "status",
            "failure",
            "workload",
            "environment",
            "effective",
            "result",
            "diagnostics",
        ):
            self.assertIn(key, payload)

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "smoke.json"
            self._report().write(path)
            self.assertEqual(SmokeReport.read(path).to_dict(), self._report().to_dict())

    def test_a_failure_still_serializes(self):
        report = SmokeReport(
            workload={"workload_id": "x", "canonical": False},
            environment={},
            status="failed",
            failure="RuntimeError: no CUDA device",
        )
        self.assertFalse(report.ok)
        payload = json.loads(report.to_json())
        self.assertEqual(payload["status"], "failed")
        self.assertIn("no CUDA device", payload["failure"])
        self.assertIn("NON-CANONICAL", report.summary())

    def test_summary_flags_a_diagnostic_memory_number(self):
        self.assertIn("diagnostic", self._report().summary())


class TestOptionalDependency(unittest.TestCase):
    """Transformers is optional: the package imports without it, and the error
    when it is finally needed says what to install."""

    SCRIPT = """
import sys
sys.modules["transformers"] = None          # any import of it now raises
import evograd.bench.workloads.qwen3 as pkg
from evograd.bench.workloads.qwen3.levels.level4.model import MissingDependencyError, require_transformers
assert pkg.CANONICAL.workload_id
try:
    require_transformers()
except MissingDependencyError as exc:
    print("MESSAGE:" + str(exc).replace(chr(10), " | "))
else:
    raise SystemExit("require_transformers did not raise")
"""

    def _run(self) -> str:
        env = {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
        }
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_package_imports_without_transformers_and_the_error_is_actionable(self):
        out = self._run()
        self.assertIn("MESSAGE:", out)
        message = out.split("MESSAGE:", 1)[1]
        self.assertIn("pip install", message)
        self.assertIn("transformers>=4.51", message)

    def test_importing_the_package_does_not_import_transformers(self):
        """Import-time cost and import-time failure both belong to the caller."""
        script = (
            "import sys; import evograd.bench.workloads.qwen3;"
            " print('transformers' in sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestTinyQwen3Step(unittest.TestCase):
    """One real forward/loss/backward, at a size that fits a CPU test.

    Same code path as the canonical run -- ``build_model``, ``make_inputs``,
    ``training_step`` -- with the architecture shrunk. The full 0.6B model is
    never instantiated here.
    """

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.qwen3.levels.level4.smoke import run_smoke

        cls.spec = tiny_spec()
        cls.report = run_smoke(cls.spec)

    def test_the_step_succeeds(self):
        self.assertTrue(self.report.ok, self.report.failure)

    def test_loss_is_finite_and_near_uniform(self):
        """A freshly initialised model predicts uniformly, so the loss should sit
        near ln(vocab). Far from it means the labels or the shift are wrong."""
        import math

        self.assertTrue(self.report.result["loss_is_finite"])
        self.assertAlmostEqual(
            self.report.result["loss"], math.log(self.spec.arch["vocab_size"]), delta=0.5
        )

    def test_every_trainable_parameter_received_a_finite_gradient(self):
        result = self.report.result
        self.assertGreater(result["trainable_params"], 0)
        self.assertEqual(result["params_with_grad"], result["trainable_params"])
        self.assertEqual(result["missing_grad_params"], [])
        self.assertEqual(result["non_finite_grad_params"], [])
        self.assertTrue(result["grads_all_finite"])

    def test_the_report_is_marked_non_canonical(self):
        self.assertFalse(self.report.workload["canonical"])
        self.assertNotEqual(
            self.report.workload["workload_id"], self.report.workload["canonical_workload_id"]
        )

    def test_effective_settings_match_the_request(self):
        effective = self.report.effective
        self.assertEqual(effective["attn_implementation"], "sdpa")
        self.assertEqual(effective["attn_implementation_per_module"], ["sdpa"])
        self.assertFalse(effective["use_cache"])
        self.assertFalse(effective["gradient_checkpointing"])
        self.assertTrue(effective["training_mode"])
        self.assertEqual(effective["param_dtypes"], ["torch.float32"])
        self.assertEqual(effective["param_devices"], ["cpu"])

    def test_no_kv_cache_is_returned(self):
        self.assertFalse(self.report.effective["returned_past_key_values"])

    def test_labels_are_a_clone_of_the_inputs(self):
        self.assertTrue(self.report.effective["labels_match_input_ids"])

    def test_inputs_are_deterministic(self):
        from evograd.bench.workloads.qwen3.levels.level4.model import make_inputs

        first, _ = make_inputs(self.spec)
        second, _ = make_inputs(self.spec)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(first.sum()), self.report.effective["input_ids_checksum"])
        self.assertFalse(
            torch.equal(first, make_inputs(self.spec.replace(seed=1))[0]),
            "a different seed must give a different token stream",
        )

    def test_the_whole_step_is_reproducible(self):
        from evograd.bench.workloads.qwen3.levels.level4.smoke import run_smoke

        again = run_smoke(self.spec)
        self.assertEqual(again.result["loss"], self.report.result["loss"])


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestSettingsAreEnforcedOnTheBuiltModel(unittest.TestCase):
    """The report reads the model, not the request. These prove it can tell them
    apart -- otherwise every verification field would be a tautology."""

    def test_an_alternative_backend_is_reported_as_itself(self):
        from evograd.bench.workloads.qwen3.levels.level4.model import build_model, effective_settings

        spec = tiny_spec(attn_implementation="eager")
        model = build_model(spec)
        self.assertEqual(effective_settings(model, spec)["attn_implementation"], "eager")

    def test_a_mismatch_between_request_and_reality_is_detected(self):
        from evograd.bench.workloads.qwen3.levels.level4.model import (
            build_model,
            check_effective_settings,
            effective_settings,
        )

        spec = tiny_spec()
        model = build_model(spec)
        effective = effective_settings(model, spec)
        self.assertEqual(check_effective_settings(effective, spec), [])

        # Enabling checkpointing behind the spec's back must be caught.
        model.gradient_checkpointing_enable()
        problems = check_effective_settings(effective_settings(model, spec), spec)
        self.assertTrue(any("checkpointing" in p for p in problems), problems)

    def test_the_model_is_built_in_train_mode_with_the_cache_off(self):
        from evograd.bench.workloads.qwen3.levels.level4.model import build_model

        model = build_model(tiny_spec())
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.is_gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
