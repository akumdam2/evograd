"""Bridges to existing tier-3 code, run on tiny CPU configurations.

* the removed block scope: ``LegacyBlockScope`` (a deprecated name) refuses a
  legacy block adapter without importing removed block modules;
* ``WorkloadModelScope`` over the Qwen3-0.6B and Llama-3.2-1B workloads,
  shrunk with ``arch_overrides`` (random initialization, no download);
* Gate 3 over the Llama workload with its own ``SyntheticBatches``.

Real-size models, bf16 and GPUs are not exercised here.
"""

from __future__ import annotations

import logging
import unittest
import warnings

import torch
from torch import nn

from evograd.verification import gate2, gate3
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.common.reports import FAILED, PASSED
from evograd.verification.gate2.adapters.block import LegacyBlockScope
from evograd.verification.gate2.adapters.model import WorkloadModelScope
from evograd.verification.gate2.calibration import calibrate

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
FAST = "aot_eager"

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
needs_transformers = unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")

ARCH = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "head_dim": 16, "vocab_size": 128,
        "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1}


def samples(prefix, seeds):
    return [Sample(f"{prefix}/seed{s}", s) for s in seeds]


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def bound_eager(op_name, scale=1.0):
    """The declared eager pair through bind and the rank adapter -- the route a
    legacy candidate takes -- optionally with its output scaled (a fault)."""
    from evograd.benchmark import get_task
    from evograd.evaluation.tier3.patch import eager_pair_for, kernel_from_pair

    op = get_task(op_name)
    kernel = kernel_from_pair(op, eager_pair_for(op))
    if scale == 1.0:
        return kernel
    return lambda *args: kernel(*args) * scale


class TestNoBlockScope(unittest.TestCase):
    """Gate 2 has no block scope: the tier-3 block bridge is gone and the generic
    module adapter is model-scope only."""

    def test_legacy_block_bridge_refuses_to_build(self):
        with self.assertRaisesRegex(ValueError, "no block scope"):
            LegacyBlockScope(lambda s: None)

    def test_module_scope_refuses_block(self):
        from evograd.verification.gate2.adapters.module import ModuleScope
        from tests.verification import _fixtures as F

        with self.assertRaisesRegex(ValueError, "no 'block' scope"):
            ModuleScope(scope="block", name="b", factory=lambda sites: F.ToyBlock(sites),
                        native_sites=dict(F.NATIVE_SITES), make_invocation=F.block_invocation,
                        select_outputs=F.block_outputs, output_names=("y", "aux", "tag"))

    def test_a_block_scoped_adapter_cannot_calibrate_or_be_judged(self):
        from tests.verification import _fixtures as F

        scope = F.toy_module_scope()
        scope.scope = "block"                       # forced past the constructor
        cal = calibrate(scope, patch_set=("norm",), samples=samples("b", [1]), compile_backend=FAST)
        self.assertIsNone(cal.policy)
        self.assertEqual([f.code for f in cal.findings], ["unknown_scope"])
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}), policy=None,
                                          samples=samples("b", [1]))
        self.assertIn("unknown_scope", codes(report))
        self.assertNotEqual(report.status, PASSED)


@needs_transformers
class TestWorkloadModelBridge(unittest.TestCase):
    def check(self, workload, site, op_name, prefix):
        scope = WorkloadModelScope(workload)
        fixed = samples(f"{prefix}/fixed", [1, 2])
        cal = calibrate(scope, patch_set=(site,), samples=fixed, compile_backend=FAST)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        good = gate2.verify_integration(scope, candidate("bound", {site: bound_eager(op_name)}),
                                        policy=cal.policy, samples=fixed)
        self.assertEqual(good.status, PASSED, codes(good))
        layers = ARCH["num_hidden_layers"]
        self.assertEqual(good.cases[0].evidence["candidate"]["site_calls"], {site: layers})
        bad = gate2.verify_integration(scope, candidate("off", {site: bound_eager(op_name, 1.01)}),
                                       policy=cal.policy, samples=fixed)
        self.assertEqual(bad.status, FAILED)
        # A 1% fault inside a random tiny model barely moves the scalar loss;
        # the logits and the parameter gradients carry it.
        failed = {r.name for c in bad.cases for r in c.results if r.status == FAILED}
        self.assertIn("out:logits", failed)
        self.assertTrue(any(name.startswith("grad:") for name in failed))

    def test_llama(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

        workload = Llama3Workload.from_config({
            "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
            "arch_overrides": {**ARCH, "num_key_value_heads": 1}})
        self.check(workload, "swiglu_mlp", "llama3_swiglu_mlp", "llama")

    def test_llama_live_boundaries(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

        workload = Llama3Workload.from_config({
            "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
            "arch_overrides": {**ARCH, "num_key_value_heads": 1}})
        scope = WorkloadModelScope(workload)
        fixed = samples("lb/fixed", [1, 2])
        cal = calibrate(scope, patch_set=("swiglu_mlp",), samples=fixed,
                        compile_backend=FAST, boundaries=True)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        names = {n for n in cal.policy.thresholds_for("lb/fixed/seed1")
                 if n.startswith("boundary:swiglu_mlp#1/")}
        self.assertEqual(len(names), 5)          # the output and four argument gradients
        good = gate2.verify_integration(scope, candidate("bound", {"swiglu_mlp": bound_eager(
            "llama3_swiglu_mlp")}), policy=cal.policy, samples=fixed)
        self.assertEqual(good.status, PASSED, codes(good))
        bad = gate2.verify_integration(scope, candidate("off", {"swiglu_mlp": bound_eager(
            "llama3_swiglu_mlp", 1.01)}), policy=cal.policy, samples=fixed)
        failed = {r.name for c in bad.cases for r in c.results if r.status == FAILED}
        self.assertIn("boundary:swiglu_mlp#0/out:out", failed)

    def test_qwen3(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

        workload = Qwen3Workload(device="cpu", dtype="float32", batch_size=1, seq_len=16,
                                 arch_overrides={**ARCH, "num_key_value_heads": 2})
        self.check(workload, "swiglu_mlp", "qwen3_swiglu_mlp", "qwen3")


@needs_transformers
class TestWorkloadTrainingBridge(unittest.TestCase):
    def test_llama_continuous_training_measurement(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.training import SyntheticBatches
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

        workload = Llama3Workload.from_config({
            "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
            "arch_overrides": {**ARCH, "num_key_value_heads": 1}})

        def adapter():
            return gate3.workload_training(
                workload, train_data=SyntheticBatches(workload, offset=1000, count=4, role="train"),
                validation_data=SyntheticBatches(workload, offset=900_000, count=2,
                                                 role="validation"),
                validation_count=2, train_count=4, optimizer=gate3.adamw(lr=1e-3))

        plan = gate3.TrainingPlan(steps=4, window=2, checkpoints=(0, 4))
        native = gate3.measure_training(adapter(), Deployment.native(), plan, record_state=True)
        cand = gate3.measure_training(adapter(), candidate("bound", {
            "swiglu_mlp": bound_eager("llama3_swiglu_mlp")}), plan)
        self.assertTrue(native.completed, [f.to_dict() for f in native.findings])
        self.assertTrue(cand.completed, [f.to_dict() for f in cand.findings])
        self.assertEqual(native.state["4"]["optimizer"]["step_counts"], [4])
        self.assertEqual(cand.validation["0"], native.validation["0"])
        self.assertEqual(cand.site_calls["total"], {"swiglu_mlp": 2 * plan.steps})
        comparison = gate3.compare_curves(cand, native).to_dict()
        self.assertEqual(comparison["status"], "incomplete")
        self.assertIsNotNone(comparison["distances"])


if __name__ == "__main__":
    unittest.main()
