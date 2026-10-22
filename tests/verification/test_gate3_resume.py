"""A checkpointed Gate 3 run equals the run that never stopped.

Tiny random Llama on CPU, AdamW, six steps: suspended after a step and resumed
in a fresh build, the native run, a bound candidate and an ``aot_eager``
compile control each reproduce every objective, the run id and the run digest
of an unbroken run. A checkpoint of another run is set aside, never resumed.
Warmup and gradient clipping resume the same way, and leave a constant-rate,
unclipped run exactly as it was.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from evograd.verification import gate3
from evograd.verification.common.execution import Deployment, candidate
from evograd.verification.gate3.checkpoint import RunCheckpoint
from evograd.verification.gate3.stepwise import run_digest

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

ARCH = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "head_dim": 16, "vocab_size": 128,
        "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1,
        "num_key_value_heads": 1}
STEPS = 6
PLAN = gate3.TrainingPlan(steps=STEPS, window=1, checkpoints=(), validation=False)


@dataclass(frozen=True)
class StopAfter(RunCheckpoint):
    """Expires once ``after`` steps have been checked (a deterministic deadline)."""

    after: int = 0
    seen: list = field(default_factory=list)

    def expired(self) -> bool:
        self.seen.append(1)
        return len(self.seen) >= self.after


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class ResumedRunEqualsUnbroken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.training import SyntheticBatches
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

        cls.workload = Llama3Workload.from_config({
            "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
            "arch_overrides": ARCH})
        cls.data = SyntheticBatches(cls.workload, offset=1000, count=STEPS, role="train")

    def adapter(self, **optimizer):
        return gate3.workload_training(self.workload, train_data=self.data, train_count=STEPS,
                                       optimizer=gate3.adamw(lr=1e-3, **optimizer))

    def deployments(self):
        from tests.verification.test_bridges import bound_eager

        from evograd.verification.gate2.adapters.model import WorkloadModelScope
        from evograd.verification.gate2.calibration import compile_control

        compiled, _ = compile_control(WorkloadModelScope(self.workload), ("swiglu_mlp",),
                                      backend="aot_eager")
        return {"native": Deployment.native(),
                "bound": candidate("bound", {"swiglu_mlp": bound_eager("llama3_swiglu_mlp")}),
                "compile": compiled}

    def run_split(self, deployment, after, path, **optimizer):
        first = gate3.measure_training(self.adapter(**optimizer), deployment, PLAN, seed=0,
                                       checkpoint=StopAfter(path, every=0, after=after))
        self.assertTrue(first.suspended, [f.to_dict() for f in first.findings])
        self.assertFalse(first.completed)
        self.assertEqual(len(first.train_steps), after)
        second = gate3.measure_training(self.adapter(**optimizer), deployment, PLAN, seed=0,
                                        checkpoint=RunCheckpoint(path, every=2))
        self.assertTrue(second.completed, [f.to_dict() for f in second.findings])
        self.assertFalse(path.exists(), "a completed run removes its checkpoint")
        return first, second

    def test_resume_reproduces_every_record(self):
        for name, deployment in self.deployments().items():
            with self.subTest(deployment=name), tempfile.TemporaryDirectory() as tmp:
                unbroken = gate3.measure_training(self.adapter(), deployment, PLAN, seed=0)
                self.assertTrue(unbroken.completed, [f.to_dict() for f in unbroken.findings])
                first, resumed = self.run_split(deployment, 3, Path(tmp) / "run.ckpt")
                self.assertEqual([s["objective"] for s in resumed.train_steps],
                                 [s["objective"] for s in unbroken.train_steps])
                self.assertEqual(resumed.train_steps, unbroken.train_steps)
                self.assertEqual(resumed.identity, unbroken.identity)
                self.assertEqual(resumed.run_id, first.run_id)
                self.assertEqual(resumed.site_calls, unbroken.site_calls)
                # Same run id, identity and records: the digest of the run that never stopped.
                self.assertEqual(run_digest(resumed.to_dict()),
                                 run_digest({**unbroken.to_dict(), "run_id": first.run_id}))
                self.assertEqual([seg.get("resumed_at_step") for seg in resumed.segments],
                                 [None, 4])

    def test_checkpoint_of_another_run_is_set_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.ckpt"
            gate3.measure_training(self.adapter(), Deployment.native(), PLAN, seed=0,
                                   checkpoint=StopAfter(path, every=0, after=2))
            other = gate3.measure_training(self.adapter(), Deployment.native(), PLAN, seed=1,
                                           checkpoint=RunCheckpoint(path, every=0))
            self.assertTrue(other.completed)
            self.assertIn("discarded_checkpoint", other.segments[0])
            self.assertEqual(other.segments[1]["started_at_step"], 1)

    def test_unbroken_run_without_checkpoint_serializes_as_before(self):
        run = gate3.measure_training(self.adapter(), Deployment.native(), PLAN, seed=0)
        self.assertNotIn("segments", run.to_dict())
        self.assertNotIn("suspended", run.to_dict())
        # Constant rate, no clipping: neither the configuration nor a step record mentions them.
        self.assertEqual(run.identity["optimizer"]["schedule"], "constant")
        self.assertNotIn("warmup_steps", run.identity["optimizer"])
        self.assertNotIn("clip_grad_norm", run.identity["optimizer"])
        self.assertFalse({"lr", "grad_norm"} & set(run.train_steps[0]))

    def test_warmup_and_clipping_are_recorded_and_resume(self):
        settings = {"warmup_steps": 4, "clip_grad_norm": 1e-3}
        with tempfile.TemporaryDirectory() as tmp:
            unbroken = gate3.measure_training(self.adapter(**settings), Deployment.native(), PLAN,
                                              seed=0)
            self.assertTrue(unbroken.completed, [f.to_dict() for f in unbroken.findings])
            self.assertEqual(unbroken.identity["optimizer"]["schedule"], "linear_warmup")
            self.assertEqual(unbroken.identity["optimizer"]["warmup_steps"], 4)
            self.assertEqual(unbroken.identity["optimizer"]["clip_grad_norm"], 1e-3)
            self.assertEqual([s["lr"] for s in unbroken.train_steps],
                             [1e-3 * min(1.0, t / 4) for t in range(1, STEPS + 1)])
            self.assertTrue(all(s["grad_norm"] > 1e-3 for s in unbroken.train_steps))
            first, resumed = self.run_split(Deployment.native(), 3, Path(tmp) / "run.ckpt",
                                            **settings)
            self.assertEqual(resumed.train_steps, unbroken.train_steps)
            self.assertEqual(resumed.identity, unbroken.identity)
        plain = gate3.measure_training(self.adapter(), Deployment.native(), PLAN, seed=0)
        # Same first loss (read before any update), then the schedule and the clip change the run.
        self.assertEqual(plain.train_steps[0]["objective"], unbroken.train_steps[0]["objective"])
        self.assertNotEqual(plain.train_steps[1]["objective"], unbroken.train_steps[1]["objective"])
        self.assertNotEqual(plain.identity, unbroken.identity)


if __name__ == "__main__":
    unittest.main()
