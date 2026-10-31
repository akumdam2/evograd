"""Optional real Transformers checkpoint roundtrip; no network or GPU required."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evograd.evaluation.model_cache import PristineModelCache


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers is optional")
class CheckpointModelCacheTests(unittest.TestCase):
    def test_qwen_checkpoint_load_once_matches_outputs_and_gradients(self):
        from transformers import Qwen3ForCausalLM
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3 import model as builder
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import CANONICAL

        arch = {**CANONICAL.arch, "num_hidden_layers": 1, "hidden_size": 16,
                "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "vocab_size": 32, "bos_token_id": 1, "eos_token_id": 2}
        spec = CANONICAL.replace(arch=arch, device="cpu", batch_size=1, seq_len=4,
                                 weights="pretrained:local/test@fixed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            source = Qwen3ForCausalLM._from_config(builder.build_config(spec), dtype=torch.bfloat16)
            source.save_pretrained(checkpoint)
            with patch.object(builder, "pretrained_snapshot", return_value=str(checkpoint)), \
                    patch.object(builder, "_load_pretrained", wraps=builder._load_pretrained) as loader:
                direct = builder.build_model(spec)
                cache = PristineModelCache(root / "cache")
                first = cache.build(builder.build_model, spec)
                second = PristineModelCache(root / "cache").build(builder.build_model, spec)
                self.assertEqual(loader.call_count, 2)  # direct reference + one cached checkpoint read
                ids = torch.tensor([[1, 2, 3, 4]])
                reference = direct(ids, use_cache=False).logits
                reference.float().sum().backward()
                for model in (first, second):
                    output = model(ids, use_cache=False).logits
                    self.assertTrue(torch.equal(output, reference))
                    output.float().sum().backward()
                    self.assertTrue(torch.equal(model.lm_head.weight.grad, direct.lm_head.weight.grad))
                    self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)
                self.assertTrue(torch.equal(first.lm_head.weight.grad, second.lm_head.weight.grad))

    def test_all_agent_context_loads_checkpoint_once_and_restores_factory(self):
        from transformers import Qwen3ForCausalLM
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3 import model as builder
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import CANONICAL
        from evograd.evaluation import case_group, model_group
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload

        arch = {**CANONICAL.arch, "num_hidden_layers": 1, "hidden_size": 16,
                "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "vocab_size": 32, "bos_token_id": 1, "eos_token_id": 2}
        spec = CANONICAL.replace(arch=arch, device="cpu", batch_size=1, seq_len=4,
                                 weights="pretrained:local/test@fixed")
        factory = workload.build_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            source = Qwen3ForCausalLM._from_config(builder.build_config(spec), dtype=torch.bfloat16)
            source.save_pretrained(checkpoint)
            outputs = []

            def evaluate(_args):
                model = workload.build_model(spec)
                outputs.append(model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits.detach())
                with torch.no_grad():
                    model.lm_head.weight.add_(1)
                return 0

            def evaluate_group(plan, **_kwargs):
                for job in plan["jobs"]:
                    evaluate(job)
                return 0

            plan = {"model": "qwen3_0_6b", "state": str(root / "state"), "jobs": [
                {"out": str(root / agent), "selection": {"agent": agent, "corpus": "manifest.json",
                 "candidates": {"task/case": {"path": f"/{agent}/submission.py"}}}}
                for agent in ("gepa", "codex")]}
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                    patch.object(builder, "pretrained_snapshot", return_value=str(checkpoint)), \
                    patch.object(builder, "_load_pretrained", wraps=builder._load_pretrained) as loader, \
                    patch("evograd.verification.integrations.load",
                          return_value=SimpleNamespace(load_workload=lambda: SimpleNamespace(spec=spec))), \
                    patch.object(case_group, "evaluate_group", side_effect=evaluate_group):
                self.assertEqual(model_group.run(plan, deadline_min=45), 0)
                self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertIs(workload.build_model, factory)

    def test_qwen35_cache_replays_loader_rng_and_preserves_training(self):
        """Real hybrid checkpoint: cold, warm and disk hits under different seeds."""
        from transformers import Qwen3_5ForCausalLM
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import model as builder
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
        from tests.qwen3_5_4b._tiny import TINY_ARCH

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=4,
                                 weights="pretrained:local/test@fixed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            source = Qwen3_5ForCausalLM._from_config(builder.build_config(spec), dtype=torch.bfloat16)
            source.save_pretrained(checkpoint)
            cache = PristineModelCache(root / "cache", model_name="qwen3_5_4b")
            with patch.object(builder, "pretrained_snapshot", return_value=str(checkpoint)), \
                    patch.object(builder, "_load_pretrained", wraps=builder._load_pretrained) as loader:
                for seed in (101, 202, 0):
                    with self.subTest(seed=seed), torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        before = torch.get_rng_state().clone()
                        direct = builder.build_model(spec)
                        after = torch.get_rng_state().clone()
                        self.assertFalse(torch.equal(before, after))
                        torch.set_rng_state(before)
                        memory = cache.build(builder.build_model, spec)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        torch.set_rng_state(before)
                        disk = PristineModelCache(root / "cache", model_name="qwen3_5_4b")
                        snapshot = disk.build(builder.build_model, spec)
                        self.assertEqual(disk.loads, 0)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        models = (direct, memory, snapshot)
                        # Include non-persistent buffers, not just state_dict weights.
                        for get in ("named_parameters", "named_buffers"):
                            original = dict(getattr(direct, get)())
                            for model in models[1:]:
                                actual = dict(getattr(model, get)())
                                self.assertEqual(original.keys(), actual.keys())
                                self.assertTrue(all(torch.equal(original[k], actual[k]) for k in original))
                        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in models]
                        for _ in range(2):
                            losses, grads = [], []
                            for model, optimizer in zip(models, optimizers):
                                optimizer.zero_grad(set_to_none=True)
                                ids = torch.tensor([[1, 2, 3, 4]])
                                output = model(ids, labels=ids, use_cache=False)
                                losses.append(output.loss.detach())
                                output.loss.backward()
                                grads.append({name: None if p.grad is None else p.grad.clone()
                                              for name, p in model.named_parameters()})
                                optimizer.step()
                            self.assertTrue(all(torch.equal(losses[0], loss) for loss in losses[1:]))
                            for model in models[1:]:
                                actual = dict(model.named_parameters())
                                self.assertTrue(all(torch.equal(value, actual[name])
                                                    for name, value in direct.named_parameters()))
                            for other in grads[1:]:
                                for name, value in grads[0].items():
                                    self.assertTrue(other[name] is None if value is None
                                                    else torch.equal(value, other[name]), name)
                        # Training and agent patches must not alter the pristine snapshot.
                        memory.agent_patch = "agent-a"
                        torch.set_rng_state(before)
                        fresh = cache.build(builder.build_model, spec)
                        self.assertFalse(hasattr(fresh, "agent_patch"))
                        self.assertTrue(all(p.grad is None for p in fresh.parameters()))
                        self.assertTrue(torch.equal(fresh.lm_head.weight, source.lm_head.weight))
                        self.assertIs(fresh.lm_head.weight, fresh.model.embed_tokens.weight)
                import subprocess
                import sys

                torch.manual_seed(303)
                builder.build_model(spec)
                expected_rng = torch.get_rng_state()
                torch.save(expected_rng, root / "expected_rng.pt")
                script = """
import sys, torch
from pathlib import Path
from unittest.mock import patch
from evograd.evaluation.model_cache import checkpoint_model_cache
from evograd.evaluation.tier3.workloads.qwen3_5_4b import workload
from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import model as builder
from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
from tests.qwen3_5_4b._tiny import TINY_ARCH
root = Path(sys.argv[1])
spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=4,
                         weights="pretrained:local/test@fixed")
with checkpoint_model_cache("qwen3_5_4b", root / "cache") as cache:
    with patch.object(builder, "_load_pretrained", side_effect=AssertionError("checkpoint reopened")):
        torch.manual_seed(303)
        model = workload.build_model(spec)
        assert torch.equal(torch.get_rng_state(), torch.load(root / "expected_rng.pt", weights_only=True))
        assert cache.loads == 0
        assert all(p.grad is None for p in model.parameters())
"""
                child = subprocess.run([sys.executable, "-c", script, str(root)],
                                       capture_output=True, text=True)
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertEqual(cache.loads, 1)
                self.assertEqual(loader.call_count, 5)  # four references plus one snapshot load

    def test_qwen35_partial_checkpoint_is_not_cached(self):
        from transformers import Qwen3_5ForCausalLM
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import model as builder
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
        from tests.qwen3_5_4b._tiny import TINY_ARCH

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=4,
                                 weights="pretrained:local/test@fixed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Qwen3_5ForCausalLM._from_config(builder.build_config(spec), dtype=torch.bfloat16)
            state = {name: value for name, value in source.state_dict().items() if not name.endswith("A_log")}
            source.save_pretrained(root / "checkpoint", state_dict=state)
            with patch.object(builder, "pretrained_snapshot", return_value=str(root / "checkpoint")):
                cache = PristineModelCache(root / "cache", model_name="qwen3_5_4b")
                with self.assertRaisesRegex(RuntimeError, "Incomplete Qwen3.5 pretrained checkpoint"):
                    cache.build(builder.build_model, spec)
                self.assertFalse(list((root / "cache").glob("*.pt")))

    def test_qwen35_priming_preserves_rng_and_unknown_rng_changes_are_rejected(self):
        from transformers import Qwen3_5ForCausalLM
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import model as builder
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
        from evograd.evaluation.model_cache import checkpoint_model_cache
        from evograd.evaluation.tier3.workloads.qwen3_5_4b import workload
        from tests.qwen3_5_4b._tiny import TINY_ARCH

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=4,
                                 weights="pretrained:local/test@fixed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Qwen3_5ForCausalLM._from_config(builder.build_config(spec), dtype=torch.bfloat16)
            source.save_pretrained(root / "checkpoint")
            with patch.object(builder, "pretrained_snapshot", return_value=str(root / "checkpoint")):
                before = torch.get_rng_state().clone()
                with checkpoint_model_cache("qwen3_5_4b", root / "cache") as cache:
                    cache.prime(spec)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    for seed in (101, 202):
                        torch.manual_seed(seed)
                        builder.build_model(spec)
                        expected = torch.get_rng_state().clone()
                        torch.manual_seed(seed)
                        workload.build_model(spec)
                        self.assertTrue(torch.equal(expected, torch.get_rng_state()))
                    self.assertEqual(cache.loads, 1)
                cache = PristineModelCache(root / "bad-cache", model_name="qwen3_5_4b")
                with self.assertRaisesRegex(RuntimeError, "do not match"):
                    cache.build(qwen35_extra_rng, spec)
                self.assertFalse(list((root / "bad-cache").glob("*.pt")))


def qwen35_extra_rng(spec):
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import model as builder

    model = builder.build_model(spec)
    torch.rand(1)
    return model
