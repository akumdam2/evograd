"""Every LM harvest observes its own integration's real run.

``tests/verification/test_integration_qwen3.py`` runs the whole chain on a tiny
Qwen3; here each of the four harvests is checked to hand ``base.real_run`` its
own integration (the one whose snapshot a capture checks) and to observe what
comes back under the verification workload's spec, and the chain (harvest,
snapshot, the capture's run check) runs on tiny CPU Llama, OLMoE and Qwen3.5
workloads in place of the pinned checkpoints and WikiText.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - the CI environment has it
    HAVE_TRANSFORMERS = False

PACKAGES = ("qwen3_0_6b", "llama3_2_1b", "olmoe_1b_7b", "qwen3_5_4b")


class EveryHarvestObservesItsIntegration(unittest.TestCase):
    def test_the_harvest_passes_its_integrations_run_to_the_observer(self):
        for package in PACKAGES:
            with self.subTest(package):
                harvest = importlib.import_module(
                    f"evograd.benchmark.topdown.{package}.harvest.harvest")
                declaration = importlib.import_module(
                    f"evograd.benchmark.topdown.{package}.declaration")
                integration = importlib.import_module(f"evograd.verification.integrations.{package}")
                workload = SimpleNamespace(spec="the verification spec")
                ran = []

                def real_run(model, *, device):
                    ran.append((model, device))
                    return workload, "net", "batch", {"order": 101}

                with mock.patch("evograd.verification.integrations.base.real_run", real_run), \
                        mock.patch("evograd.benchmark.topdown.common.harvest.run_harvest",
                                   return_value={"run": "observed"}) as observe:
                    self.assertEqual(harvest.harvest_real_run(device="cuda"), {"run": "observed"})
                self.assertEqual(ran, [(integration.MODEL, "cuda")])
                self.assertEqual(integration.MODEL.name, declaration.WORKLOAD.name)
                observe.assert_called_once_with(declaration.WORKLOAD, "the verification spec",
                                                model="net", batch="batch",
                                                identity={"order": 101})



def tiny_workloads():
    """``package -> (tiny CPU workload, extract keywords)`` for the three LMs."""
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload
    from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload
    from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload
    from tests.llama3.test_level3_workload import tiny_spec as llama_spec
    from tests.olmoe_1b_7b import _tiny as olmoe
    from tests.qwen3_5_4b import test_harvest as qwen3_5

    llama, gdn = llama_spec(), qwen3_5.tiny_spec()
    return {
        "llama3_2_1b": (Llama3Workload(device="cpu", dtype=llama.dtype, batch_size=llama.batch_size,
                                       seq_len=llama.seq_len, arch_overrides=dict(llama.arch)),
                        {"layer_index": 1}),
        "olmoe_1b_7b": (OlmoeWorkload(device="cpu", batch_size=1, seq_len=olmoe.SEQ,
                                      arch_overrides=dict(olmoe.TINY_ARCH)),
                        {"representative_layers": olmoe.LAYERS}),
        "qwen3_5_4b": (Qwen3_5Workload(device="cpu", dtype=gdn.dtype, batch_size=gdn.batch_size,
                                       seq_len=gdn.seq_len, arch_overrides=dict(gdn.arch)),
                       {"representative_layers": qwen3_5.LAYERS}),
    }


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyRealRunChain(unittest.TestCase):
    """harvest_real_run -> snapshot.extract -> the capture's run check, on CPU."""

    def test_a_capture_of_the_harvested_run_is_accepted_and_of_another_refused(self):
        import torch

        from evograd.verification.common.tensors import tensor_digest
        from evograd.verification.integrations import base
        from tests.verification._toy_integration import configured

        for package, (workload, extract) in tiny_workloads().items():
            with self.subTest(package):
                vocab, batch, seq = (workload.spec.arch["vocab_size"], workload.batch_size,
                                     workload.seq_len)

                def train_batches(_workload, orders, offset=0):
                    ids = {o: torch.randint(0, vocab, (batch, seq),
                                            generator=torch.Generator().manual_seed(o + offset))
                           for o in orders}
                    return ({o: (i, i.clone()) for o, i in ids.items()},
                            {"dataset": "tiny", "batches": {
                                str(o): {"order": o, "input_ids_sha256": tensor_digest(i)}
                                for o, i in ids.items()}})

                integration = importlib.import_module(f"evograd.verification.integrations.{package}")
                harvest = importlib.import_module(f"evograd.benchmark.topdown.{package}.harvest.harvest")
                snapshot = importlib.import_module(f"evograd.benchmark.topdown.{package}.harvest.snapshot")
                model = dataclasses.replace(integration.MODEL,
                                            workload_config={"model": package, "device": "cpu"},
                                            load_workload=lambda w=workload: w,
                                            train_batches=train_batches)
                with configured(), mock.patch.object(integration, "MODEL", model):
                    manifest = json.loads(json.dumps(harvest.harvest_real_run(device="cpu"),
                                                     default=str))
                    state = base.objective_of(model).parameter_state(base.build_model(workload))
                derived = snapshot.extract(manifest, **extract)
                self.assertEqual(derived["schema_version"], snapshot.SCHEMA_VERSION)
                self.assertEqual(derived["run"], manifest["run"])
                with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: derived):
                    base.require_harvested_run(
                        model, base.run_identity(model, state, train_batches(workload, base.ORDERS)[1]))
                    other = base.run_identity(model, state,
                                              train_batches(workload, base.ORDERS, offset=1)[1])
                    with self.assertRaisesRegex(RuntimeError, "another run"):
                        base.require_harvested_run(model, other)


if __name__ == "__main__":
    unittest.main()
