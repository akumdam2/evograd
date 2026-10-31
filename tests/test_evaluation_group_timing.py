from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evograd.benchmark import get_task
from evograd.evaluation import group_timing
from evograd.evaluation.tier1 import fair


class GroupTimingTests(unittest.TestCase):
    def test_each_provider_measured_once_per_block_with_shared_pair_indices(self):
        op = get_task("swiglu")
        providers = {name: SimpleNamespace(name=name) for name in ("candidate:a", "candidate:b", "eager", "torch_compile")}
        counts = Counter()

        def measure(provider, own, _dout, **_kwargs):
            counts[provider.name] += 1
            return {"forward": [1.0], "backward": [2.0], "full": [3.0]}

        values = {name: torch.ones(2, 4) for name in ("a", "b", "dc")}
        with patch.object(fair, "L2Cache", return_value=SimpleNamespace(clear=lambda: None)), \
                patch.object(fair, "_warm_provider"), patch.object(fair, "_measure_provider_block", side_effect=measure):
            report = group_timing.measure_case(op, providers, values, blocks=3)
        self.assertEqual(counts, Counter({name: 3 for name in providers}))
        self.assertEqual(report["errors"], {})
        for name in providers:
            self.assertEqual([b["pair_index"] for b in report["raw_blocks"][name]], [0, 1, 2])

    def test_bad_candidate_does_not_remove_shared_baselines_or_other_candidates(self):
        op = get_task("swiglu")
        providers = {name: SimpleNamespace(name=name) for name in ("bad", "good", "eager")}

        def measure(provider, *_args, **_kwargs):
            if provider.name == "bad":
                raise RuntimeError("candidate failed")
            return {"forward": [1.0], "backward": [2.0], "full": [3.0]}

        with patch.object(fair, "L2Cache", return_value=SimpleNamespace(clear=lambda: None)), \
                patch.object(fair, "_warm_provider"), patch.object(fair, "_measure_provider_block", side_effect=measure):
            report = group_timing.measure_case(op, providers, {n: torch.ones(2, 4) for n in ("a", "b", "dc")})
        self.assertIn("bad", report["errors"])
        self.assertEqual(set(report["providers"]), {"good", "eager"})

    def test_all_agents_reference_identical_baseline_samples(self):
        op = replace(get_task("swiglu"), performance_baselines={})
        measured = []

        def call(_op, providers, values, **_kwargs):
            measured.append(tuple(providers))
            raw = {name: [{"pair_index": index, "samples_ms": {
                "forward": [float(i + 1)], "backward": [float(i + 1)], "full": [float(i + 1)]}}
                for index in range(3)] for i, name in enumerate(providers)}
            return {"errors": {}, "raw_blocks": raw, "providers": {
                name: {metric: {"median_ms": float(i + 1)} for metric in ("forward", "backward", "pair_full")}
                for i, name in enumerate(providers)}}

        candidate = lambda op, module: SimpleNamespace(name="candidate")
        with patch("evograd.evaluation.common.provider.candidate_autograd_provider", side_effect=candidate), \
                patch("evograd.evaluation.common.provider.renamed_provider", side_effect=lambda obj, name: SimpleNamespace(name=name)), \
                patch("evograd.evaluation.common.provider.pytorch_autograd_provider", return_value=SimpleNamespace(name="eager")), \
                patch("evograd.evaluation.common.provider.torch_compile_provider", return_value=SimpleNamespace(name="torch_compile")), \
                patch.object(group_timing, "measure_case", side_effect=call):
            report = group_timing.time_case(op, {"a": object(), "b": object()},
                                           [(None, lambda: {})], options={"warmup": 10, "reps": 50, "blocks": 3, "seed": 0})
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0].count("eager"), 1)
        self.assertEqual(report["rows"]["a"]["eager"]["baseline_ms"], report["rows"]["b"]["eager"]["baseline_ms"])
        self.assertEqual(report["rows"]["a"]["eager"]["ci95"]["low"], 3.0)
        self.assertEqual(report["rows"]["b"]["eager"]["ci95"]["low"], 1.5)
