"""The four-part protocol's metrics and policy, on CPU.

The KL is checked against a small float64 oracle, for direction, for token
normalisation, for masks and the causal shift, and for invariance to a
constant logit offset. The gradient metric is checked for the case a norm
comparison cannot see: equal norm, different direction. The policy is checked
for identity rejection, for the candidate having no route into a threshold, and
for stopping before timing when an enforced gate fails.
"""

from __future__ import annotations

import math
import unittest

import torch

from evograd.evaluation.tier3.workloads.qwen3_0_6b import prediction as P
from evograd.evaluation.tier3.workloads.qwen3_0_6b import protocol4 as P4
from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import (
    PatchSet, PolicyMismatch, global_grad_rel_l2,
)
from evograd.evaluation.tier3.workloads.qwen3_0_6b.training import (
    TrainingPlan, training_distances,
)

QKV = PatchSet(("qkv_norm_rope",), ("attention",), {"attention": 28, "qkv_norm_rope": 28})
RESIDUAL = PatchSet(("residual_rmsnorm",), (), {"residual_rmsnorm": 56})


def _oracle_kl(ref, prov, labels):
    """Unchunked float64 KL(P_ref || P_prov) over valid next-token positions."""
    mask = labels[:, 1:] != P.IGNORE_INDEX
    lp = torch.log_softmax(ref[:, :-1].double(), -1)
    lq = torch.log_softmax(prov[:, :-1].double(), -1)
    per = (lp.exp() * (lp - lq)).sum(-1)
    return float(per[mask].sum() / mask.sum())


class TestKL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L, self.V = 2, 11, 131
        self.ref = torch.randn(self.B, self.L, self.V, dtype=torch.float64) * 3
        self.prov = self.ref + 0.05 * torch.randn_like(self.ref)
        self.labels = torch.randint(0, self.V, (self.B, self.L))

    def test_chunked_float32_agrees_with_the_float64_oracle(self):
        for chunk in (1, 3, 7, 1000):
            got = P.mean_token_kl(self.ref, self.prov, self.labels, chunk=chunk)["kl_mean"]
            self.assertAlmostEqual(got, _oracle_kl(self.ref, self.prov, self.labels),
                                   delta=1e-5 * max(1.0, abs(got)) + 1e-8, msg=f"chunk {chunk}")

    def test_direction_is_kl_of_reference_against_provider(self):
        skewed = self.ref.clone()
        skewed[..., 0] += 4.0          # the provider over-predicts token 0
        forward = P.mean_token_kl(self.ref, skewed, self.labels)["kl_mean"]
        reverse = P.mean_token_kl(skewed, self.ref, self.labels)["kl_mean"]
        self.assertAlmostEqual(forward, _oracle_kl(self.ref, skewed, self.labels), delta=1e-6)
        self.assertNotAlmostEqual(forward, reverse, places=4)

    def test_identical_distributions_give_exactly_zero(self):
        self.assertEqual(P.mean_token_kl(self.ref, self.ref, self.labels)["kl_mean"], 0.0)
        bf = self.ref.to(torch.bfloat16)
        self.assertEqual(P.mean_token_kl(bf, bf.clone(), self.labels)["kl_mean"], 0.0)

    def test_invariant_to_a_constant_logit_offset_per_position(self):
        base = P.mean_token_kl(self.ref, self.prov, self.labels)["kl_mean"]
        offset = torch.randn(self.B, self.L, 1, dtype=torch.float64) * 10
        shifted = P.mean_token_kl(self.ref + offset, self.prov - offset, self.labels)["kl_mean"]
        self.assertAlmostEqual(base, shifted, delta=1e-6)

    def test_normalises_by_valid_tokens_not_by_positions_or_batches(self):
        labels = self.labels.clone()
        labels[0, 1:] = P.IGNORE_INDEX          # sequence 0 contributes nothing
        result = P.mean_token_kl(self.ref, self.prov, labels)
        self.assertEqual(result["valid_positions"], self.L - 1)   # only sequence 1
        only_seq1 = _oracle_kl(self.ref[1:], self.prov[1:], self.labels[1:])
        self.assertAlmostEqual(result["kl_mean"], only_seq1, delta=1e-6)

    def test_causal_shift_the_last_position_never_counts(self):
        mask = P.valid_positions(self.labels)
        self.assertEqual(tuple(mask.shape), (self.B, self.L - 1))
        labels = self.labels.clone()
        labels[:, -1] = P.IGNORE_INDEX            # ignoring the last *label* ...
        self.assertEqual(P.mean_token_kl(self.ref, self.prov, labels)["valid_positions"],
                         self.B * (self.L - 2))   # ... removes position L-2's target

    def test_an_empty_mask_is_rejected(self):
        with self.assertRaises(P.PredictionError):
            P.mean_token_kl(self.ref, self.prov, torch.full_like(self.labels, P.IGNORE_INDEX))

    def test_non_finite_inputs_are_rejected(self):
        bad = self.prov.clone()
        bad[0, 3, 5] = float("nan")
        with self.assertRaises(P.PredictionError):
            P.mean_token_kl(self.ref, bad, self.labels)

    def test_roundoff_negatives_are_flagged_substantial_ones_raise(self):
        # A real log_softmax cannot produce a substantially negative mean, so the
        # policy is tested on the finalisation step it lives in.
        tiny = -0.5 * P.ROUNDOFF_TOLERANCE
        mean, flagged = P.finalize_mean_kl(tiny * 10, 10)
        self.assertAlmostEqual(mean, tiny)
        self.assertTrue(flagged)                      # reported as measured, flagged
        with self.assertRaises(P.PredictionError):    # not clamped
            P.finalize_mean_kl(-1e-3 * 10, 10)
        with self.assertRaises(P.PredictionError):
            P.finalize_mean_kl(float("nan"), 10)
        self.assertEqual(P.finalize_mean_kl(0.0, 10), (0.0, False))

    def test_nll_matches_cross_entropy_sum_and_reports_tokens(self):
        labels = self.labels.clone()
        labels[1, 2:5] = P.IGNORE_INDEX
        result = P.mean_token_nll(self.ref, labels)
        ce = torch.nn.functional.cross_entropy(
            self.ref[:, :-1].reshape(-1, self.V), labels[:, 1:].reshape(-1),
            ignore_index=P.IGNORE_INDEX, reduction="sum")
        self.assertAlmostEqual(result["nll_sum"], float(ce), delta=1e-5)
        self.assertEqual(result["valid_tokens"], int((labels[:, 1:] != P.IGNORE_INDEX).sum()))


class TestGradientVectorError(unittest.TestCase):
    def test_equal_norm_different_direction_is_not_zero(self):
        # The case a comparison of gradient *norms* would call identical.
        g = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([0.0, 4.0])}
        rotated = {"a": torch.tensor([0.0, 3.0]), "b": torch.tensor([4.0, 0.0])}
        norm_ref = math.sqrt(9 + 16)
        norm_rot = math.sqrt(9 + 16)
        self.assertAlmostEqual(norm_ref, norm_rot)
        rel = global_grad_rel_l2(rotated, g)["rel_l2"]
        expected = math.sqrt(9 + 9 + 16 + 16) / norm_ref
        self.assertAlmostEqual(rel, expected, places=6)
        self.assertGreater(rel, 1.0)

    def test_matched_by_name_missing_and_extra_fail(self):
        ref = {"a": torch.ones(3), "b": torch.ones(3)}
        r = global_grad_rel_l2({"a": torch.ones(3), "z": torch.ones(3)}, ref)
        self.assertEqual(r["missing"], ["b"])
        self.assertEqual(r["extra"], ["z"])
        self.assertFalse(r["ok"])


class TestTrainingDistances(unittest.TestCase):
    def _run(self, windows, val):
        return {"train_window_nll": windows, "validation_nll": {str(k): v for k, v in val.items()}}

    def test_max_abs_delta_over_windows_and_checkpoints(self):
        ref = self._run([2.0, 1.9, 1.8], {0: 2.5, 100: 2.4})
        prov = self._run([2.0, 1.95, 1.79], {0: 2.5, 100: 2.43})
        d = training_distances(prov, ref)
        self.assertAlmostEqual(d["train_window_nll_max_abs_delta"], 0.05)
        self.assertAlmostEqual(d["val_nll_max_abs_delta"], 0.03)

    def test_mismatched_horizons_are_refused(self):
        with self.assertRaises(ValueError):
            training_distances(self._run([1.0], {0: 1.0}), self._run([1.0, 1.0], {0: 1.0}))
        with self.assertRaises(ValueError):
            training_distances(self._run([1.0], {0: 1.0}), self._run([1.0], {0: 1.0, 5: 1.0}))

    def test_the_plan_is_checked(self):
        with self.assertRaises(ValueError):
            TrainingPlan(steps=100, window=30)
        with self.assertRaises(ValueError):
            TrainingPlan(steps=100, checkpoints=(0, 150))


def _policy(**over):
    kw = dict(
        compile_distances=[{m: 2e-3 for m in P4.HARD_METRICS}],
        repeat_distances=[{m: 5e-4 for m in P4.HARD_METRICS}],
        workload_id="realtext", workload_hash="h", dtype="bfloat16",
        environment_hash="env", patch_set=QKV, data_identity_digest="data",
        training_plan=TrainingPlan().to_dict(),
    )
    kw.update(over)
    return P4.derive_policy(**kw)


def _measured(kl=1e-3, grad=1e-3, train=1e-3, val=1e-3, **over):
    m = {"kl_mean": kl, "global_grad_rel_l2": grad,
         "train_window_nll_max_abs_delta": train, "val_nll_max_abs_delta": val,
         "missing_grads": [], "grad_presence": {"missing": [], "shape_mismatch": [], "extra": []},
         "finite": {"ok": True}, "non_finite_training_steps": []}
    m.update(over)
    return m


class TestPolicy(unittest.TestCase):
    def test_threshold_is_margin_times_max_of_three_terms(self):
        p = _policy()
        for m in P4.HARD_METRICS:
            self.assertAlmostEqual(p.thresholds[m], 2e-3 * P4.MARGIN)
            self.assertEqual(p.derivation[m]["binding_term"], "compile_vs_eager")

    def test_independent_floors_bind_independently(self):
        p = _policy(compile_distances=[{m: 0.0 for m in P4.HARD_METRICS}],
                    repeat_distances=[{m: 0.0 for m in P4.HARD_METRICS}])
        self.assertAlmostEqual(p.thresholds["kl_mean"], 1e-6 * P4.MARGIN)
        self.assertAlmostEqual(p.thresholds["global_grad_rel_l2"], 1e-4 * P4.MARGIN)
        self.assertNotEqual(P4.FLOORS["kl_mean"], P4.FLOORS["global_grad_rel_l2"])

    def test_identity_rejection_on_every_bound_field(self):
        p = _policy()
        good = dict(workload_id="realtext", workload_hash="h", dtype="bfloat16",
                    environment_hash="env", patch_set=QKV, data_identity_digest="data",
                    training_plan=TrainingPlan().to_dict())
        p.require_binding(**good)
        for field, bad in (("workload_id", "synthetic-canonical"), ("workload_hash", "x"),
                           ("dtype", "float16"), ("environment_hash", "other"),
                           ("patch_set", RESIDUAL), ("data_identity_digest", "other"),
                           ("training_plan", TrainingPlan(steps=500, checkpoints=(0, 500)).to_dict())):
            with self.assertRaises(PolicyMismatch, msg=field):
                p.require_binding(**{**good, field: bad})

    def test_the_synthetic_calibration_cannot_bind_to_real_text(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import CANONICAL, WorkloadSpec

        real = WorkloadSpec(weights=P4.__name__ and "pretrained:Qwen/Qwen3-0.6B@c1899de2",
                            data="wikitext-2-raw:b08601e0").validate()
        self.assertNotEqual(real.workload_hash, CANONICAL.workload_hash)
        p = _policy(workload_id=CANONICAL.workload_id, workload_hash=CANONICAL.workload_hash)
        with self.assertRaises(PolicyMismatch):
            p.require_binding(workload_id=real.workload_id, workload_hash=real.workload_hash,
                              dtype="bfloat16", environment_hash="env", patch_set=QKV,
                              data_identity_digest="data",
                              training_plan=TrainingPlan().to_dict())

    def test_candidate_has_no_route_into_the_derivation(self):
        import inspect
        params = set(inspect.signature(P4.derive_policy).parameters)
        for name in ("candidate", "program", "module", "kernels", "artifact"):
            self.assertNotIn(name, params)
        p = _policy()
        before = dict(p.thresholds)
        for v in (0.0, 1.0, 1e6):
            P4.check(p, _measured(kl=v, grad=v, train=v, val=v))
        self.assertEqual(p.thresholds, before)

    def test_inside_passes_outside_fails_and_names_the_metric(self):
        p = _policy()
        self.assertTrue(P4.check(p, _measured())["ok"])
        for m in P4.HARD_METRICS:
            v = P4.check(p, _measured(**{ {"kl_mean": "kl", "global_grad_rel_l2": "grad",
                                            "train_window_nll_max_abs_delta": "train",
                                            "val_nll_max_abs_delta": "val"}[m]: 5e-3}))
            self.assertFalse(v["ok"]); self.assertEqual(v["failed_at"], m)

    def test_presence_and_finiteness_outrank_magnitudes(self):
        p = _policy()
        self.assertEqual(P4.check(p, _measured(missing_grads=["x"]))["failed_at"], "presence")
        self.assertEqual(P4.check(p, _measured(finite={"ok": False}))["failed_at"], "finite")
        self.assertEqual(P4.check(p, _measured(non_finite_training_steps=[7]))["failed_at"], "finite")

    def test_old_schemas_are_refused_not_reinterpreted(self):
        payload = _policy().to_dict()
        for old in ("evograd-qwen3-t3-numerics/3", "evograd-qwen3-t3-numerics/2"):
            with self.assertRaises(PolicyMismatch):
                P4.Protocol4Policy.from_dict({**payload, "schema": old})

    def test_a_failed_enforced_gate_stops_before_timing(self):
        # The runner's contract: `model_correctness_hook` returns ok=False and
        # tier3 does not time. Exercised through the same summarize() path the
        # CLI uses, with a verdict whose only failure is the protocol-4 stage.
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.gate import summarize
        verdict = {"ok": False, "failed_at": "numerical_envelopes",
                   "reason": "kl_mean = 1.0e-02 > 4.0e-03 (2.50x)", "stages": []}
        summary = summarize(verdict)
        self.assertFalse(summary["ok"])
        self.assertIn("kl_mean", str(summary.get("reason", "")))


if __name__ == "__main__":
    unittest.main()


class _ToyLM(torch.nn.Module):
    """A causal LM stand-in: embeddings -> linear -> logits, with `.logits`."""

    def __init__(self, vocab=37, dim=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, use_cache=False, labels=None):
        class Out:  # the two attributes the loop reads
            pass
        out = Out()
        out.logits = self.head(torch.tanh(self.embed(input_ids)))
        return out


class _ToyBatches:
    def __init__(self, n, batch, length, vocab, seed):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab, (n, batch, length), generator=g)
        self.batches_per_epoch = n

    def batch(self, step):
        ids = self.data[step % self.batches_per_epoch]
        return ids, ids.clone()

    def describe(self):
        return {"n": self.batches_per_epoch}


class _ToyWorkload:
    last_build = None

    def __init__(self, seed=0):
        self.seed = seed

    def build_patched(self, kernels):
        class Prov:
            def to_dict(self):
                return {"method": "toy"}
        return _ToyLM(seed=self.seed), Prov()


class TestTrainingLoopIntegration(unittest.TestCase):
    """The loop itself, on a toy model: continuous state, windows, evaluator."""

    def _run(self, kernels="k", seed=0, evaluator_seed=99):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.training import run_training

        plan = TrainingPlan(steps=6, window=2, checkpoints=(0, 2, 6), learning_rate=1e-2)
        train = _ToyBatches(4, 2, 9, 37, seed=1)
        val = _ToyBatches(3, 2, 9, 37, seed=2)
        captured = {}

        def evaluator_factory():
            m = _ToyLM(seed=evaluator_seed)          # deliberately different init
            captured["evaluator"] = m
            return m

        result = run_training(_ToyWorkload(seed=seed), kernels, plan=plan,
                              train_batches=train, validation_batches=val,
                              validation_steps=3, data_seed=1,
                              evaluator_factory=evaluator_factory)
        return result, captured

    def test_windows_are_token_weighted_means_and_cover_the_horizon(self):
        result, _ = self._run()
        self.assertEqual(len(result["train_window_nll"]), 3)
        for w in result["train_windows"]:
            self.assertEqual(w["valid_tokens"], 2 * 2 * 8)      # window*batch*(len-1)
        # token-weighted window mean == mean of the two per-step means here
        # because every step has the same token count
        steps = result["train_per_step_nll"]
        self.assertAlmostEqual(result["train_window_nll"][0], (steps[0] + steps[1]) / 2, places=6)

    def test_validation_goes_through_the_evaluator_with_the_trained_weights(self):
        result, captured = self._run()
        self.assertEqual(sorted(result["validation_nll"]), ["0", "2", "6"])
        # The evaluator started from a different init; if the trained weights
        # had not been loaded into it, checkpoint 0 could not equal a fresh
        # trained model's own validation loss. Recompute independently.
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.training import evaluate_validation
        fresh = _ToyLM(seed=0)
        expected0 = evaluate_validation(fresh, _ToyBatches(3, 2, 9, 37, seed=2), steps=3)["nll_mean"]
        self.assertAlmostEqual(result["validation_nll"]["0"], expected0, places=5)
        self.assertNotAlmostEqual(result["validation_nll"]["0"], result["validation_nll"]["6"], places=3)

    def test_training_is_continuous_and_deterministic(self):
        a, _ = self._run(seed=0)
        b, _ = self._run(seed=0)
        self.assertEqual(a["train_per_step_nll"], b["train_per_step_nll"])
        self.assertEqual(a["validation_nll"], b["validation_nll"])
        # the loss moves under training (the optimizer state is live, not reset)
        self.assertLess(a["train_window_nll"][-1], a["train_window_nll"][0])

    def test_identical_providers_have_zero_training_distance(self):
        a, _ = self._run(seed=0)
        b, _ = self._run(seed=0)
        d = training_distances(a, b)
        self.assertEqual(d["train_window_nll_max_abs_delta"], 0.0)
        self.assertEqual(d["val_nll_max_abs_delta"], 0.0)


class TestHoldoutRecord(unittest.TestCase):
    """The holdout file must carry the eager reference curves, not only distances to them."""

    def test_the_payload_stores_every_seed_s_eager_curve_whole(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import holdout_payload
        ref = {"train_per_step_nll": [2.9, 2.8, 7.1], "train_window_nll": [2.85, 7.1],
               "validation_nll": {"0": 3.1, "2": 7.0}, "provider": "eager"}
        results = [{"seed": 17, "provider": "compile", "measured": {"train_window_deltas": [0.0, 5.0]},
                    "verdict": {"ok": False}}]
        payload = holdout_payload("policy.json", results, {17: ref})
        self.assertEqual(payload["results"], results)
        self.assertEqual(payload["eager_reference_train"]["17"], ref)
        self.assertEqual(payload["eager_reference_train"]["17"]["train_per_step_nll"][2], 7.1)
        self.assertEqual(payload["policy_file"], "policy.json")

    def test_the_holdout_command_writes_through_the_payload_builder(self):
        import inspect
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import protocol4_cli
        source = inspect.getsource(protocol4_cli.command_holdout)
        self.assertIn("references[seed] = ref_train", source)
        self.assertIn("holdout_payload(args.policy, results, references, diagnostics)", source)
        self.assertNotIn('"results": results})', source)

    def test_part_d_reader_ignores_the_new_key(self):
        import json, tempfile
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import holdout_payload
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import _protocol4_training_part
        rows = [{"seed": 17, "provider": "compile",
                 "measured": {"kernel_origin": ["trusted_torch_compile"],
                              "train_window_nll_max_abs_delta": 6.0, "val_nll_max_abs_delta": 4.5}}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(holdout_payload("p.json", rows, {17: {"train_per_step_nll": [1.0]}}), fh)
        part = _protocol4_training_part(fh.name, ("trusted_torch_compile",))
        self.assertAlmostEqual(part["train_window_nll_max_abs_delta"], 6.0)


class TestScreeningPolicy(unittest.TestCase):
    """An A/B/C screening policy: two enforced metrics, SCREENING_PLAN, no part D."""

    def _samples(self, kl, grad):
        return [{"kl_mean": kl, "global_grad_rel_l2": grad}]

    def _derive(self, **kw):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4 import (
            SCREENING_METRICS, SCREENING_PLAN, derive_policy)
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import PatchSet
        base = dict(compile_distances=self._samples(1e-3, 3e-2),
                    repeat_distances=self._samples(0.0, 2e-3),
                    workload_id="w", workload_hash="h", dtype="bfloat16", environment_hash="e",
                    patch_set=PatchSet(("attention",), ("qkv_norm_rope",), {}),
                    data_identity_digest="d", training_plan=SCREENING_PLAN,
                    metrics=SCREENING_METRICS)
        base.update(kw)
        return derive_policy(**base)

    def test_screening_thresholds_cover_exactly_b_and_c(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4 import (
            is_screening, requires_training_part)
        policy = self._derive()
        self.assertEqual(sorted(policy.thresholds), ["global_grad_rel_l2", "kl_mean"])
        self.assertAlmostEqual(policy.thresholds["kl_mean"], 2e-3)
        self.assertAlmostEqual(policy.thresholds["global_grad_rel_l2"], 6e-2)
        self.assertTrue(is_screening(policy))
        self.assertFalse(requires_training_part(policy))
        self.assertEqual(policy.to_dict()["hard_metrics"], list(policy.thresholds))

    def test_screening_metrics_require_the_screening_plan_and_vice_versa(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4 import HARD_METRICS
        with self.assertRaises(ValueError):
            self._derive(training_plan={"steps": 1000})
        with self.assertRaises(ValueError):
            self._derive(metrics=HARD_METRICS)

    def test_check_judges_only_the_metrics_the_policy_carries(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4 import check
        policy = self._derive()
        measured = {"kl_mean": 1.5e-3, "global_grad_rel_l2": 4e-2, "missing_grads": [],
                    "grad_presence": {"missing": [], "shape_mismatch": [], "extra": []},
                    "finite": {"ok": True}}
        verdict = check(policy, measured)
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["screening"])
        self.assertEqual(sorted(verdict["ratios"]), ["global_grad_rel_l2", "kl_mean"])
        measured["kl_mean"] = 3e-3
        self.assertEqual(check(policy, measured)["failed_at"], "kl_mean")

    def test_a_four_part_policy_still_needs_all_four(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4 import (
            HARD_METRICS, check, derive_policy, requires_training_part)
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import PatchSet
        full = {m: 1e-3 for m in HARD_METRICS}
        policy = derive_policy(compile_distances=[full], repeat_distances=[dict(full)],
                               workload_id="w", workload_hash="h", dtype="bfloat16",
                               environment_hash="e", patch_set=PatchSet(("qkv_norm_rope",), ("attention",), {}),
                               data_identity_digest="d", training_plan={"steps": 1000})
        self.assertTrue(requires_training_part(policy))
        verdict = check(policy, {"kl_mean": 1e-4, "global_grad_rel_l2": 1e-4,
                                 "finite": {"ok": True}, "grad_presence": {}})
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "train_window_nll_max_abs_delta")

    def test_screening_rows_are_matched_by_origin(self):
        import json, tempfile
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import _protocol4_screening_rows
        rows = [{"seed": 11, "provider": "compile",
                 "measured": {"kernel_origin": ["trusted_torch_compile"]},
                 "verdict": {"ok": True, "ratios": {"kl_mean": 0.4}}},
                {"seed": 11, "provider": "candidate",
                 "measured": {"kernel_origin": ["baseline:liger"]},
                 "verdict": {"ok": False, "ratios": {}}}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"results": rows}, fh)
        got = _protocol4_screening_rows(fh.name, ("trusted_torch_compile",))
        self.assertEqual([r["seed"] for r in got], [11])
        self.assertTrue(got[0]["ok"])
        self.assertEqual(_protocol4_screening_rows(fh.name, ("candidate:direct_deployment",)), [])
        self.assertEqual(_protocol4_screening_rows(None, ("x",)), [])


class TestPatchSpecs(unittest.TestCase):
    def test_patch_specs_parse_and_refuse_duplicates(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import (
            parse_patch_set_spec, parse_patch_specs)
        self.assertEqual(parse_patch_specs(["attention=compile", "residual_rmsnorm=liger",
                                            "qkv_norm_rope=examples/a.py"]),
                         {"attention": "compile", "residual_rmsnorm": "liger",
                          "qkv_norm_rope": "examples/a.py"})
        with self.assertRaises(ValueError):
            parse_patch_specs(["attention=compile", "attention=liger"])
        with self.assertRaises(ValueError):
            parse_patch_specs(["attention"])
        name, patches = parse_patch_set_spec("all_compiled:attention=compile,swiglu_mlp=compile")
        self.assertEqual(name, "all_compiled")
        self.assertEqual(patches, {"attention": "compile", "swiglu_mlp": "compile"})
        with self.assertRaises(ValueError):
            parse_patch_set_spec("no_colon_here")

    def test_candidate_patch_set_must_equal_the_policy_sites(self):
        from types import SimpleNamespace
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import (
            build_kernels, patches_of, sites_of)
        args = SimpleNamespace(sites="attention,swiglu_mlp", patch=["attention=compile"],
                               candidate=None)
        self.assertEqual(sites_of(args), ("attention", "swiglu_mlp"))
        self.assertEqual(patches_of(args), {"attention": "compile"})
        workload = SimpleNamespace(site_registry=None)
        with self.assertRaises(ValueError):
            build_kernels(workload, "candidate", None, sites=sites_of(args), patches=patches_of(args))
        legacy = SimpleNamespace(sites="qkv_norm_rope", patch=[], candidate="examples/h.py")
        self.assertEqual(patches_of(legacy), {"qkv_norm_rope": "examples/h.py"})

    def test_liger_route_is_only_offered_where_declared(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import kernels_from_patches
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites
        with self.assertRaises(ValueError):
            kernels_from_patches({"attention": "liger"}, qwen3_sites())
        kernels = kernels_from_patches({"residual_rmsnorm": "liger"}, qwen3_sites())
        self.assertEqual(kernels.patched, ("residual_rmsnorm",))
        self.assertEqual([s.origin for s in kernels.sources], ["baseline:liger"])


class TestGateDispatch(unittest.TestCase):
    """With a protocol-4 policy the runner must not consult the synthetic canonical calibration."""

    def _workload(self, **extra):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import PRETRAINED, REAL_TEXT
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        return Qwen3Workload.from_config({"dtype": "bfloat16", "device": "cpu", "seed": 0,
                                          "data_seed": 0, "weights": PRETRAINED, "data": REAL_TEXT,
                                          "calibration_path": "/nonexistent/canonical.json",
                                          "protocol4_calibration_path": "/some/policy.json",
                                          "protocol4_verdict_path": "/some/holdout.json", **extra})

    def test_protocol4_hook_is_reached_before_the_legacy_calibration(self):
        workload = self._workload()
        sentinel = {"gate": "qwen3_protocol4", "ok": True}
        object.__setattr__(workload, "_protocol4_hook", lambda kernels, device: sentinel)
        self.assertIs(workload.model_correctness(object(), device="cpu"), sentinel)

    def test_diagnostic_timing_flag_threads_through_the_cli_config(self):
        import evograd.evaluation.tier3.cli as tier3_cli
        args = tier3_cli._parser().parse_args(
            ["--model", "qwen3_0_6b", "--real-text", "--device", "cpu",
             "--protocol4-calibration", "p.json", "--protocol4-verdict", "h.json",
             "--protocol4-diagnostic-timing"])
        workload = tier3_cli.build_workload(args)
        self.assertTrue(workload.protocol4_diagnostic_timing)
        self.assertTrue(workload.to_config()["protocol4_diagnostic_timing"])
        plain = tier3_cli._parser().parse_args(["--model", "qwen3_0_6b", "--device", "cpu"])
        self.assertFalse(tier3_cli.build_workload(plain).protocol4_diagnostic_timing)

    def test_the_hook_refuses_a_failed_screening_unless_diagnostic(self):
        """The property, not the spelling.

        Behaviour is exercised end to end in ``tests/qwen3/test_report_first.py``
        (strict refuses a failed frozen screening; ``--protocol4-diagnostic-timing``
        times it anyway, labelled; report-first records it and continues). Here:
        the guard and the label still exist in the hook, and the protocol-4 hook
        is still reached before the legacy calibration.
        """
        import inspect
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        source = inspect.getsource(Qwen3Workload._protocol4_hook)
        self.assertIn("self.protocol4_diagnostic_timing", source)
        self.assertIn('"screening_holdout"', source)
        self.assertIn('measured["diagnostic_only"]', source)
        dispatch = inspect.getsource(Qwen3Workload.model_correctness)
        self.assertLess(dispatch.index("self._protocol4_hook(kernels, device)"),
                        dispatch.index("load_policy(self.calibration_path)"))


class TestRunnerWiring(unittest.TestCase):
    def test_the_hook_refuses_without_a_holdout_verdict_for_the_provider(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import (
            _protocol4_training_part,
        )
        self.assertIsNone(_protocol4_training_part(None, ("trusted_torch_compile",)))
        self.assertIsNone(_protocol4_training_part("/nonexistent/verdict.json",
                                                   ("candidate:direct_deployment",)))

    def test_part_d_is_matched_by_kernel_origin_and_takes_the_worst_seed(self):
        import json, tempfile
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import (
            _protocol4_training_part,
        )
        rows = [
            {"seed": 11, "provider": "compile",
             "measured": {"kernel_origin": ["trusted_torch_compile"],
                          "train_window_nll_max_abs_delta": 1e-3, "val_nll_max_abs_delta": 2e-3}},
            {"seed": 17, "provider": "compile",
             "measured": {"kernel_origin": ["trusted_torch_compile"],
                          "train_window_nll_max_abs_delta": 3e-3, "val_nll_max_abs_delta": 1e-3}},
            {"seed": 11, "provider": "candidate",
             "measured": {"kernel_origin": ["candidate:direct_deployment"],
                          "train_window_nll_max_abs_delta": 9e-1, "val_nll_max_abs_delta": 9e-1}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"results": rows}, fh)
        part = _protocol4_training_part(fh.name, ("trusted_torch_compile",))
        self.assertAlmostEqual(part["train_window_nll_max_abs_delta"], 3e-3)
        self.assertAlmostEqual(part["val_nll_max_abs_delta"], 2e-3)
        self.assertEqual(part["training_seeds"], [11, 17])

    def test_the_cli_builds_a_real_text_workload_that_carries_the_protocol_paths(self):
        from evograd.evaluation.tier3.cli import _parser, build_workload
        args = _parser().parse_args([
            "--model", "qwen3_0_6b", "--real-text", "--device", "cpu",
            "--protocol4-calibration", "/p/policy.json",
            "--protocol4-verdict", "/p/holdout.json", "--baseline", "none"])
        w = build_workload(args)
        self.assertTrue(w.spec.real_text and w.spec.pretrained)
        self.assertIn("pretrained.realtext", w.spec.workload_id)
        self.assertEqual(w.protocol4_calibration_path, "/p/policy.json")
        self.assertEqual(w.protocol4_verdict_path, "/p/holdout.json")
        # and the child rebuilds the same thing from the serialized config
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        again = Qwen3Workload.from_config(w.to_config())
        self.assertEqual(again.spec.workload_id, w.spec.workload_id)
        self.assertEqual(again.protocol4_verdict_path, w.protocol4_verdict_path)

    def test_the_synthetic_canonical_identity_is_unchanged_by_the_new_fields(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import CANONICAL
        self.assertTrue(CANONICAL.workload_id.endswith("6e7919ad"))
        self.assertNotIn("weights", CANONICAL.to_dict())
        self.assertNotIn("data", CANONICAL.to_dict())


class TestPretrainedArchitectureGuard(unittest.TestCase):
    def test_the_two_rope_spellings_are_the_same_rotation(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import rope_settings
        old = {"rope_theta": 1000000.0, "rope_scaling": None}
        new = {"rope_theta": None, "rope_scaling": {"rope_theta": 1000000, "rope_type": "default"}}
        self.assertEqual(rope_settings(old), rope_settings(new))
        self.assertEqual(rope_settings(new), (1000000.0, "default"))

    def test_a_different_theta_or_type_is_still_a_disagreement(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import rope_settings
        base = {"rope_theta": 1000000.0, "rope_scaling": None}
        self.assertNotEqual(rope_settings(base), rope_settings({"rope_theta": 10000.0, "rope_scaling": None}))
        self.assertNotEqual(rope_settings(base), rope_settings(
            {"rope_theta": None, "rope_scaling": {"rope_theta": 1000000, "rope_type": "yarn"}}))

    def test_the_declared_qwen3_arch_matches_the_pinned_checkpoint_config(self):
        import json, os
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.model import rope_settings
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level4.spec import QWEN3_0_6B
        snapshot = ("/u/wzhan/.cache/evograd-mF/hf/models--Qwen--Qwen3-0.6B/snapshots/"
                    "c1899de289a04d12100db370d81485cdf75e47ca/config.json")
        if not os.path.exists(snapshot):
            self.skipTest("pinned checkpoint not present locally")
        loaded = json.load(open(snapshot))
        self.assertEqual(rope_settings(QWEN3_0_6B), rope_settings(loaded))
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
                    "tie_word_embeddings", "attention_bias", "rms_norm_eps"):
            self.assertEqual(loaded.get(key), QWEN3_0_6B[key], key)
