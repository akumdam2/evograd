"""OLMoE-1B-7B real-payload capture, on a tiny random OLMoE on CPU.

Instrumentation does not change the step (the HF loss with the router's
load-balancing term), every call maps to exactly one case -- the q and k rope
calls have identical shapes here, as in the real model, and are told apart by
role -- payloads round-trip in each task's ABI (the experts' routing indices
and weights included), the model reference reproduces every live output bit
for bit (the whole attention module, and the residual site at all three of its
wirings: post-attention, into the next layer's input norm, into ``model.norm``),
each residual payload's ``dsummed`` is the sum's external gradient only, and
model Gate 2 holds the routing to the native run's while Gate 3 leaves it free.
Both gates judge that same loss, the load-balancing term included.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

B, T, H, I, HEADS, D, E, K, V, LAYERS = 2, 16, 64, 32, 4, 16, 4, 2, 512, 2


def tiny_cases():
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Provenance, Workload
    from evograd.verification.integrations.base import Case

    n, bf = B * T, "bfloat16"
    gapped = (("a", (2 * I, 1)), ("b", (2 * I, 1)))
    rows = [
        ("linear_no_bias", "qkvo", {"M": n, "K": H, "N": H}, bf, 4 * LAYERS, ()),
        ("linear_no_bias", "lm_head", {"M": n, "K": H, "N": V}, bf, 1, ()),
        ("rmsnorm", "residual_norm", {"rows": n, "hidden": H}, bf, 4 * LAYERS + 1, ()),
        ("rope", "rope", {"B": B, "n_heads": HEADS, "T": T, "head_dim": D}, bf, LAYERS, ()),
        ("rope", "rope_kv", {"B": B, "n_heads": HEADS, "T": T, "head_dim": D}, bf, LAYERS, ()),
        ("swiglu", "moe_activation", {"rows": n * K, "cols": I}, bf, LAYERS, gapped),
        ("causal_gqa_attention", "sdpa", {"B": B, "HQ": HEADS, "HK": HEADS, "T": T, "D": D}, bf,
         LAYERS, ()),
        ("cross_entropy", "logits", {"rows": n, "cols": V}, "float32", 1, ()),
        ("olmoe_attention_block", "attention_block", {"B": B, "T": T, "H": H, "HQ": HEADS,
                                                      "HK": HEADS, "D": D, "QO": H, "KVO": H},
         bf, LAYERS, ()),
        ("olmoe_moe_experts", "experts", {"T": n, "H": H, "E": E, "G": 2 * I, "I": I, "K": K}, bf,
         LAYERS, ()),
        ("olmoe_residual_rmsnorm", "residual", {"B": B, "T": T, "H": H}, bf, 2 * LAYERS, ()),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level,
                      Workload(dims=dims, dtype=dtype,
                               provenance=Provenance(model="tiny", component=slug, strides=strides)),
                      freq) for task, slug, dims, dtype, freq, strides in rows)


class RealCaseTable(unittest.TestCase):
    def test_observed_cases(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.level == 1 for c in cases.values()), 8)
        self.assertEqual({c.case_id: c.frequency for c in cases.values() if c.level == 2},
                         {f"{manifest.task_key(site)}/{site}": manifest.expected_model_frequency(site)
                          for site in manifest.SITES})
        self.assertEqual(cases["rmsnorm/residual_norm"].frequency, 65)

    def test_model_gates(self):
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        self.assertTrue(MODEL.model_gates)
        self.assertTrue(MODEL.gate3_applies)


def tiny_workload():
    from evograd.benchmark.topdown.olmoe_1b_7b.architecture import OLMOE_1B_7B, with_layers
    from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload

    arch = {**with_layers(OLMOE_1B_7B, LAYERS), "hidden_size": H, "intermediate_size": I,
            "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "num_experts": E,
            "num_experts_per_tok": K, "vocab_size": V, "eos_token_id": None, "pad_token_id": 1,
            "max_position_embeddings": 512}
    return OlmoeWorkload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T, arch_overrides=arch)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class ForcedRouting(unittest.TestCase):
    """Model Gate 2 runs on the native run's routing, and changes nothing else."""

    def setUp(self):
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.verification.common.execution import Deployment, Sample
        from evograd.verification.common.tensors import isolated_rng
        from evograd.verification.gate2.runner import SiteHooks
        from evograd.verification.integrations.olmoe_1b_7b import routing as R

        self.R, self.KernelSet, self.isolated_rng = R, KernelSet, isolated_rng
        self.native, self.hooks = Deployment.native(), SiteHooks()
        self.workload = tiny_workload()
        self.scope = R.ForcedRoutingScope(self.workload)
        self.sample = Sample("tiny/seed3", 3, {})

    def build(self):
        with self.isolated_rng(self.sample.seed, "cpu"):
            return self.scope.build(self.native, self.sample, self.hooks)

    def plain(self):
        with self.isolated_rng(self.sample.seed, "cpu"):
            model, _ = self.workload.build_patched(self.KernelSet(registry=self.workload.site_registry))
        return model.train()

    def step(self, model):
        ids, labels = self.workload.batch_for(seed=self.sample.seed)
        out = model(input_ids=ids, labels=labels, use_cache=False)
        out.loss.backward()
        return out.logits.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    def test_the_native_run_is_unchanged_bit_for_bit(self):
        logits, grads = self.step(self.build())
        plain_logits, plain_grads = self.step(self.plain())
        self.assertTrue(torch.equal(logits, plain_logits))
        self.assertEqual(sorted(grads), sorted(plain_grads))
        for name, grad in grads.items():
            self.assertTrue(torch.equal(grad, plain_grads[name]), name)

    def experts_used(self, model):
        used = {}
        for path, router in self.R.routers(model):
            experts = model.get_submodule(path.rsplit(".", 1)[0]).experts
            experts.register_forward_pre_hook(
                lambda _m, a, path=path: used.__setitem__(path, a[1].clone()))
        return used

    def perturb(self, model):
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for _path, router in self.R.routers(model):
                router.weight.add_(torch.randn(router.weight.shape, generator=generator).to(router.weight))

    def test_a_flip_cannot_happen_on_forced_routing(self):
        route = self.scope.route(self.sample)
        free, forced = self.plain(), self.build()
        self.perturb(free), self.perturb(forced)
        free_used, forced_used = self.experts_used(free), self.experts_used(forced)
        self.step(free), self.step(forced)
        self.assertTrue(any(not torch.equal(free_used[p], route[p]) for p in route), "nothing flipped")
        for path, indices in route.items():
            self.assertTrue(torch.equal(forced_used[path], indices), path)

    def test_the_router_is_still_trained(self):
        model = self.build()
        self.step(model)
        for path, router in self.R.routers(model):
            self.assertTrue(router.weight.grad.abs().sum() > 0, path)

    def test_a_patched_site_is_held_to_it_too(self):
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.execution import candidate
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        site = "moe_experts"
        task = self.workload.site_registry.require(site).op
        deployment = candidate("c", {site: resolve_runtime_forward(MODEL.reference_op(task))}, programs={})
        route = self.scope.route(self.sample)
        with self.isolated_rng(self.sample.seed, "cpu"):
            model = self.scope.build(deployment, self.sample, self.hooks)
        self.perturb(model)
        seen = {}
        for path, router in self.R.routers(model):
            router.register_forward_hook(lambda _m, _a, out, path=path: seen.__setitem__(path, out[2].clone()))
        self.step(model)
        self.assertEqual(self.scope.expected_invocations(deployment), {site: LAYERS})
        self.assertEqual(sorted(seen), sorted(route))
        for path, indices in route.items():
            self.assertTrue(torch.equal(seen[path], indices), path)

    def test_recording_a_route_leaves_a_seeded_build_alone(self):
        with self.isolated_rng(self.sample.seed, "cpu"):
            before = torch.get_rng_state()
            self.scope.route(self.sample)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_each_sample_is_recorded_once(self):
        first = self.scope.route(self.sample)
        self.assertIs(self.scope.route(self.sample), first)

    def test_the_rule_is_bound_into_the_policy(self):
        self.assertEqual(self.scope.identity()["routing"], self.R.RULE)
        self.assertIn("forced_outputs", self.scope.reference_parts())

    def test_the_backward_is_tier3s_loss_with_the_load_balancing_term(self):
        """The logits are judged; the loss, load-balancing term included, only
        seeds the backward, so the gradients compared are ``workload.loss``'s."""
        from evograd.verification import gate2, gate3

        ex = gate2.execute(self.scope, self.native, self.sample)
        self.assertIsNone(ex.error, ex.error)
        self.assertEqual({k for k in ex.results if k.startswith("out:")}, {"out:logits"})
        batch = self.workload.batch_for(seed=self.sample.seed)
        model = self.build()
        out = model(input_ids=batch[0], labels=batch[1], use_cache=False)
        self.assertTrue(torch.equal(ex.results["out:logits"], out.logits.detach()))
        self.assertTrue(torch.equal(out.loss, self.workload.loss(self.build(), batch)))
        total, count = gate3.next_token_nll(out.logits, batch[1])
        aux = model.config.router_aux_loss_coef * float(out.aux_loss.detach())
        self.assertGreater(aux, 0.0)
        self.assertAlmostEqual(float(out.loss) - float(total.detach() / count), aux, places=5)
        out.loss.backward()
        params = dict(model.named_parameters())
        grads = {k: v for k, v in ex.results.items() if k.startswith("grad:")}
        self.assertTrue(grads)
        for name, grad in grads.items():
            self.assertTrue(torch.equal(grad, params[name.split(":", 1)[1]].grad), name)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TheGatesReachOLMoE(unittest.TestCase):
    def test_model_scope_is_the_forced_one(self):
        from evograd.verification.integrations.olmoe_1b_7b import MODEL, routing as R
        from evograd.verification.integrations.gates import model_scope

        self.assertIsInstance(model_scope(MODEL, tiny_workload()), R.ForcedRoutingScope)

    def test_gate3_trains_on_free_routing(self):
        from evograd.verification import gate3
        from evograd.verification.common.execution import Deployment
        from evograd.verification.gate2.runner import SiteHooks
        from evograd.verification.gate3.adapters import SeedBatches
        from evograd.verification.integrations.olmoe_1b_7b import MODEL, routing as R

        workload = tiny_workload()
        adapter = MODEL.objective.training(workload, train_data=SeedBatches(workload, 0, "train"),
                                           optimizer=gate3.adamw(**{"lr": 1e-4, **MODEL.gate3_optimizer}))
        model = adapter.build_model(Deployment.native(), SiteHooks())
        self.assertTrue(R.routers(model))
        self.assertFalse(any(router._forward_hooks for _p, router in R.routers(model)))
        loss, weight = adapter.train_loss(model, adapter.train_batch(0))
        self.assertTrue(torch.isfinite(loss))
        # The tier-3 loss, load-balancing term included, with weight 1.
        self.assertEqual(weight, 1.0)
        self.assertTrue(torch.equal(loss, workload.loss(model, adapter.train_batch(0))))
        self.assertFalse(adapter.build_evaluator().training)

    def test_a_candidate_passes_model_gate2_end_to_end(self):
        """The reference as a candidate, through calibration and the verdict, on a tiny model."""
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.execution import candidate
        from evograd.verification.integrations.gates import judge_model
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        workload = tiny_workload()
        site = "moe_experts"
        task = workload.site_registry.require(site).op
        deployment = candidate("c", {site: resolve_runtime_forward(MODEL.reference_op(task))}, programs={})
        reports = judge_model(MODEL, deployment, workload, compile_backend="eager")
        self.assertEqual(reports["gate2_model"].status, "passed", reports["gate2_model"].to_dict())

    def test_a_faulty_candidate_fails_model_gate2(self):
        """A 10% error at the experts fails. On this tiny bf16 model 1% sits inside the noise the
        thresholds are calibrated from; the real model's sensitivity is not measured here."""
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.execution import candidate
        from evograd.verification.integrations.gates import judge_model
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        workload = tiny_workload()
        site = "moe_experts"
        forward = resolve_runtime_forward(MODEL.reference_op(workload.site_registry.require(site).op))

        def faulty(*args):
            return forward(*args) * 1.1

        reports = judge_model(MODEL, candidate("c", {site: faulty}, programs={}), workload,
                              compile_backend="eager")
        self.assertEqual(reports["gate2_model"].status, "failed", reports["gate2_model"].to_dict())


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.olmoe_1b_7b.architecture import OLMOE_1B_7B, with_layers
        from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.olmoe_1b_7b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.Q, cls.C = Q, C
        arch = {**with_layers(OLMOE_1B_7B, LAYERS), "hidden_size": H, "intermediate_size": I,
                "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "num_experts": E,
                "num_experts_per_tok": K, "vocab_size": V, "eos_token_id": None, "pad_token_id": 1,
                "max_position_embeddings": 512}
        workload = OlmoeWorkload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                 arch_overrides=arch)

        def build():
            model = Q.build_model(workload)
            g = torch.Generator().manual_seed(11)
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if name.endswith("norm.weight"):
                        p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            return model

        def step(ids, order, writer):
            model = build()
            recorder = C.LiveCapture(model).install()
            try:
                output = Q.run_step(model, ids, ids.clone())
            finally:
                recorder.remove()
            Q.record_step(writer, C.invocations(recorder, model, cls.cases),
                          {"order": order, "sha256": str(order)})
            return model, output

        cls.cases = tiny_cases()
        cls.ids = torch.randint(2, V, (B, T), generator=torch.Generator().manual_seed(7))
        plain = build()
        cls.plain = Q.step_record(plain, Q.run_step(plain, cls.ids, cls.ids.clone()))
        del plain
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        cls.model, output = step(cls.ids, 101, writer)
        cls.instrumented = Q.step_record(cls.model, output)
        step(torch.randint(2, V, (B, T), generator=torch.Generator().manual_seed(8)), 303, writer)
        cls.coverage = Q.coverage(cls.cases, writer.payloads)
        cls.corpus = Corpus.open(writer.finish(identity={"name": "tiny"}, splits=Q.splits(writer.payloads),
                                               coverage=cls.coverage))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_instrumentation_changes_nothing(self):
        self.assertEqual(self.plain["loss_digest"], self.instrumented["loss_digest"])
        self.assertEqual(self.plain["logits_digest"], self.instrumented["logits_digest"])
        self.assertEqual(self.plain["grads"], self.instrumented["grads"])

    def test_every_call_is_one_payload_of_its_case(self):
        self.assertTrue(self.coverage["complete"], self.coverage["problems"])
        roles = {r["case_id"] for r in self.corpus.payloads() if r["task"] == "rope"}
        self.assertEqual(roles, {"rope/rope", "rope/rope_kv"})

    def test_payloads_match_each_task_interface(self):
        from evograd.benchmark import get_task
        from evograd.verification.payloads.gate1 import check_payload_interface

        for record in self.corpus.payloads():
            self.assertEqual(check_payload_interface(get_task(record["task"]), record), [],
                             record["payload_id"])

    def test_the_residual_site_is_every_add_a_norm_follows(self):
        from collections import Counter

        residual = [r for r in self.corpus.payloads() if r["task"] == "olmoe_residual_rmsnorm"]
        for order in (101, 303):
            with self.subTest(order=order):
                roles = Counter(r["call"]["role"] for r in residual if r["batch"]["order"] == order)
                self.assertEqual(roles, {"within_layer": LAYERS, "cross_layer": LAYERS - 1,
                                         "final_model_norm": 1})
        final = [r for r in residual if r["call"]["role"] == "final_model_norm"]
        self.assertTrue(all(r["call"]["norm"] == "model.norm" for r in final))
        self.assertTrue(all(r["notes"]["absent_cotangents"] == ["dsummed"] for r in final))
        for record in final:
            self.assertFalse(bool(self.corpus.load(record, device="cpu")["dsummed"].any()))

    def test_dsummed_is_the_external_gradient_only(self):
        """A sum's total gradient is its external path (``dsummed``) plus its own norm's
        VJP. Each residual add's sum is the next site's input ``residual``: the total
        gradient a within-layer site delivers into its sum is what the cross-layer (or
        final) site that consumes it records as ``dsummed`` plus that site's norm VJP --
        bit for bit, as autograd added the two in the model."""
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.norm import olmoe_rms_norm

        def total(values):
            summed = values["residual"] + values["branch"]
            summed.requires_grad_(True)
            (vjp,) = torch.autograd.grad(olmoe_rms_norm(summed, values["weight"], values["eps"]),
                                         summed, values["dnormed"])
            return values["dsummed"] + vjp

        payloads = {(r["batch"]["order"], r["call"]["path"]): r for r in self.corpus.payloads()
                    if r["task"] == "olmoe_residual_rmsnorm"}
        checked = 0
        for (order, path), record in payloads.items():
            layer = record["call"]["layer"]
            if record["call"]["role"] != "within_layer":
                continue
            nxt = payloads[(order, f"model.layers.{layer}#residual_mlp_to_"
                                  + ("next" if layer + 1 < LAYERS else "final"))]
            within = self.corpus.load(record, device="cpu")
            consumer = self.corpus.load(nxt, device="cpu")
            self.assertTrue(torch.equal(within["dsummed"], total(consumer)), path)
            checked += 1
        self.assertEqual(checked, 2 * LAYERS)

    def test_model_reference_reproduces_every_live_output(self):
        from evograd.opdecl.inputs import as_output_tuple
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.tensors import tensor_digest
        from evograd.verification.integrations.olmoe_1b_7b.reference import reference_op

        for record in self.corpus.payloads():
            op = reference_op(record["task"])
            values = self.corpus.load(record, device="cpu")
            with torch.no_grad():
                out = resolve_runtime_forward(op)(*[values.get(a.name, getattr(a, "default", None))
                                                    for a in op.args])
            outs = dict(zip(op.output_names, as_output_tuple(op, out)))
            for name, live in record["live_outputs"].items():
                self.assertEqual(tensor_digest(outs[name]), live["digest"],
                                 f"{record['payload_id']}:{name}")


if __name__ == "__main__":
    unittest.main()
