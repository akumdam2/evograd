"""The block-scope executor and gate, on blocks that are not Qwen.

What these pin is the architecture-neutral half: paths, not positions; every
declared output gets its cotangent and every declared input its gradient; a
reset before every repetition including warmup, outside the timer; input
mutation is seen; the gate rejects a wrong backward, a missing gradient and a
missing policy entry each by name; and the shared modules import no model.
"""

from __future__ import annotations

import ast
import json
import pathlib
import unittest
from dataclasses import dataclass

import torch
from torch import nn

from evograd.evaluation.tier3 import block as B
from evograd.evaluation.tier3.gate import block as G
from evograd.evaluation.tier3.gate import numerics
from evograd.evaluation.tier3.patch import KernelSet, KernelSource, Site, SiteRegistry, patch

EVOGRAD = pathlib.Path(__file__).resolve().parents[1] / "src" / "evograd"
SHARED_BLOCK_FILES = (
    EVOGRAD / "evaluation" / "tier3" / "block.py",
    EVOGRAD / "evaluation" / "tier3" / "block_cli.py",
    EVOGRAD / "evaluation" / "tier3" / "gate" / "block.py",
)


# ── a toy block with two differentiable outputs and one site ────────────────


def scale_default(tensor, weight):
    return tensor * weight


class Counter:
    def __init__(self):
        self.counts = {}

    def hit(self, site):
        self.counts[site] = self.counts.get(site, 0) + 1

    def snapshot(self):
        return dict(self.counts)

    def reset(self):
        self.counts.clear()


class TwoOutputBlock(nn.Module):
    """``y = scale(linear(x), w2)``, ``aux = y.mean(-1)``: a tuple return."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.w2 = nn.Parameter(torch.ones(width))
        self.kernel = scale_default
        self.counter = Counter()

    def forward(self, x, *, flag=False):
        self.counter.hit("scale")
        y = self.kernel(self.linear(x), self.w2)
        return y, y.mean(-1), flag


TOY_SITES = SiteRegistry(name="toy", sites=(Site("scale", None, scale_default),))


@dataclass
class ToyAdapter:
    width: int = 8
    batch: int = 3
    seed: int = 0
    registry: SiteRegistry = TOY_SITES
    observed_suite: str | None = None
    mutate: bool = False

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        self.x = torch.randn(self.batch, self.width, generator=g)
        self.cot_y = torch.randn(self.batch, self.width, generator=g)
        self.cot_aux = torch.randn(self.batch, generator=g)
        self.case = B.BlockCase(
            architecture="toy", architecture_revision="test", block_kind="two_output",
            block_index=0, source_mode="config", source={"seed": self.seed},
            dims={"B": self.batch, "T": 1, "W": self.width}, dtype="float32",
            boundary="x -> (y, aux, flag)", sites={"scale": "none"},
            inputs_hash=B.tensor_tree_hash(self.x), cotangents_hash=B.tensor_tree_hash(
                (self.cot_y, self.cot_aux)),
        )

    def build(self, *, device):
        torch.manual_seed(self.seed)
        module = TwoOutputBlock(self.width).to(device)
        return B.BuiltBlock(module=module, parameters=dict(module.named_parameters()),
                            buffers=dict(module.named_buffers()))

    def prepare(self, built, *, device):
        return B.BlockInvocation(
            args=(self.x.to(device),), kwargs={"flag": True},
            differentiable_inputs=("args[0]",), outputs=("result[0]", "result[1]"),
            cotangents=(self.cot_y.to(device), self.cot_aux.to(device)),
            metadata_outputs=("result[2]",), source_mode="config",
        )

    def install(self, built, kernels):
        if "scale" in kernels.patched:
            built.module.kernel = kernels.kernel_for("scale")
        paths = {"scale": ("kernel",)} if kernels.patched else {}
        from evograd.evaluation.tier3.patch import PatchProvenance
        return B.Installed(PatchProvenance("module_surgery", tuple(kernels.patched),
                                           tuple(paths), paths), built.module.counter)

    def expected_invocations(self, kernels):
        # The toy block counts its one site on every call, patched or native.
        return {"scale": 1}

    def reset(self, built):
        return None

    def local_checks(self, built, kernels, invocation):
        return None

    def capture_reference(self):
        return None

    def controls(self, kernels, ops, names):
        out = {}
        for name in names:
            if name == "structural_identity":
                out[name] = patch(KernelSet(registry=self.registry), "scale", scale_default,
                                  source=KernelSource("scale", None, None, "structural_identity"))
        return out


def wrong_backward_kernels(registry, factor=1.5):
    from evograd.evaluation.tier3.gate.faults import scale_grad

    def kernel(tensor, weight):
        return scale_grad(tensor * weight, factor)

    return patch(KernelSet(registry=registry), "scale", kernel,
                 source=KernelSource("scale", None, None, f"fault:grad_scale@{factor}"))


# ── paths and invocations ────────────────────────────────────────────────────


class TestPaths(unittest.TestCase):
    def test_paths_parse_and_resolve(self):
        roots = {"args": (1, (2, 3)), "kwargs": {"a": {"b": [4, 5]}}, "result": (6,)}
        self.assertEqual(B.resolve_path(roots, "args[1][0]"), 2)
        self.assertEqual(B.resolve_path(roots, "kwargs.a.b[1]"), 5)
        self.assertEqual(B.resolve_path(roots, "result[0]"), 6)
        with self.assertRaises(B.InvocationError):
            B.resolve_path(roots, "result[3]")
        with self.assertRaises(B.InvocationError):
            B.parse_path("args[x]")

    def test_every_output_needs_exactly_one_cotangent(self):
        x = torch.randn(2, 3)
        with self.assertRaises(B.InvocationError):
            B.BlockInvocation(args=(x,), kwargs={}, differentiable_inputs=("args[0]",),
                              outputs=("result[0]", "result[1]"), cotangents=(x,))

    def test_a_differentiable_input_must_exist_in_the_call(self):
        x = torch.randn(2, 3)
        with self.assertRaises(B.InvocationError):
            B.BlockInvocation(args=(x,), kwargs={}, differentiable_inputs=("kwargs.y",),
                              outputs=("result",), cotangents=(x,))

    def test_a_tuple_return_is_never_read_as_its_first_element(self):
        x = torch.randn(2, 3)
        inv = B.BlockInvocation(args=(x,), kwargs={}, differentiable_inputs=("args[0]",),
                                outputs=("result",), cotangents=(x,))
        with self.assertRaises(B.InvocationError):
            inv.select_outputs((x * 2, x * 3))   # "result" is not a tensor here
        inv2 = B.BlockInvocation(args=(x,), kwargs={}, differentiable_inputs=("args[0]",),
                                 outputs=("result[0]",), cotangents=(x,))
        with self.assertRaises(B.InvocationError) as caught:
            inv2.select_outputs((x * 2, x * 3))   # result[1] is a tensor nobody declared
        self.assertIn("result[1]", str(caught.exception))

    def test_fresh_makes_new_leaves_and_keeps_layout(self):
        x = torch.randn(4, 6)[:, ::2]           # non-contiguous view
        inv = B.BlockInvocation(args=(x,), kwargs={}, differentiable_inputs=("args[0]",),
                                outputs=("result",), cotangents=(x,))
        fresh = inv.fresh()
        leaf = fresh.leaves()["args[0]"]
        self.assertTrue(leaf.requires_grad and leaf.is_leaf)
        self.assertIsNot(leaf, x)
        self.assertEqual(tuple(leaf.shape), tuple(x.shape))
        self.assertTrue(torch.equal(leaf.detach(), x))


# ── execution ────────────────────────────────────────────────────────────────


class TestMultiOutputVjp(unittest.TestCase):
    def test_both_outputs_receive_their_cotangents_and_metadata_is_kept(self):
        adapter = ToyAdapter()
        built = adapter.build(device="cpu")
        inv = adapter.prepare(built, device="cpu")
        result = B.run_vjp(built.module, inv, built.parameters, reset=lambda: None)
        self.assertEqual(set(result.outputs), {"result[0]", "result[1]"})
        self.assertEqual(result.metadata, {"result[2]": True})
        self.assertIsNotNone(result.input_grads["args[0]"])
        self.assertTrue(all(g is not None for g in result.param_grads.values()))
        # The manual VJP: grads from both cotangents, not from y alone.
        x = adapter.x.clone().requires_grad_(True)
        torch.manual_seed(0)
        ref = TwoOutputBlock(adapter.width)
        y, aux, _ = ref(x)
        torch.autograd.backward((y, aux), (adapter.cot_y, adapter.cot_aux))
        self.assertTrue(torch.allclose(result.input_grads["args[0]"], x.grad))
        only_y = torch.autograd.grad(ref(adapter.x.clone().requires_grad_(True))[0],
                                     [ref.w2], adapter.cot_y)[0]
        self.assertFalse(torch.allclose(result.param_grads["w2"], only_y))

    def test_a_block_returning_an_undeclared_tensor_is_refused(self):
        adapter = ToyAdapter()
        built = adapter.build(device="cpu")
        inv = adapter.prepare(built, device="cpu")
        inv = B.BlockInvocation(args=inv.args, kwargs=inv.kwargs, differentiable_inputs=inv.differentiable_inputs,
                                outputs=("result[0]",), cotangents=(inv.cotangents[0],),
                                metadata_outputs=("result[2]",))
        with self.assertRaises(B.InvocationError):
            B.run_vjp(built.module, inv, built.parameters, reset=lambda: None)

    def test_input_mutation_is_reported(self):
        class Mutating(nn.Module):
            def forward(self, x, *, table):
                table.add_(1.0)          # writes into a non-differentiable input
                return (x * table).sum(-1)
        x, table = torch.randn(3, 4), torch.ones(3, 4)
        inv = B.BlockInvocation(args=(x,), kwargs={"table": table}, differentiable_inputs=("args[0]",),
                                outputs=("result",), cotangents=(torch.ones(3),))
        result = B.run_vjp(Mutating(), inv, {}, reset=lambda: None)
        self.assertEqual(result.mutated_inputs, ["kwargs.table"])


class TestResetBeforeEveryRepetition(unittest.TestCase):
    def test_warmup_and_every_sample_start_from_reset_state(self):
        adapter = ToyAdapter()
        built = adapter.build(device="cpu")
        inv = adapter.prepare(built, device="cpu")
        seen_dirty = []
        calls = {"reset": 0, "step": 0}
        original = built.module.forward

        def spy(x, **kw):
            calls["step"] += 1
            if any(p.grad is not None for p in built.parameters.values()):
                seen_dirty.append(calls["step"])
            return original(x, **kw)

        built.module.forward = spy

        def reset():
            calls["reset"] += 1
            built.module.zero_grad(set_to_none=True)

        timing = B.time_vjp(built.module, inv, reset=reset, device="cpu",
                            warmup=2, samples=3, blocks=2)
        self.assertEqual(calls["step"], 2 + 3 * 2)
        self.assertEqual(calls["reset"], calls["step"])
        self.assertEqual(seen_dirty, [])
        self.assertEqual(len(timing["per_block_ms"]), 2)
        self.assertEqual(len(timing["per_sample_ms"][0]), 3)
        self.assertIn("before every sample including warmup", timing["reset"])

    def test_saved_state_excludes_parameters_and_counts_unique_storages(self):
        adapter = ToyAdapter()
        built = adapter.build(device="cpu")
        inv = adapter.prepare(built, device="cpu")
        saved = B.saved_state_probe(built.module, inv, built.parameters, built.buffers,
                                    reset=lambda: built.module.zero_grad(set_to_none=True))
        self.assertGreater(saved["saved_state_bytes"], 0)
        self.assertGreaterEqual(saved["saved_tensors_logical"], saved["saved_unique_storages"])
        param_bytes = sum(p.numel() * p.element_size() for p in built.parameters.values())
        # x (3x8 floats) and the linear output are saved; the weights are not counted.
        self.assertLess(saved["saved_state_bytes"], param_bytes + 4 * adapter.batch * adapter.width * 4)


# ── the gate ─────────────────────────────────────────────────────────────────


def _result(adapter, kernels=None):
    built = adapter.build(device="cpu")
    if kernels is not None:
        adapter.install(built, kernels)
    inv = adapter.prepare(built, device="cpu")
    return B.run_vjp(built.module, inv, built.parameters, reset=lambda: None), built, inv


class TestGate(unittest.TestCase):
    def setUp(self):
        self.adapter = ToyAdapter()
        self.policy = G.calibrate_policy(
            self.adapter, {"scale": wrong_backward_kernels(TOY_SITES)}, ops={}, device="cpu",
            controls=("structural_identity",), noise_repeats=2)
        self.reference, _, _ = _result(self.adapter)
        self.reference = self.reference.cpu()

    def test_the_policy_is_derived_from_controls_only_and_is_hashed(self):
        entry = self.policy["entries"]["scale"]
        self.assertEqual(sorted(entry["controls"]), ["native_repeatability", "structural_identity"])
        self.assertTrue(entry["controls"]["structural_identity"]["bitwise"])
        self.assertTrue(entry["controls_ok"])
        self.assertEqual(self.policy["policy_hash"], G.policy_hash(self.policy))
        self.assertEqual(self.policy["case_hash"], self.adapter.case.case_hash)
        for group, envelope in entry["envelopes"].items():
            kind = group.split("|")[0]
            floor = numerics.KIND_THRESHOLD_FLOOR[kind]["rel_l2"] * self.policy["margin"]
            self.assertGreaterEqual(envelope["threshold"]["rel_l2"], floor)

    def test_a_correct_provider_passes(self):
        kernels = patch(KernelSet(registry=TOY_SITES), "scale", scale_default,
                        source=KernelSource("scale", None, None, "structural_identity"))
        candidate, built, inv = _result(self.adapter, kernels)
        verdict = G.check_candidate(self.reference, candidate, invocation=inv, parameters=built.parameters,
                                    kernels=kernels, policy=self.policy)
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["comparison"]["bitwise"])

    def test_a_wrong_backward_is_rejected_by_name_before_timing(self):
        kernels = wrong_backward_kernels(TOY_SITES)
        candidate, built, inv = _result(self.adapter, kernels)
        verdict = G.check_candidate(self.reference, candidate, invocation=inv, parameters=built.parameters,
                                    kernels=kernels, policy=self.policy)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "envelope")
        self.assertTrue(all(v["bitwise"] for k, v in verdict["per_tensor"].items() if k.startswith("out:")))
        # And through the orchestrator: nothing is timed.
        entry = B.measure_block_one(self.adapter, "wrong", kernels, role="candidate", device="cpu",
                                    ops={}, policy=self.policy, reference=self.reference,
                                    verify=False, purity=False, warmup=0, samples=1, blocks=1)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "block_correctness")
        self.assertIsNone(entry["latency"])
        self.assertIsNone(entry["memory"])

    def test_large_backward_errors_are_rejected_by_block_gate(self):
        # Test the block gate itself, not the earlier operator preflight.
        for factor in (1.5, 2.0, 10.0, 100.0):
            with self.subTest(factor=factor):
                kernels = wrong_backward_kernels(TOY_SITES, factor=factor)
                entry = B.measure_block_one(
                    self.adapter, "wrong", kernels, role="candidate", device="cpu",
                    ops={}, policy=self.policy, reference=self.reference,
                    verify=False, purity=False, warmup=0, samples=1, blocks=1)
                self.assertFalse(entry["ok"], entry)
                self.assertEqual(entry["failed_at"], "block_correctness")
                self.assertIsNone(entry["latency"])

    def test_broken_structural_controls_cannot_calibrate_their_own_errors(self):
        for factor in (1.5, 2.0, 10.0, 100.0):
            with self.subTest(factor=factor):
                class BrokenControl(ToyAdapter):
                    def controls(self, kernels, ops, names):
                        return {"structural_identity": wrong_backward_kernels(self.registry, factor)}

                with self.assertRaisesRegex(B.BlockError, "control structural_identity failed"):
                    G.calibrate_policy(BrokenControl(), {"scale": wrong_backward_kernels(TOY_SITES)},
                                       ops={}, device="cpu", controls=("structural_identity",),
                                       noise_repeats=2)

    def test_invalid_or_old_policy_is_refused_even_if_candidate_is_correct(self):
        import copy

        kernels = patch(KernelSet(registry=TOY_SITES), "scale", scale_default,
                        source=KernelSource("scale", None, None, "structural_identity"))
        candidate, built, inv = _result(self.adapter, kernels)
        for flaw in ("controls_flag", "failed_control", "missing_control", "old_schema", "nan_threshold"):
            with self.subTest(flaw=flaw):
                policy = copy.deepcopy(self.policy)
                entry = policy["entries"]["scale"]
                if flaw == "controls_flag":
                    entry["controls_ok"] = False
                elif flaw == "failed_control":
                    entry["controls"]["structural_identity"]["ok"] = False
                elif flaw == "missing_control":
                    del entry["controls"]["structural_identity"]
                elif flaw == "old_schema":
                    policy["schema"] = "evograd-t3-block-policy/1"
                else:
                    next(iter(entry["envelopes"].values()))["threshold"]["rel_l2"] = float("nan")
                policy["policy_hash"] = G.policy_hash(policy)
                self.assertTrue(G.check_policy_binding(policy, self.adapter, None))
                verdict = G.check_candidate(self.reference, candidate, invocation=inv,
                                            parameters=built.parameters, kernels=kernels, policy=policy)
                self.assertFalse(verdict["ok"])
                self.assertEqual(verdict["failed_at"], "invalid_policy")

    def test_missing_requested_control_is_not_silently_ignored(self):
        with self.assertRaisesRegex(B.BlockError, "requested controls"):
            G.calibrate_policy(self.adapter, {"scale": wrong_backward_kernels(TOY_SITES)},
                               ops={}, device="cpu",
                               controls=("structural_identity", "trusted_torch_compile"), noise_repeats=2)

    def test_small_nonbitwise_backward_is_checked_against_native_noise(self):
        class RoundedControl(ToyAdapter):
            def controls(self, kernels, ops, names):
                return {"structural_identity": wrong_backward_kernels(self.registry, 1.000001)}

        policy = G.calibrate_policy(RoundedControl(), {"scale": wrong_backward_kernels(TOY_SITES)},
                                    ops={}, device="cpu", controls=("structural_identity",),
                                    noise_repeats=2)
        control = policy["entries"]["scale"]["controls"]["structural_identity"]
        self.assertFalse(control["bitwise"])
        self.assertTrue(control["forward_bitwise"])
        self.assertTrue(control["backward_native_envelope"]["ok"])
        self.assertTrue(policy["entries"]["scale"]["controls_ok"])

    def test_nonfinite_control_is_rejected(self):
        class NonfiniteControl(ToyAdapter):
            def controls(self, kernels, ops, names):
                return {"structural_identity": wrong_backward_kernels(self.registry, float("nan"))}

        with self.assertRaisesRegex(B.BlockError, "control structural_identity failed"):
            G.calibrate_policy(NonfiniteControl(), {"scale": wrong_backward_kernels(TOY_SITES)},
                               ops={}, device="cpu", controls=("structural_identity",), noise_repeats=2)

    def test_native_numerical_instability_is_rejected_before_calibration(self):
        from unittest.mock import patch as mock_patch

        original = B.run_vjp
        calls = {"n": 0}

        def unstable(*args, **kwargs):
            result = original(*args, **kwargs)
            calls["n"] += 1
            if calls["n"] > 1:
                result.param_grads = {k: g * 10 if g is not None else None
                                      for k, g in result.param_grads.items()}
            return result

        with mock_patch.object(B, "run_vjp", side_effect=unstable):
            with self.assertRaisesRegex(B.BlockError, "native calibration failed"):
                G.calibrate_policy(self.adapter, {"scale": wrong_backward_kernels(TOY_SITES)},
                                   ops={}, device="cpu", controls=("structural_identity",), noise_repeats=2)

    def test_a_missing_parameter_gradient_fails_presence(self):
        kernels = patch(KernelSet(registry=TOY_SITES), "scale", lambda t, w: t * w.detach(),
                        source=KernelSource("scale", None, None, "candidate"))
        candidate, built, inv = _result(self.adapter, kernels)
        verdict = G.check_candidate(self.reference, candidate, invocation=inv, parameters=built.parameters,
                                    kernels=kernels, policy=self.policy)
        self.assertEqual(verdict["failed_at"], "gradient_presence")
        self.assertEqual(verdict["gradient_presence"]["missing"], ["w2"])

    def test_a_provider_without_a_policy_entry_is_reported_not_judged_by_a_neighbour(self):
        kernels = wrong_backward_kernels(TOY_SITES)
        candidate, built, inv = _result(self.adapter, kernels)
        verdict = G.check_candidate(self.reference, candidate, invocation=inv, parameters=built.parameters,
                                    kernels=kernels, policy={"entries": {}, "policy_hash": "x"})
        self.assertEqual(verdict["failed_at"], "no_policy")
        entry = B.measure_block_one(self.adapter, "wrong", kernels, role="candidate", device="cpu",
                                    ops={}, policy={"entries": {}}, reference=self.reference,
                                    verify=False, purity=False, warmup=0, samples=1, blocks=1)
        self.assertEqual(entry["failed_at"], "no_policy")

    def test_a_frozen_policy_binds_to_its_case(self):
        other = ToyAdapter(seed=1)
        problems = G.check_policy_binding(self.policy, other, None)
        self.assertTrue(any("case" in p for p in problems))
        self.assertEqual(G.check_policy_binding(self.policy, self.adapter, None), [])

    def test_the_reference_verdict_checks_repeatability_finiteness_and_presence(self):
        built = self.adapter.build(device="cpu")
        inv = self.adapter.prepare(built, device="cpu")
        run = lambda: B.run_vjp(built.module, inv, built.parameters, reset=lambda: None)  # noqa: E731
        verdict = G.check_reference(self.adapter, built, inv, run(), reset=lambda: None,
                                    noise_repeats=2, capture=None, run=run)
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["repeatability"]["bitwise"])
        self.assertTrue(verdict["capture_agreement"]["skipped"])


class TestReportRoundTrip(unittest.TestCase):
    def test_a_real_entry_set_serializes_and_reads_as_a_block_report(self):
        from evograd.evaluation.tier3.report import identify_report, provider_rows

        adapter = ToyAdapter()
        policy = G.calibrate_policy(adapter, {"scale": wrong_backward_kernels(TOY_SITES)}, ops={},
                                    device="cpu", controls=("structural_identity",), noise_repeats=1)
        reference, _, _ = _result(adapter)
        native = KernelSet(registry=TOY_SITES)
        results = {
            "native": B.measure_block_one(adapter, "native", native, role="reference", device="cpu",
                                          ops={}, policy=policy, reference=None, verify=False,
                                          purity=False, noise_repeats=1, warmup=1, samples=2, blocks=2),
            "wrong": B.measure_block_one(adapter, "wrong", wrong_backward_kernels(TOY_SITES),
                                         role="candidate", device="cpu", ops={}, policy=policy,
                                         reference=reference.cpu(), verify=False, purity=False,
                                         warmup=1, samples=2, blocks=2),
        }
        report = B.assemble_block_report(adapter, results, ["wrong", "native"], policy=policy, ops={},
                                         seed=0, isolation="test", warmup=1, samples=2, blocks=2,
                                         verify=False, purity=False)
        text = json.dumps(report, sort_keys=True)            # must be JSON-clean
        parsed = json.loads(text)
        identity = identify_report(parsed)
        self.assertEqual(identity.execution_scope, "block")
        self.assertEqual(identity.benchmark_level, 3)
        self.assertEqual(identity.case, adapter.case.case_id)
        rows = {r.provider: r for r in provider_rows(parsed)}
        self.assertTrue(rows["native"].ok)
        self.assertIsNotNone(rows["native"].saved_state_bytes)
        self.assertIsNone(rows["native"].execution_peak_bytes)      # CPU: not measured, not 0
        self.assertFalse(rows["wrong"].ok)
        self.assertEqual(rows["wrong"].failed_at, "block_correctness")
        self.assertEqual(parsed["timing_protocol"]["boundary"], "forward + vjp")
        self.assertEqual(parsed["policy"]["policy_hash"], policy["policy_hash"])


# ── layering ─────────────────────────────────────────────────────────────────


class TestSharedBlockCodeNamesNoModel(unittest.TestCase):
    def _imports(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
        return names

    def test_no_workload_package_is_imported(self):
        from evograd.evaluation.tier3.workloads import TIER3_ADAPTERS

        markers = {target.split(":")[0].rsplit(".", 1)[0] for target in TIER3_ADAPTERS.values()}
        for path in SHARED_BLOCK_FILES:
            with self.subTest(file=path.name):
                for name in self._imports(path):
                    self.assertFalse(any(name.startswith(m) for m in markers), f"{path.name} imports {name}")

    def test_no_model_name_site_name_or_count_in_code(self):
        for path in SHARED_BLOCK_FILES:
            code_lines = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            body = "\n".join(code_lines)
            # Strip the module docstring's example command before the scan.
            body = body.split('"""', 2)[-1] if body.count('"""') >= 2 else body
            for marker in ("qkv_norm_rope", "swiglu_mlp", "model.layers", "Qwen3", "Llama",
                           "num_hidden_layers", "= 28", "= 56"):
                with self.subTest(file=path.name, marker=marker):
                    self.assertNotIn(marker, body)


if __name__ == "__main__":
    unittest.main()
