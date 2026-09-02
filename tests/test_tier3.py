"""Tier 3 wiring: what gets patched, and what must not change to allow it.

No GPU. What these pin is the part that decides *what is measured* — which
sites map to which declared operators, and whether making the reference layer
patchable altered the layer the oracle differentiates.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from evograd.bench.tier3 import (
    SITE_OPS,
    KernelSet,
    ModulePatch,
    ModuleWorkload,
    eager_pair_for,
    identity_control_kernels,
    loss_agreement,
    patch,
    patch_modules,
    restrict,
)
from evograd.opdecl.models import LLAMA_3_8B, LLAMA_3_8B_4L
from evograd.ops import OPS, get_op
from evograd.ops.level3.llama3_decoder_layer import forward_ref as reference
from torch import nn


class TestPatchSites(unittest.TestCase):
    def test_every_site_names_a_declared_operator(self):
        # A site whose operator does not exist would accept a candidate and then
        # fail at bind time, inside a run.
        for site, op_name in SITE_OPS.items():
            with self.subTest(site=site):
                self.assertIn(op_name, OPS)

    def test_each_site_operator_declares_its_parameter_split(self):
        # bind() and the module wrapping both need it; an undeclared split would
        # surface only once a candidate was patched in.
        for op_name in SITE_OPS.values():
            with self.subTest(op=op_name):
                self.assertIsNotNone(get_op(op_name).parameter_args)

    def test_an_unknown_site_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(), "attention", lambda *a: None)
        self.assertIn("attention", str(caught.exception))

    def test_patching_records_what_was_replaced(self):
        # The report has to say which sites a provider actually replaced;
        # "candidate was 1.2x" means nothing without it.
        kernels = patch(KernelSet(), "swiglu", lambda g, u: None)
        self.assertEqual(kernels.patched, ("swiglu",))
        kernels = patch(kernels, "rms_norm", lambda *a: None)
        self.assertEqual(kernels.patched, ("rms_norm", "swiglu"))

    def test_an_unpatched_set_says_so(self):
        self.assertEqual(KernelSet().patched, ())

    def test_the_default_kernels_are_the_declared_spellings(self):
        # The eager provider must be the production spelling, not the primitive
        # one: timing against the unfused reference inflates every ratio.
        self.assertIs(KernelSet().rms_norm, reference._rms_norm_fused)
        self.assertIs(KernelSet().swiglu, reference._default_swiglu)


class TestTheReferenceStillDescribesTheSameLayer(unittest.TestCase):
    """Making the activation injectable must not change what the oracle sees."""

    def test_the_declaration_still_calls_the_reference_positionally(self):
        op = get_op("llama3_decoder_layer")
        tree = ast.parse(
            Path(reference.__file__).read_text()
        )
        params = next(
            [a.arg for a in node.args.args]
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_llama3_decoder_layer"
        )
        # rms_norm is injected ahead of the declared args; swiglu is appended
        # behind a default, so a positional call with the declared args alone
        # still lands exactly as before.
        self.assertEqual(params[0], "rms_norm")
        self.assertEqual(params[1 : 1 + len(op.args)], [a.name for a in op.args])
        self.assertEqual(params[-1], "swiglu")

    def test_swiglu_has_a_default_so_existing_callers_are_unaffected(self):
        tree = ast.parse(Path(reference.__file__).read_text())
        node = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_llama3_decoder_layer"
        )
        defaulted = [a.arg for a in node.args.args[-len(node.args.defaults):]]
        self.assertIn("swiglu", defaulted)

    def test_the_default_activation_is_the_original_expression(self):
        # float32 SiLU, cast back: the activation is where a bfloat16 MLP loses
        # the most, and silently dropping the promotion would change numerics
        # without changing any test that only checks shapes.
        source = ast.unparse(
            next(
                n for n in ast.parse(Path(reference.__file__).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_default_swiglu"
            )
        )
        self.assertIn("float()", source)
        self.assertIn("silu", source)


class TestIterationConfig(unittest.TestCase):
    def test_the_small_config_differs_only_in_layer_count(self):
        # Layer count is the one dimension that can be cut without changing what
        # a kernel does to the answer. Anything else would make the iteration
        # numbers describe a different model than the reported ones.
        for field in ("hidden", "intermediate", "n_heads", "n_kv_heads",
                      "head_dim", "vocab", "rope_theta", "dtype"):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(LLAMA_3_8B_4L, field), getattr(LLAMA_3_8B, field)
                )
        self.assertLess(LLAMA_3_8B_4L.layers, LLAMA_3_8B.layers)

    def test_the_loss_head_shape_is_unchanged(self):
        # The [tokens, vocab] logits tensor is the memory story, and it does not
        # depend on layer count -- which is why the small config can show it.
        self.assertEqual(LLAMA_3_8B_4L.vocab, 128256)


class TestRankAdapter(unittest.TestCase):
    """A model carries [batch, tokens, hidden]; every declaration is written for rows.

    Eager PyTorch hides the difference by broadcasting over leading dimensions,
    so an unpatched model runs and a patched one dies inside someone else's
    kernel. The adapter flattens to the declared rank and restores afterwards,
    which is what HuggingFace and Liger both do at these sites.
    """

    def test_declared_rank_reads_the_shape_string(self):
        from evograd.bench.tier3_patch import _declared_rank

        self.assertEqual(_declared_rank("[rows, hidden]"), 2)
        self.assertEqual(_declared_rank("[rows]"), 1)
        self.assertEqual(_declared_rank("[]"), 0)
        self.assertIsNone(_declared_rank(None))

    def test_the_three_patch_sites_are_all_row_shaped(self):
        # If a declaration ever became 3D the adapter would be a no-op for it,
        # which is correct but worth noticing rather than assuming.
        from evograd.bench.tier3_patch import _declared_rank

        self.assertEqual(_declared_rank(get_op("rmsnorm").args[0].shape), 2)
        self.assertEqual(_declared_rank(get_op("swiglu").args[0].shape), 2)
        flce = get_op("fused_linear_cross_entropy")
        self.assertEqual(_declared_rank(flce.args[0].shape), 2)   # x
        self.assertEqual(_declared_rank(flce.args[2].shape), 1)   # target

    def test_a_scalar_output_is_never_unflattened(self):
        # fused_linear_cross_entropy returns a scalar loss. Restoring leading
        # dimensions onto it would be nonsense, and indexing shape[0] would
        # raise -- so rank 0 has to skip the restore rather than attempt it.
        from evograd.bench.tier3_patch import _declared_rank

        self.assertEqual(
            _declared_rank(get_op("fused_linear_cross_entropy").output.shape), 0
        )

    def test_row_shaped_outputs_are_restored(self):
        from evograd.bench.tier3_patch import _declared_rank

        for name in ("rmsnorm", "swiglu"):
            with self.subTest(op=name):
                self.assertEqual(_declared_rank(get_op(name).output.shape), 2)


class TestLossAgreement(unittest.TestCase):
    def _report(self, **providers):
        return {"providers": providers}

    def test_it_reports_the_largest_divergence_from_eager(self):
        report = self._report(
            eager={"ok": True, "losses": [1.0, 0.9, 0.8]},
            candidate={"ok": True, "losses": [1.0, 0.9, 0.5]},
        )
        self.assertAlmostEqual(
            loss_agreement(report)["max_abs_delta"]["candidate"], 0.3
        )

    def test_a_failed_reference_is_reported_not_guessed(self):
        report = self._report(eager={"ok": False, "error": "boom"})
        self.assertFalse(loss_agreement(report)["available"])

    def test_a_failed_provider_is_skipped_rather_than_scored(self):
        report = self._report(
            eager={"ok": True, "losses": [1.0]},
            candidate={"ok": False, "error": "boom"},
        )
        self.assertNotIn("candidate", loss_agreement(report)["max_abs_delta"])


class TestTheHarnessIsModelAgnostic(unittest.TestCase):
    """`bench.tier3` must not know what model it is measuring.

    The tier is a measurement protocol, not a Llama benchmark. If the harness
    imports the Llama workload, every future model has to be bent into that
    shape or the file has to grow a branch per model.
    """

    def test_the_harness_does_not_import_the_llama_workload(self):
        # Checked against the import statements, not the text: the module
        # docstring names the Llama workload as an example, which is fine.
        import evograd.bench.tier3 as harness

        tree = ast.parse(Path(harness.__file__).read_text())
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertFalse(
            [m for m in imported if "tier3_llama" in m],
            f"harness imports the Llama workload: {sorted(imported)}",
        )

    def test_the_llama_workload_satisfies_the_protocol(self):
        from evograd.bench.tier3_llama import LlamaWorkload

        for method in ("units_per_step", "build", "batch_for", "loss", "describe"):
            with self.subTest(method=method):
                self.assertTrue(callable(getattr(LlamaWorkload, method, None)))
        self.assertEqual(LlamaWorkload.unit_name, "tokens")

    def test_a_module_workload_satisfies_it_too(self):
        workload = ModuleWorkload(
            name="toy", factory=lambda: nn.Module(),
            make_batch=lambda seed: seed, compute_loss=lambda m, b: b, units=7,
        )
        self.assertEqual(workload.units_per_step(), 7)
        self.assertEqual(workload.batch_for(seed=3), 3)
        self.assertEqual(workload.describe()["workload"], "module")


class TestModuleSurgery(unittest.TestCase):
    """The route for a model you did not write."""

    def _tree(self):
        class Leaf(nn.Module):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag

        parent = nn.Module()
        parent.a = Leaf("replace-me")
        parent.b = Leaf("leave-me")
        return parent, Leaf

    def test_only_patched_sites_are_touched(self):
        # An eager KernelSet must leave the model exactly as the factory built
        # it, so the unpatched provider is the original model rather than a
        # rebuilt lookalike.
        model, Leaf = self._tree()
        patches = (ModulePatch("rms_norm", lambda m: getattr(m, "tag", None) == "replace-me",
                               lambda original, kernel: Leaf("patched")),)
        replaced = patch_modules(model, patches, KernelSet())
        self.assertEqual(replaced, [])
        self.assertEqual(model.a.tag, "replace-me")

    def test_matching_submodules_are_replaced_and_reported(self):
        model, Leaf = self._tree()
        patches = (ModulePatch("rms_norm", lambda m: getattr(m, "tag", None) == "replace-me",
                               lambda original, kernel: Leaf("patched")),)
        kernels = patch(KernelSet(), "rms_norm", lambda *a: None)
        replaced = patch_modules(model, patches, kernels)
        self.assertEqual(replaced, ["a"])
        self.assertEqual(model.a.tag, "patched")
        self.assertEqual(model.b.tag, "leave-me")   # non-matching left alone

    def test_the_replacement_receives_the_original(self):
        # It has to carry the trained weights across; handing it only shapes
        # would produce a model that runs and has been silently reinitialized.
        model, Leaf = self._tree()
        seen = []
        patches = (ModulePatch("rms_norm", lambda m: getattr(m, "tag", None) == "replace-me",
                               lambda original, kernel: seen.append(original) or Leaf("p")),)
        patch_modules(model, patches, patch(KernelSet(), "rms_norm", lambda *a: None))
        self.assertEqual([m.tag for m in seen], ["replace-me"])


class TestIdentityControl(unittest.TestCase):
    """The tier-3 analogue of --identity-control: same math, all the plumbing.

    Tiers 1 and 2 compare two providers in symmetric slots, so their control is
    one provider timed against itself. Here the asymmetry is not the slot, it is
    that a patched model routes through bind, an autograd.Function and the rank
    adapter while an unpatched one does not. The control has to sit on both
    sides of that machinery.
    """

    def test_it_patches_every_site_by_default(self):
        kernels = identity_control_kernels(OPS)
        self.assertEqual(set(kernels.patched), set(SITE_OPS))

    def test_it_can_be_restricted_for_attribution(self):
        kernels = identity_control_kernels(OPS, ("rms_norm",))
        self.assertEqual(kernels.patched, ("rms_norm",))

    def test_the_control_is_not_the_eager_default(self):
        # If it were the same object the control would measure nothing: it has
        # to route through the patching machinery to price it.
        control = identity_control_kernels(OPS)
        self.assertIsNot(control.rms_norm, KernelSet().rms_norm)
        self.assertIsNot(control.swiglu, KernelSet().swiglu)

    def test_the_eager_pair_exposes_the_candidate_interface(self):
        # It is fed to kernel_from_pair, which calls lookup_pair on it.
        from evograd.opdecl.bind import lookup_pair

        for op_name in SITE_OPS.values():
            with self.subTest(op=op_name):
                forward, backward = lookup_pair(
                    get_op(op_name), eager_pair_for(get_op(op_name))
                )
                self.assertTrue(callable(forward) and callable(backward))


class TestSiteRestriction(unittest.TestCase):
    def test_restricting_keeps_only_the_named_sites(self):
        kernels = KernelSet()
        for site in SITE_OPS:
            kernels = patch(kernels, site, lambda *a: None)
        self.assertEqual(restrict(kernels, ("swiglu",)).patched, ("swiglu",))

    def test_reverted_sites_go_back_to_the_eager_default(self):
        kernels = patch(KernelSet(), "rms_norm", lambda *a: None)
        reverted = restrict(kernels, ("swiglu",))
        self.assertEqual(reverted.patched, ())
        self.assertIs(reverted.rms_norm, KernelSet().rms_norm)

    def test_an_unknown_site_is_rejected(self):
        with self.assertRaises(ValueError):
            restrict(KernelSet(), ("attention",))


class TestTheThreeParts(unittest.TestCase):
    """Tier 3 is three modules with a one-way dependency, plus a facade.

        tier3_model.py   what is measured — the workload protocol, bring-your-own
        tier3_patch.py   how a kernel gets in — sites, bind wrapping, surgery
        tier3_runner.py  how it is measured — build, step, time, report

    The direction matters more than the split: patch knows nothing about models
    or measurement, so it can be read and changed without either. A cycle here
    would mean the parts are not really separate.
    """

    def _tier3_imports(self, part):
        import importlib

        module = importlib.import_module(f"evograd.bench.{part}")
        tree = ast.parse(Path(module.__file__).read_text())
        return {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and "tier3" in node.module
        }

    def test_the_patcher_depends_on_neither_other_part(self):
        self.assertEqual(self._tier3_imports("tier3_patch"), set())

    def test_the_model_layer_depends_only_on_the_patcher(self):
        self.assertLessEqual(
            self._tier3_imports("tier3_model"), {"evograd.bench.tier3_patch"}
        )

    def test_the_runner_depends_on_both_and_nothing_else(self):
        self.assertLessEqual(
            self._tier3_imports("tier3_runner"),
            {"evograd.bench.tier3_patch", "evograd.bench.tier3_model"},
        )

    def test_an_architecture_depends_only_on_the_patcher(self):
        # A new model must not need the runner or the facade to exist.
        self.assertLessEqual(
            self._tier3_imports("tier3_llama"), {"evograd.bench.tier3_patch"}
        )

    def test_the_facade_re_exports_every_part(self):
        import evograd.bench.tier3 as facade

        for name in ("KernelSet", "ModulePatch", "patch_modules",      # patch
                     "TrainingWorkload", "ModuleWorkload",             # model
                     "run_tier3", "measure_provider", "loss_agreement"):  # runner
            with self.subTest(name=name):
                self.assertIn(name, facade.__all__)
                self.assertTrue(hasattr(facade, name))


class TestCli(unittest.TestCase):
    def test_site_equals_path_is_parsed(self):
        from evograd.bench.tier3_cli import _parser

        args = _parser().parse_args(
            ["--candidate", "rms_norm=a.py", "--candidate", "swiglu=b.py"]
        )
        self.assertEqual(args.candidate, ["rms_norm=a.py", "swiglu=b.py"])

    def test_the_iteration_config_is_the_default(self):
        from evograd.bench.tier3_cli import _parser

        self.assertEqual(_parser().parse_args([]).model, "llama_3_8b_4l")


if __name__ == "__main__":
    unittest.main()
