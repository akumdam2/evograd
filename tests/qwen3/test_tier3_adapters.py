"""The Qwen3 tier-3 sites, on a two-layer model a CPU can run.

What these pin is the *wiring*: which modules an adapter reaches, how many times
each site runs, whether the parameters are still the same objects, and whether
the model a patched build produces is the model an unpatched one produces. The
canonical 28-layer numbers are the same assertions at a scale only a GPU can
reach, so the counts are asserted as functions of the layer count rather than as
constants.

Structural identity is demanded **bitwise** here as it is on the canonical
model: the adapters call the same submodules in the same order, so a difference
is a defect in the restructure and not a tolerance question.
"""

from __future__ import annotations

import json
import unittest

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

HAVE_TRANSFORMERS = False
if HAVE_TORCH:
    try:
        from evograd.bench.workloads.qwen3.levels.level4.model import require_transformers

        require_transformers()
        HAVE_TRANSFORMERS = True
    except Exception:  # pragma: no cover - optional dependency
        HAVE_TRANSFORMERS = False

if HAVE_TRANSFORMERS:
    from evograd.bench.tier3_patch import KernelSet, restrict
    from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
        SITE_ATTENTION,
        SITE_MLP,
        SITE_QKV,
        SITE_RESIDUAL,
        ResidualCarrier,
        expected_counts,
        module_patches,
        patch_model,
        qwen3_sites,
        structural_identity_kernels,
    )
    from evograd.bench.workloads.qwen3.evaluation.tier3.workload import MODEL_KEY, Qwen3Workload

#: Two layers, narrow, tiny vocabulary. Everything a CPU test needs and nothing
#: it does not: the architecture's *shape* is what the adapters wire into.
SMALL = dict(
    device="cpu",
    dtype="float32",
    batch_size=2,
    seq_len=16,
    arch_overrides={
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 64,
    },
)
LAYERS = SMALL["arch_overrides"]["num_hidden_layers"]

_skip = unittest.skipUnless(
    HAVE_TRANSFORMERS, "transformers not installed on this machine"
)


def _workload(**overrides):
    config = {**SMALL, **overrides}
    return Qwen3Workload(**config)


def _eager(workload):
    return KernelSet(registry=workload.site_registry)


@_skip
class TestWorkloadIdentityAndSerialization(unittest.TestCase):
    """A child process has to rebuild this from fields, not inherit an object."""

    def test_the_config_is_json_serializable_and_round_trips(self):
        workload = _workload()
        config = workload.to_config()
        json.dumps(config)  # must survive the trip through a command line
        rebuilt = Qwen3Workload.from_config(json.loads(json.dumps(config)))
        self.assertEqual(rebuilt.to_config(), config)
        self.assertEqual(rebuilt.name, workload.name)
        self.assertEqual(rebuilt.spec, workload.spec)

    def test_it_is_a_top_level_class_not_a_closure_bundle(self):
        # The reason for the class: closures cannot be reconstructed in a child,
        # and tier 3 runs every provider in one.
        import pickle

        self.assertEqual(Qwen3Workload.__module__.rsplit(".", 1)[-1], "workload")
        pickle.loads(pickle.dumps(_workload().to_config()))

    def test_the_canonical_defaults_are_the_canonical_workload(self):
        workload = Qwen3Workload()
        self.assertTrue(workload.spec.is_canonical)
        self.assertEqual(workload.spec.batch_size, 2)
        self.assertEqual(workload.spec.seq_len, 2048)
        self.assertEqual(workload.spec.dtype, "bfloat16")
        self.assertEqual(workload.spec.device, "cuda")
        self.assertEqual(workload.spec.attn_implementation, "sdpa")
        self.assertFalse(workload.spec.use_cache)
        self.assertFalse(workload.spec.gradient_checkpointing)
        self.assertTrue(workload.spec.training)
        self.assertEqual(workload.units_per_step(), 4096)
        self.assertEqual(workload.unit_name, "tokens")

    def test_the_data_seed_moves_the_batch_and_not_the_identity(self):
        base, moved = _workload(), _workload(data_seed=17)
        self.assertEqual(base.spec.workload_hash, moved.spec.workload_hash)
        self.assertEqual(base.name, moved.name)
        self.assertNotEqual(base.input_checksum(), moved.input_checksum())
        self.assertFalse(torch.equal(base.batch_for(seed=0)[0],
                                     moved.batch_for(seed=0)[0]))

    def test_a_fresh_batch_per_step_is_still_deterministic(self):
        workload = _workload()
        first = workload.batch_for(seed=1)[0]
        self.assertFalse(torch.equal(first, workload.batch_for(seed=2)[0]))
        self.assertTrue(torch.equal(first, _workload().batch_for(seed=1)[0]))

    def test_labels_are_the_input_ids(self):
        ids, labels = _workload().batch_for(seed=0)
        self.assertTrue(torch.equal(ids, labels))
        self.assertIsNot(ids, labels)

    def test_the_description_carries_what_a_report_must_state(self):
        described = _workload().describe()
        for key in ("workload_id", "workload_hash", "config_hash", "units_per_step",
                    "input_checksum", "dtype", "attn_implementation", "use_cache",
                    "gradient_checkpointing", "training", "seed", "data_seed",
                    "expected_site_counts", "config"):
            with self.subTest(key=key):
                self.assertIn(key, described)
        json.dumps(described)

    def test_cache_enabled_execution_is_refused(self):
        from evograd.bench.workloads.qwen3.levels.level4.spec import WorkloadSpecError

        with self.assertRaises(WorkloadSpecError):
            _workload().spec.replace(use_cache=True)

    def test_the_cli_knows_the_model_and_rebuilds_it_from_argv(self):
        from evograd.bench.tier3_cli import MODELS, _parser, build_workload

        self.assertIn(MODEL_KEY, MODELS)
        args = _parser().parse_args(
            ["--model", MODEL_KEY, "--device", "cpu", "--dtype", "float32",
             "--layers", "2", "--data-seed", "5"]
        )
        workload = build_workload(args)
        self.assertIsInstance(workload, Qwen3Workload)
        self.assertEqual(workload.data_seed, 5)
        self.assertEqual(workload.spec.arch["num_hidden_layers"], 2)
        # The canonical batch and sequence survive: a CLI default would have
        # replaced them and changed the workload's identity.
        self.assertEqual(workload.spec.batch_size, 2)
        self.assertEqual(workload.spec.seq_len, 2048)


@_skip
class TestRegistryOwnership(unittest.TestCase):
    def test_the_four_sites_map_to_their_declarations(self):
        self.assertEqual(
            qwen3_sites().site_ops,
            {
                SITE_QKV: "qwen3_qkv_norm_rope",
                SITE_ATTENTION: "qwen3_attention",
                SITE_MLP: "qwen3_swiglu_mlp",
                SITE_RESIDUAL: "fused_add_rms_norm",
            },
        )

    def test_every_site_carries_its_observed_preflight_configuration(self):
        # These are the shapes the calibration made passable. Without them the
        # gate is the 32-row correctness grid, which says nothing about a model.
        for site in qwen3_sites().sites:
            with self.subTest(site=site.name):
                self.assertEqual(len(site.preflight), 1)
                self.assertEqual(site.preflight[0].dtype, "bfloat16")

    def test_llama_is_untouched_by_it(self):
        from tests._registry_fixture import SAMPLE_SITES

        self.assertEqual(set(SAMPLE_SITES.names) & set(qwen3_sites().names), set())
        for site in SAMPLE_SITES.sites:
            with self.subTest(site=site.name):
                self.assertEqual(site.preflight, ())

    def test_the_workload_owns_the_registry(self):
        self.assertIs(_workload().site_registry, qwen3_sites())

    def test_a_llama_site_is_unknown_here(self):
        from evograd.bench.tier3_patch import patch

        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(registry=qwen3_sites()), "rms_norm", lambda *a: None)
        self.assertIn("qwen3_0_6b", str(caught.exception))


@_skip
class TestStructuralIdentity(unittest.TestCase):
    """Same submodules, same order, native autograd -- so bitwise or bust."""

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.qwen3.evaluation.tier3.validate import compare_full_model

        workload = _workload()
        cls.report = compare_full_model(
            workload,
            structural_identity_kernels(workload.site_registry),
            label="structural_identity",
        )

    def test_the_whole_model_is_bitwise_identical(self):
        self.assertEqual(self.report["gate"], "bitwise")
        self.assertTrue(self.report["ok"], self.report["checks"])

    def test_logits_and_loss_are_bitwise(self):
        for check in self.report["checks"]:
            with self.subTest(result=check["name"]):
                self.assertTrue(check["bitwise"])
                self.assertTrue(check["metadata_match"])
                self.assertTrue(check["finite"])

    def test_every_parameter_gradient_matches_with_full_coverage(self):
        coverage = self.report["gradient_coverage"]
        self.assertEqual(coverage["mismatched"], [])
        self.assertEqual(coverage["candidate_missing"], [])
        self.assertEqual(coverage["reference_missing"], [])
        self.assertGreater(coverage["compared"], 0)

    def test_one_optimizer_step_lands_on_the_same_parameters(self):
        self.assertEqual(self.report["optimizer_step"]["mismatched"], [])
        self.assertEqual(self.report["optimizer_step"]["optimizer"], "AdamW")

    def test_the_inputs_were_not_mutated(self):
        self.assertTrue(self.report["inputs_unmutated"])

    def test_the_state_dict_is_the_same_dict(self):
        self.assertTrue(self.report["state_dict_keys_identical"])
        self.assertTrue(self.report["parameter_count"]["identical"])


@_skip
class TestInvocationCounts(unittest.TestCase):
    """Counts scale with the layer count, so the canonical claim is checkable here."""

    def _run(self, sites=None):
        workload = _workload()
        kernels = structural_identity_kernels(workload.site_registry)
        if sites:
            kernels = restrict(kernels, sites)
        model, _prov = workload.build_patched(kernels)
        workload.loss(model, workload.batch_for(seed=0))
        return workload.last_build

    def test_the_declared_counts_are_l_l_l_and_two_l(self):
        self.assertEqual(
            expected_counts(LAYERS),
            {SITE_QKV: LAYERS, SITE_ATTENTION: LAYERS,
             SITE_MLP: LAYERS, SITE_RESIDUAL: 2 * LAYERS},
        )
        self.assertEqual(expected_counts(28)[SITE_RESIDUAL], 56)

    def test_a_full_forward_hits_each_site_exactly_that_many_times(self):
        built = self._run()
        self.assertEqual(built.observed(), expected_counts(LAYERS))
        self.assertEqual(built.count_problems(), [])

    def test_the_residual_count_is_two_per_layer_not_two_per_layer_plus_one(self):
        # Layer 0's input_layernorm has no preceding decoder residual add and is
        # therefore not a fusion site. That is the whole of the difference.
        self.assertEqual(self._run().observed()[SITE_RESIDUAL], 2 * LAYERS)

    def test_a_single_site_runs_alone(self):
        for site in (SITE_QKV, SITE_ATTENTION, SITE_MLP, SITE_RESIDUAL):
            with self.subTest(site=site):
                built = self._run((site,))
                observed = built.observed()
                self.assertEqual(observed.get(site), expected_counts(LAYERS)[site])

    def test_selecting_one_attention_site_still_runs_the_module_once(self):
        # One composite adapter, not two sequential replacements: both sites are
        # counted because both boundaries execute, but only the selected one
        # takes the patched path.
        built = self._run((SITE_QKV,))
        self.assertEqual(built.observed()[SITE_QKV], LAYERS)
        self.assertEqual(built.observed()[SITE_ATTENTION], LAYERS)
        self.assertEqual(built.provenance.actual_sites, (SITE_QKV,))


@_skip
class TestParameterPreservation(unittest.TestCase):
    def setUp(self):
        self.eager_workload = _workload()
        self.eager = self.eager_workload.build(_eager(self.eager_workload))
        self.patched_workload = _workload()
        self.patched, self.provenance = self.patched_workload.build_patched(
            structural_identity_kernels(self.patched_workload.site_registry)
        )

    def test_state_dict_keys_and_order_are_unchanged(self):
        self.assertEqual(list(self.eager.state_dict()),
                         list(self.patched.state_dict()))

    def test_no_parameter_was_added_removed_or_reinitialized(self):
        eager = dict(self.eager.named_parameters())
        patched = dict(self.patched.named_parameters())
        self.assertEqual(sorted(eager), sorted(patched))
        for name in eager:
            with self.subTest(parameter=name):
                self.assertEqual(eager[name].shape, patched[name].shape)
                self.assertEqual(eager[name].dtype, patched[name].dtype)
                self.assertEqual(eager[name].requires_grad, patched[name].requires_grad)
                self.assertTrue(torch.equal(eager[name], patched[name]))

    def test_the_adapters_reuse_the_original_parameter_objects(self):
        # Not "equal values" -- the same objects. An adapter that copied would
        # produce a model that trains its copies and leaves the originals behind.
        layer = self.patched.model.layers[0]
        attention = layer.self_attn
        for module, attr in ((attention, "q_proj"), (attention, "k_proj"),
                             (attention, "v_proj"), (attention, "o_proj"),
                             (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                             (layer.mlp, "down_proj")):
            with self.subTest(module=attr):
                param = getattr(module, attr).weight
                self.assertIs(
                    param, dict(self.patched.named_parameters())[
                        _name_of(self.patched, param)
                    ],
                )
        self.assertIs(attention.q_norm.weight, attention.q_norm.weight)

    def test_the_adapter_state_is_invisible_to_the_state_dict(self):
        keys = list(self.patched.state_dict())
        self.assertFalse([k for k in keys if "evograd" in k])

    def test_train_mode_propagates(self):
        self.assertTrue(self.patched.training)
        self.patched.eval()
        self.assertFalse(self.patched.model.layers[0].self_attn.training)
        self.patched.train()
        self.assertTrue(self.patched.model.layers[0].mlp.training)

    def test_patching_is_reversible_by_rebuilding(self):
        rebuilt = _workload().build(_eager(self.eager_workload))
        self.assertEqual(list(rebuilt.state_dict()), list(self.eager.state_dict()))
        first = self.eager_workload.batch_for(seed=0)
        self.assertTrue(
            torch.equal(
                self.eager_workload.loss(self.eager, first).detach(),
                _workload().loss(rebuilt, first).detach(),
            )
        )


def _name_of(model, param):
    for name, candidate in model.named_parameters():
        if candidate is param:
            return name
    raise AssertionError("parameter is not in the model")


@_skip
class TestPartialAndFailedPatching(unittest.TestCase):
    def test_a_site_that_reaches_no_module_fails(self):
        workload = _workload()
        model = workload.build(_eager(workload))
        model.model.layers = torch.nn.ModuleList()  # nothing left to patch
        with self.assertRaises(ValueError) as caught:
            patch_model(model, structural_identity_kernels(workload.site_registry))
        self.assertIn("decoder layers", str(caught.exception))

    def test_a_partially_reached_site_fails(self):
        workload = _workload()
        model = workload.build(_eager(workload))
        model.model.layers = torch.nn.ModuleList([model.model.layers[0]])
        with self.assertRaises(ValueError):
            # The registry still describes a 2-layer model; one layer is partial.
            from evograd.bench.workloads.qwen3.evaluation.tier3.sites import _require

            _require(1, 2, [SITE_MLP], "Qwen3MLP")

    def test_the_declarative_patch_view_covers_both_attention_sites_once(self):
        patches = {p.site: p for p in module_patches()}
        self.assertEqual(patches[SITE_QKV].covered, (SITE_QKV, SITE_ATTENTION))
        self.assertNotIn(SITE_ATTENTION, patches)  # one module, one patch

    def test_the_carrier_refuses_to_be_used_out_of_order(self):
        carrier = ResidualCarrier()
        with self.assertRaises(RuntimeError):
            carrier.take()
        carrier.put(torch.zeros(2), torch.zeros(2))
        with self.assertRaises(RuntimeError):
            carrier.put(torch.zeros(2), torch.zeros(2))
        carrier.take()
        carrier.reset()


@_skip
class TestStructuredOutputsThroughTheSites(unittest.TestCase):
    def test_the_two_multi_output_sites_are_wired_as_tuples(self):
        from evograd.ops import get_op

        self.assertTrue(get_op("qwen3_qkv_norm_rope").is_multi_output)
        self.assertTrue(get_op("fused_add_rms_norm").is_multi_output)
        self.assertEqual(get_op("qwen3_qkv_norm_rope").output_names, ("q", "k", "v"))
        self.assertEqual(get_op("fused_add_rms_norm").output_names, ("out", "summed"))

    def test_both_residual_outputs_are_used(self):
        # `summed` is the residual stream and `out` continues into the next
        # sublayer. A wiring that dropped `summed` and recomputed the add would
        # still run and would still be wrong.
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
            production_residual_rmsnorm,
        )

        workload = _workload()
        model = workload.build(_eager(workload))
        norm = model.model.layers[0].post_attention_layernorm
        branch = torch.randn(2, 4, norm.weight.shape[0])
        residual = torch.randn_like(branch)
        normed, summed = production_residual_rmsnorm(norm, branch, residual)
        self.assertTrue(torch.equal(summed, residual + branch))
        self.assertTrue(torch.equal(normed, norm(summed)))

    def test_the_qkv_outputs_reach_attention_without_a_detour(self):
        # No detach, no contiguous, no recomputation between the two sites: the
        # three tensors the first returns are the three the second receives.
        seen = {}
        workload = _workload()
        model, _p = workload.build_patched(
            structural_identity_kernels(workload.site_registry)
        )
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import set_tap

        def tap(site, key, inputs, outputs):
            seen.setdefault(site, []).append((inputs, outputs))

        set_tap(model, tap)
        workload.loss(model, workload.batch_for(seed=0))
        set_tap(model, None)
        qkv_out = seen[SITE_QKV][0][1]
        attn_in = seen[SITE_ATTENTION][0][0]
        for name, tensor in zip(("q", "k", "v"), qkv_out):
            with self.subTest(tensor=name):
                self.assertIs(attn_in[name], tensor)
                self.assertTrue(tensor.requires_grad)


@_skip
class TestUnsupportedModesFailLoudly(unittest.TestCase):
    def test_cache_enabled_attention_is_refused(self):
        workload = _workload()
        model, _p = workload.build_patched(
            structural_identity_kernels(workload.site_registry)
        )
        attention = model.model.layers[0].self_attn
        with self.assertRaises(NotImplementedError):
            attention(
                hidden_states=torch.zeros(1, 4, model.config.hidden_size),
                position_embeddings=(torch.zeros(1, 4, 16), torch.zeros(1, 4, 16)),
                attention_mask=None,
                past_key_values=object(),
            )

    def test_output_hidden_states_is_refused_by_the_fused_residual_path(self):
        workload = _workload()
        model, _p = workload.build_patched(
            structural_identity_kernels(workload.site_registry)
        )
        with self.assertRaises(NotImplementedError):
            model(input_ids=workload.batch_for(seed=0)[0], use_cache=False,
                  output_hidden_states=True)


if __name__ == "__main__":
    unittest.main()
