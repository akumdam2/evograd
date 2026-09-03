"""The Qwen3-0.6B Level-1 mapping.

Six generic primitives carry every configuration the canonical step runs. What
these tests protect is that the mapping stays *derived*: dims, roles,
frequencies and layouts come from the harvest, the provenance re-derives the
same dims from the published configuration, and the Level-1 shapes still compose
into the Level-2 contracts that were verified against the model.
"""

from __future__ import annotations

import unittest

import torch

from evograd.bench.workloads.qwen3.harvest import snapshot as snapshot_module
from evograd.bench.workloads.qwen3.levels.level1.mapping import COMPOSES_INTO, mapping
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.models import QWEN3_0_6B, rederive_dims
from evograd.ops import OPS, get_op

LEVEL1 = snapshot_module.load()["level1"]
TASKS = (
    "linear_no_bias",
    "rmsnorm",
    "rope",
    "swiglu",
    "causal_gqa_attention",
    "cross_entropy",
)


def _by_roles(task):
    return {tuple(c["roles"]): c for c in LEVEL1[task]["configurations"]}


class TestMappingShape(unittest.TestCase):
    def test_exactly_six_tasks_are_mapped(self):
        self.assertEqual(tuple(LEVEL1), TASKS)
        for task in TASKS:
            with self.subTest(task=task):
                self.assertIn(task, OPS)
                self.assertEqual(OPS[task].level, 1)

    def test_linear_has_the_six_deduplicated_configurations(self):
        configs = _by_roles("linear_no_bias")
        expected = {
            ("q_proj",): ({"M": 4096, "K": 1024, "N": 2048}, 28),
            ("k_proj", "v_proj"): ({"M": 4096, "K": 1024, "N": 1024}, 56),
            ("o_proj",): ({"M": 4096, "K": 2048, "N": 1024}, 28),
            ("gate_proj", "up_proj"): ({"M": 4096, "K": 1024, "N": 3072}, 56),
            ("down_proj",): ({"M": 4096, "K": 3072, "N": 1024}, 28),
            ("lm_head",): ({"M": 4096, "K": 1024, "N": 151936}, 1),
        }
        self.assertEqual(set(configs), set(expected))
        for roles, (dims, frequency) in expected.items():
            with self.subTest(roles=roles):
                self.assertEqual(configs[roles]["dims"], dims)
                self.assertEqual(configs[roles]["frequency"], frequency)
                self.assertEqual(configs[roles]["dtype"], "torch.bfloat16")

    def test_qwen_maps_onto_the_biasless_linear_task(self):
        """A zero-valued bias is not a biasless projection: it adds a broadcast
        add, a dbias reduction and a third gradient the model never computes."""
        self.assertIn("linear_no_bias", LEVEL1)
        self.assertNotIn("linear", LEVEL1)
        nobias = get_op("linear_no_bias")
        self.assertEqual([a.name for a in nobias.args], ["x", "weight"])
        self.assertEqual(nobias.grad_names(), ("dx", "dweight"))
        self.assertNotIn("bias", [a.name for a in nobias.args])
        self.assertNotIn("dbias", nobias.grad_names())
        # And the biased task no longer claims any Qwen workload.
        biased = get_op("linear")
        self.assertNotIn("qwen3_0_6b_observed", biased.benchmark_suites)
        self.assertTrue(all(w.provenance.model != "qwen3_0_6b" for w in biased.coverage))

    def test_every_mapped_linear_configuration_is_biasless(self):
        for config in LEVEL1["linear_no_bias"]["configurations"]:
            with self.subTest(roles=config["roles"]):
                self.assertFalse(config["attrs"]["bias"])

    def test_the_biased_task_labels_itself_an_ablation(self):
        biased = get_op("linear")
        self.assertTrue(all(w.provenance.scaled for w in biased.benchmark))
        for workload in biased.benchmark:
            self.assertIn("no projection biases", workload.provenance.note)
        import evograd.ops.level1.linear as linear_module

        # `op.__doc__` is OpDecl's; the claim lives in the module that declares it.
        self.assertIn("ablation", linear_module.__doc__)
        self.assertIn("linear_no_bias", linear_module.__doc__)

    def test_the_linear_frequencies_account_for_every_invocation(self):
        """197 = 28 x 7 per-layer projections + one lm_head."""
        total = sum(c["frequency"] for c in LEVEL1["linear_no_bias"]["configurations"])
        self.assertEqual(total, 197)
        self.assertEqual(total, LEVEL1["linear_no_bias"]["total_frequency"])
        self.assertEqual(total, 28 * 7 + 1)

    def test_rmsnorm_has_three_configurations(self):
        configs = _by_roles("rmsnorm")
        self.assertEqual(
            configs[("input_layernorm", "norm", "post_attention_layernorm")]["dims"],
            {"rows": 4096, "hidden": 1024},
        )
        self.assertEqual(
            configs[("input_layernorm", "norm", "post_attention_layernorm")]["frequency"], 57
        )
        self.assertEqual(configs[("q_norm",)]["dims"], {"rows": 65536, "hidden": 128})
        self.assertEqual(configs[("k_norm",)]["dims"], {"rows": 32768, "hidden": 128})
        for roles in (("q_norm",), ("k_norm",)):
            self.assertEqual(configs[roles]["frequency"], 28)
            self.assertEqual(configs[roles]["attrs"]["eps"], 1e-6)

    def test_rope_maps_one_record_onto_two_workloads(self):
        """``apply_rotary_pos_emb`` rotates q and k in one call, at different
        head counts. They are two RoPE workloads from one harvested record."""
        configs = LEVEL1["rope"]["configurations"]
        self.assertEqual(len(configs), 2)
        self.assertEqual({c["config_id"] for c in configs}, {"34378ecf454fc895"})
        self.assertEqual(
            [c["dims"] for c in configs],
            [
                {"B": 2, "n_heads": 16, "T": 2048, "head_dim": 128},
                {"B": 2, "n_heads": 8, "T": 2048, "head_dim": 128},
            ],
        )
        self.assertTrue(all(c["frequency"] == 28 for c in configs))

    def test_swiglu_carries_the_mlp_and_projection_provenance(self):
        entry = LEVEL1["swiglu"]
        config = entry["configurations"][0]
        self.assertEqual(config["dims"], {"rows": 4096, "cols": 3072})
        self.assertEqual(config["frequency"], 28)
        self.assertEqual(config["roles"], ["act_fn"])
        self.assertEqual(sorted(entry["supporting"]), ["gate_up_projection", "mlp"])
        self.assertEqual(entry["supporting"]["gate_up_projection"]["frequency"], 56)
        self.assertEqual(entry["supporting"]["mlp"]["attrs"]["intermediate_size"], 3072)

    def test_causal_gqa_attention_configuration(self):
        config = LEVEL1["causal_gqa_attention"]["configurations"][0]
        self.assertEqual(config["config_id"], "9674b971ae24b325")
        self.assertEqual(config["frequency"], 28)
        self.assertEqual(config["dims"], {"B": 2, "HQ": 16, "HK": 8, "T": 2048, "D": 128})
        attrs = config["attrs"]
        self.assertTrue(attrs["is_causal"] and attrs["enable_gqa"])
        self.assertEqual(attrs["dropout_p"], 0.0)
        self.assertFalse(attrs["attn_mask_provided"])

    def test_cross_entropy_traces_through_the_causal_wrapper(self):
        entry = LEVEL1["cross_entropy"]
        config = entry["configurations"][0]
        self.assertEqual(config["dims"], {"rows": 4096, "cols": 151936})
        self.assertEqual(config["dtype"], "torch.float32")
        self.assertEqual(config["frequency"], 1)
        self.assertEqual(config["attrs"]["ignore_index"], -100)
        self.assertEqual(config["attrs"]["reduction"], "mean")
        wrapper = entry["supporting"]["causal_wrapper"]
        # The BF16 [2, 2048, 151936] logits the model produced, upcast and
        # flattened inside Transformers before reaching the call above.
        self.assertEqual(wrapper["input_shapes"][0]["shape"], [2, 2048, 151936])
        self.assertEqual(wrapper["input_shapes"][0]["dtype"], "torch.bfloat16")
        self.assertEqual(wrapper["attrs"]["vocab_size"], 151936)

    def test_standalone_softmax_is_deliberately_unmapped(self):
        """The model runs fused SDPA and never materializes a softmax."""
        self.assertNotIn("softmax", LEVEL1)
        # No suite at all, rather than an empty one: an empty suite would look
        # like a mapping that happened to find nothing.
        self.assertNotIn("qwen3_0_6b_observed", get_op("softmax").benchmark_suites)
        self.assertTrue(
            all(w.provenance.model != "qwen3_0_6b" for w in get_op("softmax").benchmark)
        )

    def test_the_silu_record_maps_onto_swiglu_not_a_bare_activation(self):
        self.assertEqual(LEVEL1["swiglu"]["harvested_task"], "silu")
        self.assertEqual(
            LEVEL1["swiglu"]["configurations"][0]["config_id"], "be8d90aa36b9bf50"
        )


class TestProvenance(unittest.TestCase):
    def test_every_mapped_configuration_rederives_from_the_published_config(self):
        for task in TASKS:
            op = get_op(task)
            for workload in op.benchmark_workloads("qwen3_0_6b_observed"):
                with self.subTest(task=task, dims=workload.dims):
                    self.assertEqual(workload.provenance.source, "hf_config")
                    self.assertEqual(workload.provenance.model, "qwen3_0_6b")
                    self.assertEqual(workload.dims, rederive_dims(workload.provenance))

    def test_the_suites_match_the_snapshot_configuration_lists(self):
        for task in TASKS:
            with self.subTest(task=task):
                suite = get_op(task).benchmark_workloads("qwen3_0_6b_observed")
                configs = LEVEL1[task]["configurations"]
                self.assertEqual(len(suite), len(configs))
                self.assertEqual(
                    [w.dims for w in suite], [c["dims"] for c in configs]
                )

    def test_module_paths_and_layer_indices_are_carried(self):
        configs = _by_roles("linear_no_bias")
        gate_up = configs[("gate_proj", "up_proj")]
        self.assertEqual(len(gate_up["module_paths"]), 56)
        self.assertEqual(gate_up["layer_indices"], list(range(28)))
        self.assertIn("model.layers.14.mlp.gate_proj", gate_up["module_paths"])
        self.assertEqual(configs[("lm_head",)]["module_paths"], ["lm_head"])

    def test_the_snapshot_hash_still_verifies(self):
        payload = snapshot_module.load()
        self.assertEqual(payload["snapshot_hash"], snapshot_module.snapshot_hash(payload))


class TestDefaultsPreserved(unittest.TestCase):
    """The Qwen cases are added; the Llama-derived defaults are not replaced."""

    def test_default_benchmark_grids_stay_llama_derived(self):
        for task in ("linear_no_bias", "rmsnorm", "rope", "swiglu", "cross_entropy"):
            with self.subTest(task=task):
                models = {w.provenance.model for w in get_op(task).benchmark}
                self.assertEqual(models, {"llama_3_8b"})

    def test_the_new_task_also_has_a_llama_default(self):
        models = {w.provenance.model for w in get_op("causal_gqa_attention").benchmark}
        self.assertEqual(models, {"llama_3_8b"})

    def test_legacy_ablation_suites_survive(self):
        self.assertEqual(len(get_op("swiglu").benchmark_workloads("legacy")), 42)
        self.assertEqual(len(get_op("cross_entropy").benchmark_workloads("legacy")), 42)

    def test_regime_suites_still_partition_the_default_grid(self):
        for task in ("rmsnorm", "rope", "swiglu", "cross_entropy", "causal_gqa_attention"):
            with self.subTest(task=task):
                op = get_op(task)
                full = op.benchmark_workloads("full")
                small = op.benchmark_workloads("small")
                large = op.benchmark_workloads("large")
                self.assertEqual(len(full), len(small) + len(large))

    def test_the_observed_cases_are_in_coverage(self):
        for task in TASKS:
            with self.subTest(task=task):
                op = get_op(task)
                observed = set(
                    (tuple(sorted(w.dims.items())), w.dtype)
                    for w in op.benchmark_workloads("qwen3_0_6b_observed")
                )
                covered = set(
                    (tuple(sorted(w.dims.items())), w.dtype) for w in op.coverage
                )
                self.assertTrue(observed <= covered, task)


class TestObservedLayout(unittest.TestCase):
    """Real dtype, shape and stride -- not a contiguous substitute."""

    def test_rope_inputs_are_non_contiguous_head_major(self):
        op = get_op("rope")
        for workload in op.benchmark_workloads("qwen3_0_6b_observed"):
            with self.subTest(dims=workload.dims):
                values = make_case_inputs(op, workload, device="cpu")
                dims = workload.dims
                expected = [
                    dims["T"] * dims["n_heads"] * dims["head_dim"],
                    dims["head_dim"],
                    dims["n_heads"] * dims["head_dim"],
                    1,
                ]
                for name in ("x", "dy"):
                    self.assertFalse(values[name].is_contiguous(), name)
                    self.assertEqual(list(values[name].stride()), expected, name)

    def test_the_rope_strides_are_the_observed_ones(self):
        config = LEVEL1["rope"]["configurations"][0]
        op = get_op("rope")
        workload = op.benchmark_workloads("qwen3_0_6b_observed")[0]
        values = make_case_inputs(op, workload, device="cpu")
        self.assertEqual(list(values["x"].stride()), config["inputs"][0]["stride"])

    def test_attention_inputs_are_non_contiguous_head_major(self):
        op = get_op("causal_gqa_attention")
        workload = op.benchmark_workloads("qwen3_0_6b_observed")[0]
        values = make_case_inputs(op, workload, device="cpu")
        observed = LEVEL1["causal_gqa_attention"]["configurations"][0]["inputs"]
        for name, entry in zip(("q", "k", "v"), observed):
            with self.subTest(tensor=name):
                self.assertFalse(values[name].is_contiguous())
                self.assertEqual(list(values[name].shape), entry["shape"])
                self.assertEqual(list(values[name].stride()), entry["stride"])

    def test_the_llama_grids_keep_their_contiguous_layout(self):
        """Changing them would make old numbers incomparable."""
        for task in ("rope", "causal_gqa_attention"):
            op = get_op(task)
            values = make_case_inputs(op, op.benchmark[0], device="cpu")
            name = "x" if task == "rope" else "q"
            with self.subTest(task=task):
                self.assertTrue(values[name].is_contiguous())

    def test_rope_uses_the_model_s_own_rotary_base(self):
        """Llama-3's 500000 and Qwen3's 1000000 are different functions."""
        op = get_op("rope")
        qwen = make_case_inputs(op, op.benchmark_workloads("qwen3_0_6b_observed")[0], device="cpu")
        llama = make_case_inputs(op, op.benchmark[0], device="cpu")
        # Column 0 is theta^0 = 1 for every base and column -1 is the slowest
        # frequency, where both angles are near zero. Column 1 at a token far
        # enough along separates 500000 from 1000000 unmistakably.
        self.assertNotAlmostEqual(
            float(qwen["cos"][64, 1]), float(llama["cos"][64, 1]), places=2
        )
        self.assertEqual(QWEN3_0_6B.rope_theta, 1000000.0)


class TestCausalGqaAttentionTask(unittest.TestCase):
    def test_registered_as_a_generic_level_one_task(self):
        op = get_op("causal_gqa_attention")
        self.assertEqual((op.level, op.family), (1, "attention"))
        self.assertEqual([a.name for a in op.args], ["q", "k", "v"])
        self.assertEqual(op.output_names, ("o",))
        self.assertEqual(op.grad_names(), ("dq", "dk", "dv"))

    def test_runtime_forward_is_sdpa_and_the_oracle_is_not_timed(self):
        import inspect

        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
        from evograd.ops.level1.causal_gqa_attention import forward_ref

        op = get_op("causal_gqa_attention")
        self.assertIs(
            resolve_runtime_forward(op), forward_ref.causal_gqa_attention_forward_production
        )
        self.assertIs(resolve_forward(op), forward_ref.causal_gqa_attention_forward_ref)
        timed = inspect.getsource(forward_ref.causal_gqa_attention_forward_production)
        dense = inspect.getsource(forward_ref.causal_gqa_attention_forward_ref)
        self.assertIn("scaled_dot_product_attention", timed)
        self.assertNotIn("softmax", timed)
        self.assertIn("masked_fill", dense)

    def test_the_two_spellings_agree(self):
        from evograd.ops.level1.causal_gqa_attention.forward_ref import (
            causal_gqa_attention_forward_production,
            causal_gqa_attention_forward_ref,
        )

        torch.manual_seed(0)
        q = torch.randn(2, 8, 24, 16).transpose(1, 2).transpose(1, 2)
        k = torch.randn(2, 2, 24, 16)
        v = torch.randn(2, 2, 24, 16)
        dense = causal_gqa_attention_forward_ref(q, k, v)
        fused = causal_gqa_attention_forward_production(q, k, v)
        self.assertLess(
            float((dense - fused).abs().max()) / float(fused.abs().max()), 1e-6
        )

    def test_attention_is_causal_and_grouped(self):
        from evograd.ops.level1.causal_gqa_attention.forward_ref import (
            causal_gqa_attention_forward_production,
        )

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 8, 16, 16), torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16))
        base = causal_gqa_attention_forward_production(q, k, v)
        future = k.clone()
        future[:, :, -1] += 100.0
        shifted = causal_gqa_attention_forward_production(q, future, v)
        self.assertTrue(torch.allclose(base[:, :, :-1], shifted[:, :, :-1], atol=1e-5))
        odd = torch.randn(1, 3, 16, 16)  # 8 query heads is not divisible by 3
        with self.assertRaises(ValueError):
            causal_gqa_attention_forward_production(q, odd, odd)

    def test_backward_returns_three_gradients(self):
        from evograd.opdecl.oracle import oracle

        op = get_op("causal_gqa_attention")
        values = make_case_inputs(op, op.correctness[0], device="cpu")
        out, grads = oracle(op, values)
        self.assertEqual(sorted(grads), ["dk", "dq", "dv"])
        self.assertEqual(list(out.shape), [1, 4, 16, 16])
        for name, grad in grads.items():
            self.assertTrue(torch.isfinite(grad).all(), name)

    def test_the_output_projection_is_not_part_of_it(self):
        op = get_op("causal_gqa_attention")
        self.assertIn("output projection is a separate GEMM", op.extra_constraints)
        self.assertNotIn("o_weight", [a.name for a in op.args])


class TestComposition(unittest.TestCase):
    """Level-1 outputs must be the Level-2 inputs that were verified."""

    def test_linear_rmsnorm_rope_compose_into_qkv_norm_rope(self):
        level2 = get_op("qwen3_qkv_norm_rope").benchmark[0].dims
        linear = _by_roles("linear_no_bias")
        self.assertEqual(linear[("q_proj",)]["dims"]["N"], level2["QO"])
        self.assertEqual(linear[("k_proj", "v_proj")]["dims"]["N"], level2["KVO"])
        self.assertEqual(linear[("q_proj",)]["dims"]["K"], level2["H"])
        norms = _by_roles("rmsnorm")
        self.assertEqual(norms[("q_norm",)]["dims"]["hidden"], level2["D"])
        self.assertEqual(norms[("k_norm",)]["dims"]["hidden"], level2["D"])
        self.assertEqual(
            norms[("q_norm",)]["dims"]["rows"], level2["B"] * level2["T"] * level2["HQ"]
        )
        rope_q, rope_k = LEVEL1["rope"]["configurations"]
        self.assertEqual(rope_q["dims"]["n_heads"], level2["HQ"])
        self.assertEqual(rope_k["dims"]["n_heads"], level2["HK"])
        self.assertEqual(rope_q["dims"]["head_dim"], level2["D"])

    def test_attention_and_o_proj_compose_into_qwen3_attention(self):
        level2 = get_op("qwen3_attention").benchmark[0].dims
        sdpa = LEVEL1["causal_gqa_attention"]["configurations"][0]["dims"]
        for dim in ("B", "HQ", "HK", "T", "D"):
            self.assertEqual(sdpa[dim], level2[dim], dim)
        o_proj = _by_roles("linear_no_bias")[("o_proj",)]["dims"]
        self.assertEqual(o_proj["K"], level2["QO"])
        self.assertEqual(o_proj["N"], level2["H"])
        self.assertEqual(o_proj["K"], sdpa["HQ"] * sdpa["D"])

    def test_linear_and_swiglu_compose_into_qwen3_swiglu_mlp(self):
        level2 = get_op("qwen3_swiglu_mlp").benchmark[0].dims
        linear = _by_roles("linear_no_bias")
        gate_up = linear[("gate_proj", "up_proj")]["dims"]
        down = linear[("down_proj",)]["dims"]
        swiglu = LEVEL1["swiglu"]["configurations"][0]["dims"]
        self.assertEqual(gate_up["K"], level2["H"])
        self.assertEqual(gate_up["N"], level2["I"])
        self.assertEqual(swiglu["cols"], level2["I"])
        self.assertEqual(swiglu["rows"], level2["B"] * level2["T"])
        self.assertEqual(down["K"], level2["I"])
        self.assertEqual(down["N"], level2["H"])

    def test_rmsnorm_composes_into_fused_add_rms_norm(self):
        level2 = get_op("fused_add_rms_norm").benchmark_workloads("qwen3_0_6b_observed")[0]
        residual = _by_roles("rmsnorm")[
            ("input_layernorm", "norm", "post_attention_layernorm")
        ]["dims"]
        self.assertEqual(residual["rows"], level2.dims["rows"])
        self.assertEqual(residual["hidden"], level2.dims["cols"])

    def test_the_declared_composition_map_covers_every_task(self):
        self.assertEqual(set(COMPOSES_INTO), set(TASKS))
        for task, targets in COMPOSES_INTO.items():
            for target in targets:
                self.assertIn(target, OPS, f"{task} -> {target}")

    def test_the_mapping_command_reports_the_snapshot_it_read(self):
        report = mapping()
        self.assertEqual(report["snapshot_hash"], snapshot_module.load()["snapshot_hash"])
        self.assertEqual(set(report["tasks"]), set(TASKS))


if __name__ == "__main__":
    unittest.main()
