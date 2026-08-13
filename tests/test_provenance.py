"""Provenance is an assertion, not a comment.

A benchmark whose shapes are hand-typed constants cannot be audited: the pre-v1
declarations already carried Llama-3 and GPT-2 dimensions, but only in comments,
so nothing would have caught an edit that drifted away from the model it claimed
to measure. These tests re-derive every timed workload from the configuration its
provenance names and fail if the two ever disagree.

CPU-only by construction: nothing here builds a tensor.
"""

from __future__ import annotations

import unittest

from evograd.opdecl import Provenance, Workload
from evograd.opdecl.models import (
    ALPHAFOLD3,
    LLAMA_3_8B,
    MODELS,
    config_for,
    rederive_dims,
)
from evograd.ops import OPS


class TestModelRegistry(unittest.TestCase):
    def test_registry_keys_match_config_names(self):
        for key, config in MODELS.items():
            self.assertEqual(key, config.name)
            self.assertTrue(key.isidentifier(), key)

    def test_registry_imports_without_torch(self):
        import ast
        import pathlib

        import evograd.opdecl.models as models

        source = pathlib.Path(models.__file__).read_text(encoding="utf-8")
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        # Declarations must stay importable on machines without torch, and every
        # declaration that derives a shape imports this module.
        self.assertNotIn("torch", imported)

    def test_llama_derived_widths(self):
        self.assertEqual(LLAMA_3_8B.q_out, 4096)  # 32 heads x 128
        self.assertEqual(LLAMA_3_8B.kv_out, 1024)  # 8 kv heads x 128, GQA
        self.assertEqual(LLAMA_3_8B.q_out, LLAMA_3_8B.hidden)

    def test_llama_rope_theta_is_not_the_llama2_value(self):
        # A RoPE kernel built against 10000 is self-consistent and wrong.
        self.assertEqual(LLAMA_3_8B.rope_theta, 500000.0)

    def test_decoder_layer_dims_are_all_shape_recoverable(self):
        # dispatch._route rebuilds the dim dict from tensor shapes alone, so a
        # dim that appears in no shape string could never be recovered at deploy
        # time. n_heads/n_kv_heads are therefore deliberately absent.
        dims = LLAMA_3_8B.decoder_layer_dims(batch=1, seq=2048)
        self.assertNotIn("n_heads", dims)
        self.assertNotIn("n_kv_heads", dims)
        self.assertEqual(dims["q_out"] // dims["head_dim"], LLAMA_3_8B.n_heads)
        self.assertEqual(dims["kv_out"] // dims["head_dim"], LLAMA_3_8B.n_kv_heads)

    def test_triangle_attention_msa_axis_equals_the_residue_axis(self):
        # Triangle attention's N_seq axis *is* the third residue index. The
        # pre-v1 grid used S=64 at N in {128, 256}, understating the real work
        # by 2x and 4x; deriving the dims makes that unrepresentable.
        for residues in (128, 256, 384):
            dims = ALPHAFOLD3.triangle_attention_dims(batch=1, residues=residues)
            self.assertEqual(dims["S"], dims["N"], residues)

    def test_rederive_reports_an_unknown_component_clearly(self):
        bogus = Provenance(model="llama_3_8b", component="not_a_layer", free={})
        with self.assertRaises(AttributeError) as caught:
            rederive_dims(bogus)
        self.assertIn("not_a_layer", str(caught.exception))

    def test_rederive_reports_an_unknown_model_clearly(self):
        bogus = Provenance(model="llama_3_8b", component="rmsnorm", free={})
        object.__setattr__(bogus, "model", "gpt_9")
        with self.assertRaises(KeyError) as caught:
            config_for(bogus)
        self.assertIn("gpt_9", str(caught.exception))


class TestDeclaredProvenance(unittest.TestCase):
    """The load-bearing test: declared shapes must match the models they cite."""

    def _benchmark_tasks(self):
        return [(name, op) for name, op in OPS.items() if op.level is not None]

    def test_every_benchmark_workload_declares_provenance(self):
        # declare_op already enforces this at construction; assert it here too so
        # the guarantee is visible where the rest of the provenance rules live.
        for name, op in self._benchmark_tasks():
            for workload in op.benchmark:
                self.assertIsNotNone(
                    workload.provenance,
                    f"{name}: timed workload {workload.dims} has no provenance",
                )

    def test_hf_config_workloads_are_recomputable_from_their_provenance(self):
        """The strong claim: ``hf_config`` means every dim is machine-derivable.

        Only this source is re-derived. A ``paper`` shape is quoted from a
        published sweep and a ``handpicked`` one is a human's choice — forcing
        either through a model config would mean inventing a configuration to
        justify a number, which is precisely the failure mode provenance exists
        to prevent. Those two carry a mandatory note instead (below).
        """
        checked = 0
        for name, op in self._benchmark_tasks():
            for workload in op.benchmark:
                provenance = workload.provenance
                if provenance.source != "hf_config":
                    continue
                checked += 1
                with self.subTest(op=name, dims=workload.dims):
                    self.assertEqual(
                        workload.dims,
                        rederive_dims(provenance),
                        f"{name}: declared dims disagree with "
                        f"{provenance.model}.{provenance.component}"
                        f"(**{provenance.free})",
                    )
        # Guard against the test silently passing because a refactor dropped
        # every hf_config provenance.
        self.assertGreater(checked, 0, "no hf_config workload was checked")

    def test_weaker_provenance_sources_justify_themselves(self):
        for name, op in self._benchmark_tasks():
            for workload in op.benchmark:
                provenance = workload.provenance
                if provenance.source == "hf_config":
                    continue
                self.assertTrue(
                    provenance.note.strip(),
                    f"{name}: {provenance.source!r} workload {workload.dims} "
                    "must say in 'note' where the shape came from",
                )

    def test_scaled_provenance_explains_itself(self):
        # A shape that deviates from the real configuration is allowed, but it
        # has to say which dimension was scaled and why — otherwise "derived from
        # a real model" quietly stops being true.
        for name, op in self._benchmark_tasks():
            for workload in op.benchmark:
                if workload.provenance.scaled:
                    self.assertTrue(
                        workload.provenance.note.strip(),
                        f"{name}: scaled workload {workload.dims} has no note",
                    )

    def test_every_benchmark_task_declares_a_level_and_family(self):
        for name, op in OPS.items():
            self.assertIsNotNone(op.level, f"{name}: no benchmark level")
            self.assertIsNotNone(op.family, f"{name}: no benchmark family")
            self.assertIn(op.level, (1, 2, 3), name)


class TestFixedShapeSuites(unittest.TestCase):
    def test_one_case_per_suite_and_stable_keys(self):
        from evograd.ops._common import fixed_shape_suites

        workloads = (
            Workload(dims={"rows": 4096, "hidden": 4096}, dtype="bfloat16"),
            Workload(dims={"rows": 4096, "hidden": 4096}, dtype="float32"),
            Workload(dims={"rows": 8192, "hidden": 4096}, dtype="bfloat16"),
        )
        suites = fixed_shape_suites(workloads)
        self.assertEqual(len(suites), 3)
        for cases in suites.values():
            self.assertEqual(len(cases), 1)
        self.assertIn("fixed/rows4096-hidden4096-bfloat16", suites)
        self.assertEqual(suites, fixed_shape_suites(workloads))


if __name__ == "__main__":
    unittest.main()
