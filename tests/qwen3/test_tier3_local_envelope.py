"""Part A's reference-calibrated local envelope, and the whole-model compile baseline.

The envelope exists because the declared per-result tolerance, calibrated on
synthetic activations, is below one bfloat16 ULP at the magnitudes the
pretrained checkpoint produces, so a trusted implementation that rounds
differently from eager fails part A (2026-09-11 probe). These tests pin what
the fix may and may not do: it is derived from references only, it never
loosens the declared clause, and the whole-model compile provider patches
nothing and cannot be combined with a site patch.
"""

from __future__ import annotations

import unittest

import torch

from evograd.evaluation.tier3.workloads.qwen3_0_6b import boundary


def _worst(max_abs, rel, atol=0.02, rtol=0.01):
    return {"max_abs_err": max_abs, "rel_l2": rel, "atol": atol, "rtol": rtol}


class TestDeriveLocalEnvelope(unittest.TestCase):
    def test_margin_times_max_over_seeds_with_floors(self):
        reports = [
            {"swiglu_mlp": {"out": _worst(2.0, 3.7e-3)}},
            {"swiglu_mlp": {"out": _worst(0.5, 1.0e-3)}},
            {"swiglu_mlp": {"out": _worst(1.0, 2.0e-3)}},
        ]
        env = boundary.derive_local_envelope(reports)
        e = env["sites"]["swiglu_mlp"]["out"]
        self.assertAlmostEqual(e["max_abs"], 2.0 * 2.0)
        self.assertAlmostEqual(e["rel_l2"], 2.0 * 3.7e-3)
        self.assertEqual((e["binding_abs"], e["binding_rel"]), ("reference", "reference"))
        self.assertEqual(env["schema"], boundary.ENVELOPE_SCHEMA)

    def test_floors_bind_when_the_reference_is_bitwise(self):
        env = boundary.derive_local_envelope([{"attention": {"out": _worst(0.0, 0.0, atol=0.0195)}}])
        e = env["sites"]["attention"]["out"]
        self.assertAlmostEqual(e["max_abs"], 2.0 * 0.0195)
        self.assertAlmostEqual(e["rel_l2"], 2.0 * boundary.ENVELOPE_REL_FLOOR)
        self.assertEqual((e["binding_abs"], e["binding_rel"]), ("declared_atol", "floor"))

    def test_an_empty_calibration_is_refused(self):
        with self.assertRaises(ValueError):
            boundary.derive_local_envelope([])


class TestJudge(unittest.TestCase):
    def test_declared_clause_alone_when_no_envelope(self):
        want = torch.full((8,), 32.0)
        actual = want.clone(); actual[0] += 0.25          # one bf16 ULP at 32
        entry = boundary._judge(actual, want, atol=0.035, rtol=0.0, bounds=None)
        self.assertFalse(entry["ok"]); self.assertFalse(entry["declared_ok"])
        self.assertNotIn("envelope", entry)

    def test_envelope_admits_a_reference_sized_disagreement(self):
        want = torch.full((8,), 32.0)
        actual = want.clone(); actual[0] += 0.25
        entry = boundary._judge(actual, want, atol=0.035, rtol=0.0, bounds=(0.5, 7e-3))
        self.assertTrue(entry["ok"]); self.assertFalse(entry["declared_ok"])
        self.assertTrue(entry["envelope"]["ok"])

    def test_envelope_rejects_a_broad_relative_fault_the_declared_rtol_would_pass(self):
        want = torch.linspace(-30, 30, 1000)
        actual = want * 1.02                              # 2% everywhere, inside rtol 0.02
        entry = boundary._judge(actual, want, atol=0.035, rtol=0.02, bounds=(0.5, 7e-3))
        self.assertTrue(entry["declared_ok"])            # the declared clause is unchanged
        self.assertFalse(entry["envelope"]["ok"])        # the envelope alone would refuse it
        self.assertTrue(entry["ok"])                      # OR: the declared verdict stands

    def test_envelope_rejects_a_localized_fault_above_one_reference_ulp(self):
        want = torch.full((8,), 1.0)
        actual = want.clone(); actual[3] += 1.0
        entry = boundary._judge(actual, want, atol=0.035, rtol=0.01, bounds=(0.5, 7e-3))
        self.assertFalse(entry["ok"])

    def test_validate_refuses_an_envelope_of_another_schema(self):
        with self.assertRaises(ValueError):
            boundary.validate_all_invocations(None, None, envelope={"schema": "other", "sites": {}})


class TestPerResultWorst(unittest.TestCase):
    def test_worst_element_and_worst_rel_l2_per_result(self):
        report = boundary.BoundaryReport()
        report.invocations = [
            {"id": "a", "site": "s", "outputs": [{"name": "out", "max_abs_err": 1.0, "rel_l2": 1e-3,
                                                    "atol": 0.1, "rtol": 0.01, "ok": True}], "gradients": []},
            {"id": "b", "site": "s", "outputs": [{"name": "out", "max_abs_err": 0.5, "rel_l2": 5e-3,
                                                    "atol": 0.1, "rtol": 0.01, "ok": True}], "gradients": []},
        ]
        worst = report.per_result_worst()["s"]["out"]
        self.assertEqual((worst["max_abs_err"], worst["worst_id"]), (1.0, "a"))
        self.assertEqual((worst["rel_l2"], worst["worst_rel_id"]), (5e-3, "b"))


class TestWholeModelCompileProvider(unittest.TestCase):
    def test_patches_nothing_and_is_recognised(self):
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites

        kernels = W.whole_model_compile_kernels(qwen3_sites())
        self.assertEqual(kernels.patched, ())
        self.assertTrue(W.whole_model_compile_requested(kernels))
        self.assertFalse(W.whole_model_compile_requested(KernelSet(registry=qwen3_sites())))

    def test_cannot_be_combined_with_a_site_patch(self):
        from evograd.evaluation.tier3.patch import KernelSource, patch
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites

        registry = qwen3_sites()
        kernels = patch(W.whole_model_compile_kernels(registry), "swiglu_mlp", lambda *a: None,
                        source=KernelSource(site="swiglu_mlp", op_name="qwen3_swiglu_mlp",
                                            module=None, origin="callable"))
        wl = W.Qwen3Workload(device="cpu", arch_overrides={"num_hidden_layers": 1})
        with self.assertRaises(ValueError):
            wl.build_patched(kernels)


if __name__ == "__main__":
    unittest.main()


class TestPolicyIndex(unittest.TestCase):
    """One timing run, several patch sets, each judged by its own frozen policy."""

    def _policy_file(self, tmp, name, patched, supporting):
        import json
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import protocol4
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import PatchSet

        ps = PatchSet(tuple(patched), tuple(supporting), {s: 1 for s in (*patched, *supporting)})
        policy = protocol4.derive_policy(
            compile_distances=[{"kl_mean": 1e-3, "global_grad_rel_l2": 1e-2}],
            repeat_distances=[{"kl_mean": 0.0, "global_grad_rel_l2": 1e-3}],
            workload_id="w", workload_hash="h", dtype="bfloat16", environment_hash="e",
            patch_set=ps, data_identity_digest="d", training_plan=protocol4.SCREENING_PLAN,
            metrics=protocol4.SCREENING_METRICS)
        path = tmp / f"{name}.json"
        path.write_text(json.dumps({"schema": protocol4.SCHEMA_VERSION, "policy": policy.to_dict()}))
        return path

    def test_selects_by_patch_set_and_refuses_the_rest(self):
        import json, tempfile
        from pathlib import Path
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import PatchSet, PolicyMismatch

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            a = self._policy_file(tmp, "a", ["swiglu_mlp"], [])
            b = self._policy_file(tmp, "b", ["qkv_norm_rope"], ["attention"])
            index = tmp / "index.json"
            index.write_text(json.dumps({"policy_index": [
                {"calibration": "a.json", "verdict": "a_holdout.json"},
                {"calibration": str(b), "verdict": None}]}))
            wl = W.Qwen3Workload(device="cpu", arch_overrides={"num_hidden_layers": 1},
                                 protocol4_calibration_path=str(index))
            cal, ver = wl._protocol4_files_for(PatchSet(("swiglu_mlp",), (), {"swiglu_mlp": 1}))
            self.assertEqual((Path(cal), Path(ver)), (a, tmp / "a_holdout.json"))
            cal, ver = wl._protocol4_files_for(PatchSet(("qkv_norm_rope",), ("attention",),
                                                        {"qkv_norm_rope": 1, "attention": 1}))
            self.assertEqual((Path(cal), ver), (b, None))
            with self.assertRaises(PolicyMismatch):
                wl._protocol4_files_for(PatchSet(("attention",), ("qkv_norm_rope",),
                                                 {"qkv_norm_rope": 1, "attention": 1}))

    def test_a_plain_policy_file_is_used_as_before(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.simple import PatchSet
        import json, tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = self._policy_file(Path(d), "only", ["residual_rmsnorm"], [])
            wl = W.Qwen3Workload(device="cpu", arch_overrides={"num_hidden_layers": 1},
                                 protocol4_calibration_path=str(p), protocol4_verdict_path="v.json")
            self.assertEqual(wl._protocol4_files_for(PatchSet(("x",), (), {})), (str(p), "v.json"))


class TestUlpFloor(unittest.TestCase):
    def test_ulp_at_max_is_one_unit_in_the_last_place(self):
        want = torch.tensor([1.0, -4928.0, 3.0], dtype=torch.bfloat16)
        self.assertAlmostEqual(boundary.ulp_at_max(want), 32.0)          # bf16 ULP in [4096, 8192)
        self.assertAlmostEqual(boundary.ulp_at_max(torch.tensor([40.75], dtype=torch.bfloat16)), 0.25)
        self.assertEqual(boundary.ulp_at_max(torch.zeros(3, dtype=torch.bfloat16)), 0.0)

    def test_half_ulp_at_max_passes_and_two_ulps_fail(self):
        want = torch.full((64,), 4928.0, dtype=torch.bfloat16); want[3] = -278.0
        actual = want.clone(); actual[3] = -278.0 + 16.0                   # the 2026-09-11 holdout event
        entry = boundary._judge(actual, want, atol=0.0197, rtol=0.01, bounds=(4.0, 7.4e-3))
        self.assertTrue(entry["ok"]); self.assertEqual(entry["envelope"]["abs_binding"], "ulp_floor")
        actual[3] = -278.0 + 80.0                                          # 2.5 ULPs at the tensor's scale
        entry = boundary._judge(actual, want, atol=0.0197, rtol=0.01, bounds=(4.0, 7.4e-3))
        self.assertFalse(entry["ok"])

    def test_broad_faults_still_fail_through_rel_l2(self):
        want = torch.linspace(-4000, 4000, 4096, dtype=torch.bfloat16)
        entry = boundary._judge(want * 1.02, want, atol=0.0197, rtol=0.01, bounds=(4.0, 7.4e-3))
        self.assertFalse(entry["ok"])


class TestPurityDriftUlpFloor(unittest.TestCase):
    def test_ulp_at_max(self):
        from evograd.evaluation.tier3.gate import purity
        self.assertAlmostEqual(purity._ulp_at_max(torch.tensor([50.0], dtype=torch.bfloat16)), 0.25)
        self.assertEqual(purity._ulp_at_max(torch.zeros(2)), 0.0)
        self.assertEqual(purity._ulp_at_max(torch.tensor([3])), 0.0)

    def test_one_rounding_flip_is_not_drift_but_a_growing_spread_is(self):
        """A kernel whose dweight lands on the other side of a bf16 rounding
        boundary on one call (0.125 at |dweight| ~ 50) is admitted; a kernel
        whose spread exceeds one ULP at its scale is still refused."""
        from unittest import mock
        from evograd.evaluation.tier3.gate import purity
        first = {"grad:dweight": torch.full((8,), 50.0, dtype=torch.bfloat16)}
        flip = {"grad:dweight": first["grad:dweight"].clone()}; flip["grad:dweight"][2] += 0.25
        big = {"grad:dweight": first["grad:dweight"].clone()}; big["grad:dweight"][2] += 1.0
        calls = iter([{"results": first, "mutated_inputs": [], "saved_structure": ["grad:dweight"], "finite": True}]
                     + [{"results": (flip if i == 5 else first), "mutated_inputs": [], "saved_structure": ["grad:dweight"], "finite": True} for i in range(2, 12)])
        with mock.patch.object(purity, "_one_call", side_effect=lambda *a, **k: next(calls)), \
             mock.patch.object(purity, "is_production_default", return_value=False), \
             mock.patch.object(purity, "_tolerance", return_value=(1.6, 0.08)), \
             mock.patch("evograd.benchmark.get_task") as task:
            op = mock.MagicMock(); op.benchmark_workloads.return_value = [mock.MagicMock(dims={}, dtype="bfloat16")]
            task.return_value = op
            with mock.patch("evograd.opdecl.inputs.make_case_inputs", return_value={}):
                report = purity.check_site("s", "op", lambda *a: None, registry=None, suite="x", calls=11)
        self.assertIsNone(report["first_drift"]); self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["determinism_ulp_floor"]["grad:dweight"], 0.25)
        calls = iter([{"results": first, "mutated_inputs": [], "saved_structure": ["grad:dweight"], "finite": True}]
                     + [{"results": (big if i == 5 else first), "mutated_inputs": [], "saved_structure": ["grad:dweight"], "finite": True} for i in range(2, 12)])
        with mock.patch.object(purity, "_one_call", side_effect=lambda *a, **k: next(calls)), \
             mock.patch.object(purity, "is_production_default", return_value=False), \
             mock.patch.object(purity, "_tolerance", return_value=(1.6, 0.08)), \
             mock.patch("evograd.benchmark.get_task") as task:
            op = mock.MagicMock(); op.benchmark_workloads.return_value = [mock.MagicMock(dims={}, dtype="bfloat16")]
            task.return_value = op
            with mock.patch("evograd.opdecl.inputs.make_case_inputs", return_value={}):
                report = purity.check_site("s", "op", lambda *a: None, registry=None, suite="x", calls=11)
        self.assertIsNotNone(report["first_drift"]); self.assertFalse(report["ok"])
