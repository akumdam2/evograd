import sys
from pathlib import Path
import unittest

from evograd.geak_nvidia.profile_adapter import profile_result_to_geak
from evograd.ncu.profile import ProfileResult


class TestNvidiaProfileAdapter(unittest.TestCase):
    def test_maps_ncu_metrics_to_geak_schema(self):
        rows = (
            {
                "kernel": "_ln_dx",
                "metric": "gpu__time_duration.sum",
                "value": 12000.0,
                "unit": "nsecond",
            },
            {
                "kernel": "_ln_dx",
                "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "value": 32.0,
                "unit": "%",
            },
            {
                "kernel": "_ln_dx",
                "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "value": 71.0,
                "unit": "%",
            },
            {
                "kernel": "_ln_dx",
                "metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                "value": 55.0,
                "unit": "%",
            },
            {
                "kernel": "_ln_dx",
                "metric": "launch__registers_per_thread",
                "value": 72.0,
                "unit": "register/thread",
            },
        )
        payload = profile_result_to_geak(ProfileResult(ok=True, kernels=rows))
        self.assertTrue(payload["success"])
        kernel = payload["results"][0]["kernels"][0]
        self.assertAlmostEqual(kernel["duration_us"], 12.0)
        self.assertEqual(kernel["bottleneck"], "memory")
        self.assertEqual(kernel["metrics"]["memory.hbm_bandwidth_utilization"], 71.0)
        self.assertEqual(kernel["metrics"]["registers_per_thread"], 72.0)

    def test_failure_does_not_fabricate_counters(self):
        payload = profile_result_to_geak(
            ProfileResult(ok=False, error="ERR_NVGPUCTRPERM")
        )
        self.assertFalse(payload["success"])
        self.assertEqual(payload["results"], [])
        self.assertIn("ERR_NVGPUCTRPERM", payload["error"])

    def test_geak_baseline_and_guidance_consume_schema(self):
        rows = (
            {
                "kernel": "_ln_fwd",
                "metric": "gpu__time_duration.sum",
                "value": 18.0,
                "unit": "usecond",
            },
            {
                "kernel": "_ln_fwd",
                "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "value": 20.0,
                "unit": "%",
            },
            {
                "kernel": "_ln_fwd",
                "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "value": 66.0,
                "unit": "%",
            },
        )
        payload = profile_result_to_geak(ProfileResult(ok=True, kernels=rows))
        geak_src = Path(
            "/u/wzhan/tmp/geak_blackbox_layernorm_20260806/vendor/GEAK-v3.2.2/src"
        )
        sys.path.insert(0, str(geak_src))
        try:
            from minisweagent.agents.heterogeneous.workload_guidance import (
                _build_workload_guidance,
            )
            from minisweagent.run.preprocess.baseline import build_baseline_metrics

            baseline = build_baseline_metrics(payload, include_all=True)
            guidance = _build_workload_guidance(
                {
                    "file_path": "candidate.py",
                    "kernel_name": "layernorm",
                    "kernel_type": "triton",
                },
                baseline,
            )
        finally:
            sys.path.remove(str(geak_src))
        self.assertEqual(baseline["bottleneck"], "memory")
        self.assertIn("HBM utilization: 66.0%", guidance)
        self.assertIn("Prefer First", guidance)


if __name__ == "__main__":
    unittest.main()
