"""The seven Qwen3.5 Level-2 tasks, on CPU.

Three claims per task:

* the declaration is coherent: its oracle and its production spelling agree on
  every correctness case, the eager pair passes the gate, and the tolerances are
  the provisional ones, said so;
* its canonical case is the configuration's (``hf_config`` provenance that
  re-derives) at B1 x T2048;
* ``capture.py`` takes a real invocation out of a tiny block -- config-derived,
  and captured from a tiny whole-model step -- whose shapes are exactly the
  declaration's, and the production spelling, run on the captured inputs,
  reproduces what the block computed **bitwise**. That is what makes
  ``runtime_forward`` *the* model's spelling rather than an equivalent of it.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.qwen3_5_4b.architecture import config_record
from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import capture_tools, manifest
from evograd.evaluation.tier3.patch import eager_pair_for
from evograd.opdecl import bind_shape, verify
from evograd.opdecl.baselines import verify_runtime_forward
from evograd.opdecl.models import QWEN3_5_4B, rederive_dims
from evograd.opdecl.oracle import resolve_runtime_forward

from tests.qwen3_5_4b._tiny import BATCH, FULL_LAYER, GDN_LAYER, SEQ, TINY_ARCH

_SOURCES: dict = {}


def _source(kind: str, dtype: str):
    key = (kind, dtype)
    if key not in _SOURCES:
        index = GDN_LAYER if kind == "linear_attention" else FULL_LAYER
        _SOURCES[key] = capture_tools.config_source(index, arch=TINY_ARCH, batch=BATCH, seq=SEQ,
                                                    dtype=dtype, seed=0)
    return _SOURCES[key]


class TestDeclarations(unittest.TestCase):
    def test_seven_sites_in_two_block_kinds(self):
        self.assertEqual(len(manifest.SITES), 7)
        self.assertEqual(set(manifest.SITES_BY_BLOCK["linear_attention"]),
                         {"gdn_in_proj", "gated_delta_rule", "gdn_norm_out",
                          "residual_rmsnorm", "swiglu_mlp"})
        self.assertEqual(set(manifest.SITES_BY_BLOCK["full_attention"]),
                         {"qkv_norm_partial_rope", "gated_attention", "residual_rmsnorm",
                          "swiglu_mlp"})
        self.assertEqual({s: manifest.expected_model_frequency(s) for s in manifest.SITES},
                         {"gdn_in_proj": 24, "gated_delta_rule": 24, "gdn_norm_out": 24,
                          "qkv_norm_partial_rope": 8, "gated_attention": 8,
                          "residual_rmsnorm": 32, "swiglu_mlp": 32})

    def test_canonical_cases_are_config_derived_at_b1(self):
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            with self.subTest(site=site):
                self.assertEqual(len(op.benchmark), 1)
                case = op.benchmark[0]
                self.assertEqual(case.dtype, "bfloat16")
                self.assertEqual(case.provenance.model, "qwen3_5_4b")
                self.assertEqual(case.provenance.free, {"batch": 1, "seq": 2048})
                self.assertEqual(case.dims, rederive_dims(case.provenance))
                self.assertEqual(case.dims, manifest.dims_for(site, QWEN3_5_4B, batch=1,
                                                              seq=2048))
                self.assertEqual(op.coverage, op.benchmark)
                self.assertIn("qwen3_5_4b_observed", op.benchmark_suites)

    def test_every_task_is_calibrated(self):
        """Run 20261011_calib: each declaration carries its generated block and
        points at it; nothing provisional is left."""
        for site in manifest.SITES:
            module = importlib.import_module(
                f"evograd.benchmark.topdown.qwen3_5_4b.levels.level2.{site}.task")
            with self.subTest(site=site):
                self.assertIs(module.PROVISIONAL_TOLERANCES, False)
                self.assertIs(module.CALIBRATED, True)
                self.assertEqual(module.op.tolerances, module.CALIBRATED_TOLERANCES)
                self.assertIs(module.op.tolerance_hook, module.CALIBRATED_HOOK)
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertIn("#: Calibrated: see the calibrated-tolerances block", source)
                self.assertNotIn("Not calibrated", source)
                self.assertIn("#: Calibrated for Qwen3.5-4B (run ", source)

    def test_oracle_and_production_agree_on_the_grid(self):
        for site, task in manifest.SITE_TASKS.items():
            with self.subTest(site=site):
                verify_runtime_forward(TASKS[task], device="cpu")

    def test_the_eager_pair_passes_the_gate(self):
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            with self.subTest(site=site):
                report = verify(op, eager_pair_for(op), device="cpu")
                self.assertTrue(report.ok, report.to_dict())

    def test_library_baselines_are_declared(self):
        self.assertIn("fla", TASKS["qwen3_5_gated_delta_rule"].performance_baselines)
        self.assertIn("fla", TASKS["qwen3_5_gdn_norm_out"].performance_baselines)
        self.assertIn("liger", TASKS["qwen3_5_residual_rmsnorm"].performance_baselines)
        self.assertIn("liger", TASKS["qwen3_5_swiglu_mlp"].performance_baselines)


class TestCapturedInvocations(unittest.TestCase):
    def _check_shapes(self, call, op, dims):
        for arg in op.args:
            if getattr(arg, "shape", None) is None:
                continue
            self.assertEqual(tuple(call.inputs[arg.name].shape), bind_shape(arg.shape, dims),
                             arg.name)
        for out in op.outputs:
            self.assertEqual(tuple(call.upstream[out.grad_name].shape),
                             bind_shape(out.shape, dims), out.grad_name)
            self.assertEqual(tuple(call.outputs[out.name].shape), bind_shape(out.shape, dims),
                             out.name)

    def test_every_site_captures_with_its_declared_shapes(self):
        record = config_record(TINY_ARCH, name="tiny")
        captures = manifest.captures()
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            dims = manifest.dims_for(site, record, batch=BATCH, seq=SEQ)
            for kind, sites in manifest.SITES_BY_BLOCK.items():
                if site not in sites:
                    continue
                with self.subTest(site=site, kind=kind):
                    call = captures[site](_source(kind, "bfloat16"))
                    self.assertEqual(call.task, task)
                    self.assertEqual(call.source, "config")
                    self._check_shapes(call, op, dims)

    def test_production_reproduces_the_block_bitwise(self):
        captures = manifest.captures()
        for dtype in ("float32", "bfloat16"):
            for site, task in manifest.SITE_TASKS.items():
                op = TASKS[task]
                production = resolve_runtime_forward(op)
                for kind, sites in manifest.SITES_BY_BLOCK.items():
                    if site not in sites:
                        continue
                    with self.subTest(site=site, kind=kind, dtype=dtype):
                        call = captures[site](_source(kind, dtype))
                        args = [call.inputs.get(a.name, getattr(a, "default", None))
                                for a in op.args]
                        out = production(*args)
                        out = out if isinstance(out, tuple) else (out,)
                        for name, value in zip(op.output_names, out):
                            self.assertTrue(torch.equal(value, call.outputs[name]), name)
                            self.assertEqual(value.stride(), call.outputs[name].stride(), name)

    def test_v_reaches_the_delta_rule_as_a_strided_view(self):
        call = manifest.captures()["gated_delta_rule"](_source("linear_attention", "bfloat16"))
        v = call.inputs["v"]
        self.assertFalse(v.is_contiguous())
        conv_dim = config_record(TINY_ARCH, name="tiny").conv_dim
        self.assertEqual(v.stride(), (conv_dim * SEQ, 1, v.shape[-1] * SEQ, SEQ))
        self.assertEqual(call.inputs["g"].dtype, torch.float32)
        self.assertEqual(call.inputs["beta"].dtype, torch.bfloat16)

    def test_sites_capture_from_a_captured_layer_artifact(self):
        """The captured source (a Level-3 artifact from a tiny whole-model step)
        feeds the same captures, and production still reproduces it bitwise."""
        from evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest import run_harvest
        from evograd.benchmark.topdown.qwen3_5_4b.layer.capture import (
            run_capture, write_captures)
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
        import json

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=SEQ)
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            write_captures(run_capture(spec, manifest_path=harvest,
                                       layers=[GDN_LAYER, FULL_LAYER]), Path(tmp))
            record = config_record(TINY_ARCH, name="tiny")
            captures = manifest.captures()
            for kind, index in (("linear_attention", GDN_LAYER), ("full_attention", FULL_LAYER)):
                source = capture_tools.captured_source(Path(tmp) / f"layer{index}.pt",
                                                       canonical=False)
                self.assertEqual((source.kind, source.source), (kind, "captured"))
                for site in manifest.SITES_BY_BLOCK[kind]:
                    op = TASKS[manifest.SITE_TASKS[site]]
                    with self.subTest(site=site, kind=kind):
                        call = captures[site](source)
                        self.assertEqual(call.source, "captured")
                        self._check_shapes(call, op,
                                           manifest.dims_for(site, record, batch=1, seq=SEQ))
                        out = resolve_runtime_forward(op)(
                            *[call.inputs.get(a.name, getattr(a, "default", None))
                              for a in op.args])
                        out = out if isinstance(out, tuple) else (out,)
                        for name, value in zip(op.output_names, out):
                            self.assertTrue(torch.equal(value, call.outputs[name]), name)


class TestProductionCompiles(unittest.TestCase):
    """Every production spelling traces whole under ``torch.compile(fullgraph=True)``
    (Dynamo's ``eager`` backend, no toolchain needed): the Tier-2 compile baseline,
    the Tier-3 per-site compile provider and the trusted-compile control all
    compile them."""

    def test_fullgraph(self):
        from evograd.opdecl.inputs import as_output_tuple, make_case_inputs

        for task in manifest.TASKS:
            with self.subTest(task=task):
                op = TASKS[task]
                workload = next(c for c in op.correctness if c.dtype == "float32")
                values = make_case_inputs(op, workload, device="cpu")
                args = [values.get(a.name, getattr(a, "default", None)) for a in op.args]
                production = resolve_runtime_forward(op)
                torch._dynamo.reset()
                compiled = torch.compile(production, backend="eager", fullgraph=True,
                                         dynamic=False)
                for got, want in zip(as_output_tuple(op, compiled(*args)),
                                     as_output_tuple(op, production(*args))):
                    self.assertTrue(torch.equal(got, want))
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
