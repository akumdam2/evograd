"""The four OLMoE Level-2 tasks, on CPU.

Three claims per task:

* the declaration is coherent: its oracle and its production spelling agree on
  every correctness case, the eager pair passes the gate, and the tolerances are
  the provisional ones, said so;
* its canonical case is the configuration's (``hf_config`` provenance that
  re-derives) at B2 x T2048;
* ``capture.py`` takes a real invocation out of a tiny layer -- config-derived,
  and captured from a tiny whole-model step -- whose shapes are exactly the
  declaration's, and the production spelling, run on the captured inputs,
  reproduces what the layer computed **bitwise**. For the experts site that is
  what makes the grouped-GEMM spelling *the* model's kernel rather than an
  equivalent of it.
"""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.benchmark import TASKS
from evograd.benchmark.topdown.olmoe_1b_7b.architecture import config_record
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import capture_tools, manifest
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.task_tools import PROVISIONAL
from evograd.evaluation.tier3.patch import eager_pair_for
from evograd.opdecl import bind_shape, verify
from evograd.opdecl.baselines import verify_runtime_forward
from evograd.opdecl.models import OLMOE_1B_7B, rederive_dims
from evograd.opdecl.oracle import resolve_runtime_forward

from tests.olmoe_1b_7b._tiny import BATCH, LAYER, SEQ, TINY_ARCH

_SOURCES: dict = {}


def _source(dtype: str, experts: str = "grouped_mm"):
    key = (dtype, experts)
    if key not in _SOURCES:
        _SOURCES[key] = capture_tools.config_source(LAYER, arch=TINY_ARCH, batch=BATCH, seq=SEQ,
                                                    dtype=dtype, seed=0,
                                                    experts_implementation=experts)
    return _SOURCES[key]


class TestDeclarations(unittest.TestCase):
    def test_four_sites_once_per_layer(self):
        self.assertEqual(manifest.SITES, ("qkv_norm_rope", "attention", "residual_rmsnorm",
                                          "moe_experts"))
        self.assertEqual(manifest.SITE_TASKS["moe_experts"], "olmoe_moe_experts")
        self.assertEqual({s: manifest.expected_model_frequency(s) for s in manifest.SITES},
                         {s: 16 for s in manifest.SITES})
        self.assertIn("router", manifest.DEFERRED)

    def test_canonical_cases_are_config_derived_at_b2(self):
        for site, task in manifest.SITE_TASKS.items():
            op = TASKS[task]
            with self.subTest(site=site):
                self.assertEqual(len(op.benchmark), 1)
                case = op.benchmark[0]
                self.assertEqual(case.dtype, "bfloat16")
                self.assertEqual(case.provenance.model, "olmoe_1b_7b")
                self.assertEqual(case.provenance.free, {"batch": 2, "seq": 2048})
                self.assertEqual(case.dims, rederive_dims(case.provenance))
                self.assertEqual(case.dims, manifest.dims_for(site, OLMOE_1B_7B, batch=2,
                                                              seq=2048))
                self.assertEqual(op.coverage, op.benchmark)
                self.assertIn("olmoe_1b_7b_observed", op.benchmark_suites)

    def test_every_task_says_it_is_provisional(self):
        for site in manifest.SITES:
            module = importlib.import_module(
                f"evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.{site}.task")
            with self.subTest(site=site):
                self.assertIs(module.PROVISIONAL_TOLERANCES, True)
                self.assertEqual(module.op.tolerances, PROVISIONAL)
                self.assertFalse(getattr(module, "CALIBRATED", False))

    def test_the_experts_contract(self):
        op = TASKS["olmoe_moe_experts"]
        self.assertEqual(op.output_names, ("out",))
        self.assertEqual(op.grad_names(), ("dx", "dgate_up_proj", "ddown_proj", "dtop_k_weights"))
        index = next(a for a in op.args if a.name == "top_k_index")
        self.assertEqual(index.dtype, "int64")
        self.assertEqual(op.benchmark[0].dims,
                         {"T": 4096, "H": 2048, "E": 64, "G": 2048, "I": 1024, "K": 8})

    def test_the_qkv_contract_is_full_width(self):
        op = TASKS["olmoe_qkv_norm_rope"]
        shapes = {a.name: a.shape for a in op.args}
        self.assertEqual(shapes["q_norm_weight"], "[QO]")
        self.assertEqual(shapes["k_norm_weight"], "[KVO]")
        self.assertEqual(op.output_names, ("q", "k", "v"))
        self.assertEqual(op.grad_names(), ("dx", "dq_weight", "dk_weight", "dv_weight",
                                           "dq_norm_weight", "dk_norm_weight"))

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

    def test_the_experts_production_kernels_agree(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2.moe_experts.reference import (
            moe_experts_forward_production, moe_experts_forward_ref)
        from evograd.opdecl.inputs import make_case_inputs

        op = TASKS["olmoe_moe_experts"]
        for case in op.correctness:
            values = make_case_inputs(op, case, device="cpu")
            args = [values[a.name] for a in op.args]
            with self.subTest(dims=case.dims, dtype=case.dtype):
                grouped = moe_experts_forward_production(*args)
                eager = moe_experts_forward_production(*args, experts_implementation="eager")
                oracle = moe_experts_forward_ref(*args)
                tol = 1e-5 if case.dtype == "float32" else 5e-2
                self.assertTrue(torch.allclose(grouped.float(), eager.float(), atol=tol, rtol=tol))
                self.assertTrue(torch.allclose(grouped.float(), oracle.float(), atol=tol, rtol=tol))


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
            with self.subTest(site=site):
                call = captures[site](_source("bfloat16"))
                self.assertEqual(call.task, task)
                self.assertEqual(call.source, "config")
                self._check_shapes(call, op, dims)

    def test_production_reproduces_the_layer_bitwise(self):
        captures = manifest.captures()
        for dtype in ("float32", "bfloat16"):
            for site, task in manifest.SITE_TASKS.items():
                op = TASKS[task]
                production = resolve_runtime_forward(op)
                with self.subTest(site=site, dtype=dtype):
                    call = captures[site](_source(dtype))
                    args = [call.inputs.get(a.name, getattr(a, "default", None))
                            for a in op.args]
                    out = production(*args)
                    out = out if isinstance(out, tuple) else (out,)
                    for name, value in zip(op.output_names, out):
                        self.assertTrue(torch.equal(value, call.outputs[name]), name)
                        self.assertEqual(value.stride(), call.outputs[name].stride(), name)

    def test_the_experts_capture_carries_the_routers_choice(self):
        call = manifest.captures()["moe_experts"](_source("bfloat16"))
        index, weights = call.inputs["top_k_index"], call.inputs["top_k_weights"]
        self.assertEqual(index.dtype, torch.int64)
        self.assertEqual(weights.dtype, torch.bfloat16)
        self.assertEqual(tuple(index.shape), (BATCH * SEQ, 2))
        self.assertTrue(bool((index < 4).all()))
        self.assertTrue(bool((index[:, 0] != index[:, 1]).all()), "distinct experts per token")
        self.assertTrue(bool((weights.float().sum(-1) < 1).all()), "not renormalized")
        self.assertEqual(tuple(call.inputs["gate_up_proj"].shape), (4, 64, 64))
        self.assertTrue(call.inputs["x"].is_contiguous())

    def test_v_reaches_sdpa_as_a_head_major_view(self):
        call = manifest.captures()["attention"](_source("bfloat16"))
        v = call.inputs["v"]
        self.assertFalse(v.is_contiguous())
        self.assertEqual(v.stride(), (SEQ * 2 * 16, 16, 2 * 16, 1))

    def test_sites_capture_from_a_captured_layer_artifact(self):
        """The captured source (a Level-3 artifact from a tiny whole-model step)
        feeds the same captures, and production still reproduces it bitwise."""
        from evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest import run_harvest
        from evograd.benchmark.topdown.olmoe_1b_7b.layer.capture import (
            run_capture, write_captures)
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.spec import CANONICAL

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=SEQ)
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            write_captures(run_capture(spec, manifest_path=harvest, layers=[LAYER]), Path(tmp))
            record = config_record(TINY_ARCH, name="tiny")
            captures = manifest.captures()
            source = capture_tools.captured_source(Path(tmp) / f"layer{LAYER}.pt",
                                                   canonical=False)
            self.assertEqual((source.kind, source.source), ("decoder_layer", "captured"))
            for site in manifest.SITES:
                op = TASKS[manifest.SITE_TASKS[site]]
                with self.subTest(site=site):
                    call = captures[site](source)
                    self.assertEqual(call.source, "captured")
                    self._check_shapes(call, op, manifest.dims_for(site, record, batch=1, seq=SEQ))
                    out = resolve_runtime_forward(op)(
                        *[call.inputs.get(a.name, getattr(a, "default", None)) for a in op.args])
                    out = out if isinstance(out, tuple) else (out,)
                    for name, value in zip(op.output_names, out):
                        self.assertTrue(torch.equal(value, call.outputs[name]), name)


class TestProductionCompiles(unittest.TestCase):
    """Every production spelling traces under ``torch.compile`` (Dynamo's
    ``eager`` backend, no toolchain needed): the Tier-2 compile baseline, the
    Tier-3 per-site compile provider and the trusted-compile control all
    compile them. The experts site traces whole too: its grouped GEMM goes
    through Transformers' custom-op fallback under Dynamo, and the shim the
    production spelling carries is a plain class Dynamo can read."""

    def test_compiles(self):
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
                    self.assertTrue(torch.allclose(got.float(), want.float(), atol=1e-6,
                                                   rtol=1e-6))
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
