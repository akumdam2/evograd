"""MegaFold's contracts, site adapters and recorded-call subset (CPU, no MegaFold import).

* the two MegaFold contracts are, bit for bit, what MegaFold's own PyTorch code
  computes (``Attend``'s default path on strided projection views,
  ``LayernormLinear``'s torch path with autocast's casts), gradients included;
* every site adapter hands its kernel the contract's arguments and gives the
  caller what the module it replaced returns -- the same values and the same
  dtype -- outside autocast and under a bfloat16 one;
* the corpus keeps, per case and order, the first, middle and last call, or the
  first only above the size limit, and coverage holds a case to that count.

Autocast is simulated by fixing the adapters' view of it (``_autocast_dtype``):
CPU autocast does not run the CUDA op lists MegaFold trains under, so the
native side is spelled with the casts CUDA autocast makes.
"""

from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn

from evograd.benchmark.topdown.megafold.contracts import (
    megafold_evoattention_forward,
    megafold_layernorm_linear_forward,
)
from evograd.evaluation.tier3.workloads.megafold import workload as W

DTYPES = (torch.float32, torch.bfloat16)


def attend(q, k, v, mask, attn_bias):
    """``megafold/model/attention.py`` ``Attend.forward``, default path, as CUDA autocast runs it."""
    dtype = q.dtype
    q = q * q.shape[-1] ** -0.5
    sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
    sim = sim + attn_bias
    sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(sim.dtype).max)
    attn = sim.float().softmax(dim=-1).to(dtype)  # softmax is on autocast's float32 list
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


def projections(dtype, b=1, s=3, n=17, h=4, d=8, seed=0):
    """MegaFold's ``[b, h, n, d]`` q, k, v: split-head views of one fused projection,
    the bias repeated over the ``s`` rows, and a key mask."""
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(b * s, n, 3 * h * d, generator=g).to(dtype)
    q, k, v = (t.reshape(b * s, n, h, d).permute(0, 2, 1, 3) for t in proj.chunk(3, dim=-1))
    bias = torch.randn(b, h, n, n, generator=g).to(dtype)
    mask = torch.rand(b * s, n, generator=g) > 0.2
    mask[:, 0] = True
    return q, k, v, bias, mask


def repeat_rows(bias, s):
    b, h, n, _ = bias.shape
    return bias[:, None].expand(b, s, h, n, n).reshape(b * s, h, n, n)


def as_declared(t, b, s):
    return t.permute(0, 2, 1, 3).unflatten(0, (b, s))


class TheContractsAreMegaFoldsComputation(unittest.TestCase):
    def test_attention_forward_and_backward(self):
        b, s = 1, 3
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                q, k, v, bias, mask = projections(dtype, b=b, s=s)
                live = attend(q, k, v, mask, repeat_rows(bias, s))
                out = megafold_evoattention_forward(*(as_declared(t, b, s) for t in (q, k, v)),
                                                    mask.view(b, s, -1), bias[:, None])
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(out, as_declared(live, b, s)))

                leaves = [t.detach().requires_grad_() for t in (q, k, v, bias)]
                grad = torch.randn(live.shape).to(dtype)
                attend(*leaves[:3], mask, repeat_rows(leaves[3], s)).backward(grad)
                ours = [as_declared(t.detach(), b, s).requires_grad_() for t in (q, k, v)]
                ours.append(bias.detach()[:, None].requires_grad_())
                megafold_evoattention_forward(*ours[:3], mask.view(b, s, -1), ours[3]).backward(
                    as_declared(grad, b, s))
                for mine, theirs in zip(ours[:3], leaves[:3]):
                    self.assertTrue(torch.equal(mine.grad, as_declared(theirs.grad, b, s)))
                self.assertTrue(torch.equal(ours[3].grad[:, 0], leaves[3].grad))

    def test_layernorm_linear_is_the_torch_path_under_autocast(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 5, 7, 16, generator=g)
        w, b = 1 + 0.1 * torch.randn(16, generator=g), 0.1 * torch.randn(16, generator=g)
        weight = torch.randn(16, 24, generator=g)
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                live = F.linear(F.layer_norm(x, (16,), w, b).to(dtype), weight.T.to(dtype))
                out = megafold_layernorm_linear_forward(x.reshape(-1, 16), w, b, weight.to(dtype))
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(out, live.reshape(-1, 24)))


def autocast_as(dtype):
    """The adapters' view of an enclosing autocast."""
    return mock.patch.object(W, "_autocast_dtype", lambda _t: dtype)


class TheAdaptersAreDropIns(unittest.TestCase):
    def test_layernorm(self):
        norm = nn.LayerNorm(16)
        with torch.no_grad():
            norm.weight.add_(0.1), norm.bias.add_(0.2)
        patched = W._MegaFoldPatchedLayerNorm(norm, lambda x, w, b, eps: F.layer_norm(x, (16,), w, b, eps))
        x = torch.randn(2, 3, 5, 16)
        self.assertTrue(torch.equal(patched(x), norm(x)))
        with autocast_as(torch.bfloat16):  # autocast runs layer_norm in float32 and returns it
            xb = x.bfloat16()
            out = patched(xb)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, F.layer_norm(xb.float(), (16,), norm.weight, norm.bias)))

    def _layernorm_linear(self):
        g = torch.Generator().manual_seed(2)
        p = lambda *s: nn.Parameter(torch.randn(*s, generator=g))  # noqa: E731
        return SimpleNamespace(WEIGHT=p(16), BIAS=p(16), linear_weight=p(16, 24), linear_bias=None,
                               K=16, N=24, has_layernorm_bias=True, has_linear_bias=False)

    def test_layernorm_linear(self):
        original = self._layernorm_linear()
        eager = W._EagerLayernormLinear(original)
        patched = W._PatchedLayernormLinear(original, megafold_layernorm_linear_forward)
        x = torch.randn(2, 3, 5, 16)
        self.assertTrue(torch.equal(patched(x), eager(x)))
        with autocast_as(torch.bfloat16):
            for xin in (x, x.bfloat16()):  # a bfloat16 input is cast up for the LayerNorm
                with self.subTest(x=xin.dtype):
                    out = patched(xin)
                    live = F.linear(F.layer_norm(xin.float(), (16,), original.WEIGHT, original.BIAS)
                                    .bfloat16(), original.linear_weight.T.bfloat16())
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(out, live))

    def test_parameters_keep_the_original_names(self):
        """A gradient compared by name must find the same parameter in either tree."""
        lnl = self._layernorm_linear()
        names = lambda m: sorted(n for n, _ in m.named_parameters())  # noqa: E731
        self.assertEqual(names(W._PatchedLayernormLinear(lnl, megafold_layernorm_linear_forward)),
                         names(W._EagerLayernormLinear(lnl)))
        ff = nn.Sequential(nn.Linear(16, 32, bias=False), W._EagerSwiGLU(), nn.Linear(16, 16, bias=False))
        original = SimpleNamespace(ff=ff)
        self.assertEqual(names(W._PatchedSwiGLUTransition(original, lambda a, b: F.silu(a) * b)),
                         names(W._EagerTransition(original)))

    def test_layernorm_linear_gradient_reaches_the_float32_weight(self):
        original = self._layernorm_linear()
        patched = W._PatchedLayernormLinear(original, megafold_layernorm_linear_forward)
        with autocast_as(torch.bfloat16):
            patched(torch.randn(4, 16)).float().sum().backward()
        self.assertEqual(original.linear_weight.grad.dtype, torch.float32)

    def test_transition(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                ff = nn.Sequential(nn.Linear(16, 32, bias=False), W._EagerSwiGLU(),
                                   nn.Linear(16, 16, bias=False)).to(dtype)
                original = SimpleNamespace(ff=ff)
                seen = []

                def kernel(a, b):
                    seen.append((a.shape, a.stride(), b.stride()))
                    return F.silu(a) * b

                patched = W._PatchedSwiGLUTransition(original, kernel)
                x = torch.randn(2, 3, 16).to(dtype)
                self.assertTrue(torch.equal(patched(x), W._EagerTransition(original)(x)))
                # the halves are [rows, cols] views of the projection, not copies
                self.assertEqual(seen[0], (torch.Size([6, 16]), (32, 1), (32, 1)))

    def test_attention(self):
        original = SimpleNamespace(is_local_attn=False, scale=None, enable_attn_softclamp=False,
                                   dropout=0.0)
        patched = W._PatchedAttend(original, megafold_evoattention_forward)
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                q, k, v, bias, mask = projections(dtype, b=1, s=3)
                rows = repeat_rows(bias, 3)
                out = patched(q, k, v, mask=mask, attn_bias=rows, batch_size=1)
                live = attend(q, k, v, mask, rows)
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(out, live))

    def test_attention_refuses_a_bias_in_another_dtype(self):
        original = SimpleNamespace(is_local_attn=False, scale=None, enable_attn_softclamp=False,
                                   dropout=0.0)
        patched = W._PatchedAttend(original, megafold_evoattention_forward)
        q, k, v, bias, mask = projections(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "one dtype"):
            patched(q, k, v, mask=mask, attn_bias=repeat_rows(bias, 3).float(), batch_size=1)


class KeptCallsMoveToTheHostAsRecorded(unittest.TestCase):
    def test_layouts_offsets_and_shared_storage_survive(self):
        from evograd.verification.integrations.megafold.capture import _host

        buf = torch.randn(9, 32)
        k, v = buf[1:, :16], buf[1:, 16:]  # two chunks of one projection, offset by a row
        moved = _host({"k": k, "v": v, "eps": 1e-5})
        for name, t in (("k", k), ("v", v)):
            self.assertEqual((moved[name].shape, moved[name].stride()), (t.shape, t.stride()))
            self.assertTrue(torch.equal(moved[name], t))
        self.assertEqual(moved["k"].untyped_storage().data_ptr(), moved["v"].untyped_storage().data_ptr())
        self.assertEqual(moved["v"].storage_offset() - moved["k"].storage_offset(), 16)
        self.assertEqual(moved["eps"], 1e-5)

    def test_a_small_view_of_a_large_storage_gets_its_own(self):
        from evograd.verification.integrations.megafold.capture import _host

        rows = torch.randn(1, 4, 8, 8).expand(1024, 4, 8, 8).contiguous()  # a bias repeated over rows
        first = rows.reshape(1, 1024, 4, 8, 8)[:, :1]
        moved = _host({"pair_bias": first})["pair_bias"]
        self.assertTrue(torch.equal(moved, first))
        self.assertEqual(moved.stride(), first.stride())
        self.assertLess(moved.untyped_storage().nbytes(), rows.untyped_storage().nbytes() // 100)


class TheRecordedSubset(unittest.TestCase):
    def case(self, frequency, dims=None, task="megafold_evoattention", dtype="bfloat16"):
        from evograd.opdecl.activity import Workload
        from evograd.verification.integrations.base import Case

        dims = dims or dict(B=1, S=1, H=16, N=384, D=64)
        return Case(f"{task}/c", task, 2, Workload(dims=dims, dtype=dtype), frequency)

    def test_first_middle_and_last(self):
        from evograd.verification.integrations.megafold.capture import recorded_indices

        self.assertEqual(recorded_indices(self.case(48)), (0, 23, 47))
        self.assertEqual(recorded_indices(self.case(2)), (0, 1))
        self.assertEqual(recorded_indices(self.case(1)), (0,))

    def test_one_call_above_the_limit(self):
        from evograd.verification.integrations.megafold.capture import (
            RECORD_ONE_ABOVE,
            call_bytes,
            recorded_indices,
        )

        big = self.case(4, dims=dict(rows=12097 * 384, cols=512), task="swiglu")
        self.assertGreater(call_bytes(big), RECORD_ONE_ABOVE)
        self.assertEqual(recorded_indices(big), (0,))

    def test_coverage_holds_a_case_to_its_recorded_count(self):
        from dataclasses import replace

        from evograd.verification.integrations.base import coverage

        subset = replace(self.case(48), recorded=3)
        payloads = [{"case_id": subset.case_id, "batch": {"order": o}} for o in (101, 202) for _ in range(3)]
        self.assertTrue(coverage([subset], payloads)["complete"])
        self.assertFalse(coverage([replace(subset, recorded=None)], payloads)["complete"])

    def test_the_integration_records_a_subset_and_has_model_gate2_without_gate3(self):
        from evograd.verification.integrations import load
        from evograd.verification.integrations.megafold.capture import recorded_calls

        model = load("megafold")
        self.assertIs(model.recorded_calls, recorded_calls)
        self.assertTrue(model.model_gates)
        self.assertFalse(model.gate3_applies)
        self.assertEqual(model.objective.primary, "distogram_logits")


class EachCaseHasItsSite(unittest.TestCase):
    """The registry's sites carry the configurations the harvest observed there."""

    def setUp(self):
        from evograd.benchmark.topdown import load_snapshot

        snapshot = load_snapshot("megafold")
        self.configurations = [c for level in ("level1", "tasks") for entry in snapshot[level].values()
                               for c in entry["configurations"]]

    def test_every_observed_level2_case_is_placed_where_it_was_observed(self):
        from evograd.verification.integrations import load
        from evograd.verification.integrations.base import case_component, observed_cases
        from evograd.verification.integrations.gates import site_for_case

        observed_at = {c["provenance"]["component"]: c["provenance"]["note"] for c in self.configurations}
        cases = [c for c in observed_cases(load("megafold")) if c.level == 2]
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.case_id):
                self.assertEqual(site_for_case(W.MEGAFOLD_SITES, case), observed_at[case_component(case)])
        self.assertEqual(sum(len(W.MEGAFOLD_SITES.require(s).preflight) for s in W.MEGAFOLD_SITES),
                         len(self.configurations))

    def test_no_case_lies_where_no_gradient_reaches(self):
        for c in self.configurations:
            for path in c["module_paths"]:
                self.assertNotIn(path.split(".", 1)[0], W.NOT_SITES, path)


class CandidatesTakeTheirOwnConfiguration(unittest.TestCase):
    """At a site observed at several configurations a candidate gets its own
    configuration's calls; every other call runs the model reference."""

    def setUp(self):
        from evograd.evaluation.tier3.patch import Site, SiteRegistry
        from evograd.opdecl.activity import Workload
        from evograd.verification.integrations.base import Case
        from evograd.verification.integrations.megafold.reference import reference_op

        task = "megafold_layernorm_linear"
        self.workloads = (Workload(dims=dict(M=6, K=8, N=3), dtype="float32"),
                          Workload(dims=dict(M=5, K=8, N=4), dtype="float32"))
        self.cases = [Case(f"{task}/c{i}", task, 2, w, 1) for i, w in enumerate(self.workloads)]
        self.model = SimpleNamespace(reference_op=reference_op)
        self.registry = lambda preflight: SiteRegistry(
            "megafold", (Site("layernorm_linear", task, lambda *a: None, preflight=preflight),))

    def args(self, workload):
        m, k, n = (workload.dims[d] for d in "MKN")
        return torch.randn(m, k), torch.rand(k) + 0.5, torch.randn(k), torch.randn(k, n), 1e-5

    def site(self, preflight, calls):
        from evograd.verification.integrations.gates import by_configuration

        return by_configuration(self.model, self.registry(preflight), "layernorm_linear", calls)

    def test_each_call_goes_by_its_configuration(self):
        seen = []
        call = self.site(self.workloads, [(self.cases[0], lambda *a: seen.append(a) or "candidate")])
        self.assertEqual(call(*self.args(self.workloads[0])), "candidate")
        other = self.args(self.workloads[1])
        self.assertTrue(torch.equal(call(*other), megafold_layernorm_linear_forward(*other)))
        x, *rest = self.args(self.workloads[0])
        transposed = torch.randn(x.shape[1], x.shape[0]).T  # a layout the case does not record
        self.assertTrue(torch.equal(call(transposed, *rest),
                                    megafold_layernorm_linear_forward(transposed, *rest)))
        self.assertEqual(len(seen), 1)

    def test_a_site_with_one_configuration_holds_its_candidate_as_is(self):
        candidate = lambda *a: "candidate"  # noqa: E731
        self.assertIs(self.site(self.workloads[:1], [(self.cases[0], candidate)]), candidate)
        with self.assertRaises(ValueError):
            self.site(self.workloads[:1], [(self.cases[0], candidate), (self.cases[1], candidate)])

    def test_a_configuration_the_site_never_saw_is_refused(self):
        from evograd.opdecl.activity import Workload

        unseen = (self.workloads[1], Workload(dims=dict(M=7, K=8, N=3), dtype="float32"))
        with self.assertRaises(ValueError):
            self.site(unseen, [(self.cases[0], lambda *a: None)])


class ControlsReplaceWhatTheCandidateReplaces(unittest.TestCase):
    """A model-scope control at a site whose candidate takes one configuration
    replaces that configuration's calls only, and the policy binds which."""

    def setUp(self):
        from evograd.evaluation.tier3.patch import Site, SiteRegistry
        from evograd.opdecl.activity import Workload
        from evograd.verification.gate2.adapters.model import WorkloadModelScope

        task = "megafold_layernorm_linear"
        self.workloads = (Workload(dims=dict(M=6, K=8, N=3), dtype="float32"),
                          Workload(dims=dict(M=5, K=8, N=4), dtype="float32"))
        registry = SiteRegistry("megafold", (Site("layernorm_linear", task, lambda *a: None,
                                                  preflight=self.workloads),))
        workload = SimpleNamespace(site_registry=registry, describe=lambda: {"workload": "test"})
        self.scope = WorkloadModelScope(
            workload, native_sites={"layernorm_linear": megafold_layernorm_linear_forward},
            replaced={"layernorm_linear": self.workloads[:1]})

    def args(self, workload):
        m, k, n = (workload.dims[d] for d in "MKN")
        return torch.randn(m, k), torch.rand(k) + 0.5, torch.randn(k), torch.randn(k, n), 1e-5

    def test_the_compile_control_takes_the_replaced_configuration_only(self):
        from evograd.verification.gate2.calibration import compile_control

        control, graphs = compile_control(self.scope, ("layernorm_linear",), backend="eager")
        site = control.sites["layernorm_linear"]
        site(*self.args(self.workloads[0]))
        self.assertEqual(graphs["layernorm_linear"], 1)
        # Another configuration's call would need a graph of its own if it were compiled.
        other = self.args(self.workloads[1])
        self.assertTrue(torch.equal(site(*other), megafold_layernorm_linear_forward(*other)))
        self.assertEqual(graphs["layernorm_linear"], 1)

    def test_the_replaced_configurations_are_bound(self):
        replaced = self.scope.identity()["replaced_configurations"]
        self.assertEqual(replaced, {"layernorm_linear": [
            {"dims": dict(M=6, K=8, N=3), "dtype": "float32", "strides": []}]})

    def test_policy_files_are_kept_apart_by_configuration(self):
        from evograd.verification.common.execution import candidate
        from evograd.verification.integrations.gates import _patch_name

        whole = candidate("candidate", {"layernorm_linear": None}, programs={})
        one = candidate("candidate", {"layernorm_linear": None}, programs={},
                        configurations={"layernorm_linear": ["megafold_layernorm_linear/a"]})
        other = candidate("candidate", {"layernorm_linear": None}, programs={},
                          configurations={"layernorm_linear": ["megafold_layernorm_linear/b"]})
        self.assertEqual(_patch_name(whole), "layernorm_linear")
        self.assertEqual(len({_patch_name(whole), _patch_name(one), _patch_name(other)}), 3)


class AcceptanceWithoutGate3(unittest.TestCase):
    """A model whose Gate 3 does not apply accepts a Level-2 candidate on model Gate 2."""

    def entry(self, **gates):
        return {"level": 2, "gate1_payloads": "passed", "gate2_operator": "passed", **gates}

    def test_level2(self):
        from evograd.verification.integrations.gates import NOT_APPLICABLE, accepted

        without = SimpleNamespace(model_gates=True, gate3_applies=False)
        full = SimpleNamespace(model_gates=True, gate3_applies=True)
        self.assertTrue(accepted(without, self.entry(gate2_model="passed", gate3=NOT_APPLICABLE)))
        self.assertFalse(accepted(without, self.entry(gate2_model="failed", gate3=NOT_APPLICABLE)))
        self.assertFalse(accepted(full, self.entry(gate2_model="passed", gate3=NOT_APPLICABLE)))
        self.assertTrue(accepted(full, self.entry(gate2_model="passed", gate3="passed")))


class TheWarmStartIsItsModelsOnly(unittest.TestCase):
    def test_load_checks_the_model_and_hashes_the_file(self):
        import hashlib
        import tempfile
        from pathlib import Path

        from evograd.evaluation.tier3.workloads.common.init_state import SCHEMA, load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save({"schema": SCHEMA, "model": "megafold", "recipe": {"steps": 2},
                        "state_dict": {"w": torch.ones(2)}}, path)
            state, identity = load(path, "megafold")
            self.assertTrue(torch.equal(state["w"], torch.ones(2)))
            self.assertEqual(identity, {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "recipe": {"steps": 2}})
            with self.assertRaises(ValueError):
                load(path, "dit_s_2")


if __name__ == "__main__":
    unittest.main()
