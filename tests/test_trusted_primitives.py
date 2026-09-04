"""The controlled vendor-GEMM capability: granted by name, enforced on the AST.

Two things are being defended here. One is that an operator which did not ask
for a primitive cannot reach it -- the grant is per run, not a global loosening.
The other is that the grant buys exactly one fixed function and nothing else:
the whole-operator spellings, autograd and ``torch.compile`` stay forbidden
whether or not a capability was given, and a contraction performed anywhere
outside the trusted primitive is a violation either way.
"""

from __future__ import annotations

import textwrap
import unittest

import torch

from evograd.pipelines.shared.primitives import (
    REGISTRY,
    VENDOR_GEMM_SOURCE,
    PrimitiveViolation,
    catalogue,
    check_source,
    normalize,
    render_primitive_layer,
)

PURE_TRITON = """\
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
@triton.jit
def _k(X, Y, N: tl.constexpr):
    acc = tl.dot(tl.load(X), tl.load(Y))
    tl.store(Y, acc)


def op_forward_with_saved(x, w):
    out = torch.empty_like(x)
    _k[(1,)](x, out, 32)
    return out, (x,)


def op_backward_from_saved(dout, saved):
    (x,) = saved
    return (torch.empty_like(x),)
# EVOLVE-BLOCK-END
"""


def _with_body(body: str, *, trusted: str = "") -> str:
    """A minimal artifact whose evolvable pair contains ``body``."""
    return (
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        "# EVOLVE-BLOCK-START\n"
        "def op_forward_with_saved(x, w):\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
        + "\n\ndef op_backward_from_saved(dout, saved):\n    return (dout,)\n"
        "# EVOLVE-BLOCK-END\n" + trusted
    )


class TestCapabilityIsOptIn(unittest.TestCase):
    def test_the_same_call_is_accepted_with_the_grant_and_rejected_without(self):
        source = _with_body("return vendor_gemm(x, w.t()), (x,)",
                            trusted=VENDOR_GEMM_SOURCE)
        usage = check_source(source, allowed=("vendor_gemm",))
        self.assertEqual(usage["primitive_call_sites"], {"vendor_gemm": 1})
        self.assertTrue(usage["uses_trusted_primitives"])

        # Ungranted, the definition itself is already a violation ...
        with self.assertRaises(PrimitiveViolation) as caught:
            check_source(source, allowed=())
        self.assertIn("was not granted", str(caught.exception))
        # ... and so is the bare call with no definition in sight.
        with self.assertRaises(PrimitiveViolation) as caught:
            check_source(_with_body("return vendor_gemm(x, w.t()), (x,)"), allowed=())
        self.assertIn("did not grant", str(caught.exception))

    def test_a_lookalike_definition_cannot_borrow_the_trusted_name(self):
        # Same name, different body: every report downstream reads the name.
        forged = _with_body(
            "return vendor_gemm(x, w), (x,)",
            trusted="\n\ndef vendor_gemm(a, b):\n    return a\n",
        )
        with self.assertRaises(PrimitiveViolation):
            check_source(forged, allowed=())
        with self.assertRaises(PrimitiveViolation) as caught:
            check_source(forged, allowed=("vendor_gemm",))
        self.assertIn("not the trusted one", str(caught.exception))

    def test_the_trusted_body_may_not_move_inside_the_evolve_block(self):
        inside = (
            "import torch\n\n# EVOLVE-BLOCK-START\n"
            + VENDOR_GEMM_SOURCE
            + "\n\ndef op_forward_with_saved(x, w):\n"
            "    return vendor_gemm(x, w), (x,)\n"
            "\n\ndef op_backward_from_saved(dout, saved):\n    return (dout,)\n"
            "# EVOLVE-BLOCK-END\n"
        )
        with self.assertRaises(PrimitiveViolation) as caught:
            check_source(inside, allowed=("vendor_gemm",))
        self.assertIn("must be fixed", str(caught.exception))

    def test_an_unknown_capability_is_refused_rather_than_ignored(self):
        with self.assertRaises(PrimitiveViolation):
            normalize(("cublas_everything",))

    def test_the_catalogue_names_what_was_granted(self):
        self.assertEqual(catalogue(()), [])
        entry = catalogue(("vendor_gemm",))[0]
        self.assertEqual(entry["symbols"], ["vendor_gemm"])
        self.assertIn("cuBLAS", entry["summary"])


class TestForbiddenStaysForbidden(unittest.TestCase):
    """The grant is one function, not an amnesty."""

    def _rejects(self, body: str, needle: str):
        # Rejected with the capability and without it: the grant is one extra
        # function, never permission for the operation it happens to perform.
        for allowed in ((), ("vendor_gemm",)):
            trusted = VENDOR_GEMM_SOURCE if allowed else ""
            with self.assertRaises(PrimitiveViolation, msg=f"{body} @ {allowed}") as c:
                check_source(_with_body(body, trusted=trusted), allowed=allowed)
            self.assertIn(needle, str(c.exception))

    def test_direct_torch_contractions(self):
        self._rejects("return torch.matmul(x, w), (x,)", "torch.matmul")
        self._rejects("return torch.mm(x, w), (x,)", "torch.mm")
        self._rejects("return torch.einsum('ij,jk->ik', x, w), (x,)", "torch.einsum")
        self._rejects("return torch.bmm(x, w), (x,)", "torch.bmm")

    def test_the_matmul_operator_is_the_same_violation_as_the_function(self):
        self._rejects("return x @ w, (x,)", "'@'")

    def test_method_spellings_do_not_escape(self):
        self._rejects("return x.matmul(w), (x,)", "matmul")
        self._rejects("return x.mm(w), (x,)", "mm")

    def test_functional_and_whole_operator_spellings(self):
        self._rejects("return F.linear(x, w), (x,)", "F.linear")
        self._rejects("return torch.nn.functional.linear(x, w), (x,)",
                      "torch.nn.functional.linear")
        self._rejects("return torch.scaled_dot_product_attention(x, x, x), (x,)",
                      "scaled_dot_product_attention")
        self._rejects("return torch.rms_norm(x, w.shape, w), (x,)", "rms_norm")

    def test_autograd_and_compile(self):
        self._rejects("return torch.compile(lambda a: a)(x), (x,)", "torch.compile")
        self._rejects("return torch.autograd.grad(x, w), (x,)", "torch.autograd.grad")
        self._rejects("x.backward()\nreturn x, (x,)", "backward")

    def test_reference_and_library_imports(self):
        self._rejects("import transformers\nreturn x, (x,)", "transformers")
        self._rejects("from evograd.ops import get_op\nreturn x, (x,)", "evograd")

    def test_an_aliased_import_does_not_launder_the_call(self):
        self._rejects("from torch import matmul\nreturn matmul(x, w), (x,)",
                      "matmul")

    def test_triton_dot_is_the_kernels_own_arithmetic_and_stays_allowed(self):
        # `tl.dot` is what an evolved matmul kernel is made of; denying it would
        # forbid the search space rather than police it.
        usage = check_source(PURE_TRITON, allowed=())
        self.assertFalse(usage["uses_trusted_primitives"])


class TestExistingArtifactsAreUnaffected(unittest.TestCase):
    def test_a_pure_triton_artifact_validates_with_no_capability(self):
        usage = check_source(PURE_TRITON, allowed=())
        self.assertEqual(usage["allowed_primitives"], [])
        self.assertEqual(usage["primitive_call_sites"], {})

    def test_rendering_without_a_grant_adds_nothing(self):
        self.assertEqual(render_primitive_layer(()), "")
        self.assertIn("def vendor_gemm", render_primitive_layer(("vendor_gemm",)))

    def test_the_deployment_layer_records_the_grant_either_way(self):
        from evograd.ops import get_op
        from evograd.pipelines.shared.artifact import render_deployment_layer

        op = get_op("qwen3_qkv_norm_rope")
        plain = render_deployment_layer(op)
        hybrid = render_deployment_layer(op, allowed_primitives=("vendor_gemm",))
        self.assertIn("'allowed_primitives': []", plain)
        self.assertIn("'allowed_primitives': ['vendor_gemm']", hybrid)
        self.assertNotIn("def vendor_gemm", plain)
        self.assertIn("def vendor_gemm", hybrid)


class TestVendorGemmRuntime(unittest.TestCase):
    """The primitive itself: what it computes, and what it refuses to remember."""

    def setUp(self):
        namespace: dict = {"torch": torch}
        exec(compile(VENDOR_GEMM_SOURCE, "<vendor_gemm>", "exec"), namespace)
        self.gemm = namespace["vendor_gemm"]
        self.real_mm = torch.mm
        self.calls: list[tuple] = []

        def spy(a, b, *args, **kwargs):
            self.calls.append((tuple(a.shape), tuple(b.shape)))
            return self.real_mm(a, b, *args, **kwargs)

        torch.mm = spy
        self.addCleanup(setattr, torch, "mm", self.real_mm)

    def test_it_computes_the_gemm_and_reaches_the_vendor_route(self):
        a = torch.randn(8, 4)
        b = torch.randn(4, 6)
        out = self.gemm(a, b)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.allclose(out, self.real_mm(a, b)))

    def test_it_builds_no_autograd_graph_even_on_grad_requiring_inputs(self):
        a = torch.randn(8, 4, requires_grad=True)
        b = torch.randn(4, 6, requires_grad=True)
        out = self.gemm(a, b)
        self.assertIsNone(out.grad_fn)
        self.assertFalse(out.requires_grad)

    def test_a_transposed_argument_stays_a_view(self):
        w = torch.randn(6, 4)
        view = w.t()
        self.assertEqual(view.data_ptr(), w.data_ptr())
        self.assertFalse(view.is_contiguous())
        out = self.gemm(torch.randn(8, 4), view)
        self.assertEqual(out.shape, (8, 6))

    def test_changing_a_parameter_changes_the_result(self):
        a = torch.randn(8, 4)
        w = torch.randn(4, 6)
        first = self.gemm(a, w).clone()
        w.add_(1.0)                       # the optimiser's step, in miniature
        second = self.gemm(a, w)
        self.assertFalse(torch.allclose(first, second))

    def test_nothing_is_cached_between_calls(self):
        a, b = torch.randn(8, 4), torch.randn(4, 6)
        for expected in (1, 2, 3):
            self.gemm(a, b)
            self.assertEqual(len(self.calls), expected)

    def test_it_refuses_a_batched_or_mismatched_contraction(self):
        with self.assertRaises(ValueError):
            self.gemm(torch.randn(2, 8, 4), torch.randn(2, 4, 6))
        with self.assertRaises(ValueError):
            self.gemm(torch.randn(8, 4), torch.randn(5, 6))


class TestEveryGemmRunsInsideTheTimedRegion(unittest.TestCase):
    """A pair may not hoist a weight-dependent product out of the measured call."""

    def setUp(self):
        namespace: dict = {"torch": torch}
        exec(compile(VENDOR_GEMM_SOURCE, "<vendor_gemm>", "exec"), namespace)
        self.gemm = namespace["vendor_gemm"]
        self.real_mm = torch.mm
        self.count = 0

        def spy(a, b, *args, **kwargs):
            self.count += 1
            return self.real_mm(a, b, *args, **kwargs)

        torch.mm = spy
        self.addCleanup(setattr, torch, "mm", self.real_mm)

    def _pair(self):
        """A miniature pair in the shape a generated one takes."""
        gemm = self.gemm

        def forward_with_saved(x, w):
            return gemm(x, w.t()), (x, w)

        def backward_from_saved(dout, saved):
            x, w = saved
            return gemm(dout, w), gemm(dout.t(), x)

        return forward_with_saved, backward_from_saved

    def test_the_forward_and_the_explicit_backward_each_run_their_gemms(self):
        forward, backward = self._pair()
        x = torch.randn(8, 4)
        w = torch.randn(6, 4)

        before = self.count
        out, saved = forward(x, w)
        self.assertEqual(self.count - before, 1, "forward GEMM did not run")

        before = self.count
        dx, dw = backward(torch.randn_like(out), saved)
        self.assertEqual(self.count - before, 2, "backward GEMMs did not run")
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(dw.shape, w.shape)

    def test_no_gemm_runs_before_the_pair_is_called(self):
        self._pair()
        self.assertEqual(self.count, 0)

    def test_repeating_the_step_repeats_every_gemm(self):
        forward, backward = self._pair()
        x, w = torch.randn(8, 4), torch.randn(6, 4)
        for step in range(1, 4):
            out, saved = forward(x, w)
            backward(torch.randn_like(out), saved)
            self.assertEqual(self.count, 3 * step)

    def test_the_backward_gradients_match_autograd(self):
        # The pair differentiates itself; this is the check that it agrees with
        # the framework it is not allowed to call.
        forward, backward = self._pair()
        x = torch.randn(8, 4, dtype=torch.float64)
        w = torch.randn(6, 4, dtype=torch.float64)
        out, saved = forward(x, w)
        dout = torch.randn_like(out)
        dx, dw = backward(dout, saved)

        xa = x.clone().requires_grad_(True)
        wa = w.clone().requires_grad_(True)
        (self.real_mm(xa, wa.t()) * dout).sum().backward()
        self.assertTrue(torch.allclose(dx, xa.grad))
        self.assertTrue(torch.allclose(dw, wa.grad))


class TestRegistryShape(unittest.TestCase):
    def test_every_registered_primitive_declares_its_symbols_and_source(self):
        for name, spec in REGISTRY.items():
            self.assertEqual(spec.name, name)
            self.assertTrue(spec.symbols)
            for symbol in spec.symbols:
                self.assertIn(f"def {symbol}(", spec.source)
            self.assertTrue(spec.summary)


if __name__ == "__main__":
    unittest.main()
