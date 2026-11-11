"""The controlled vendor-GEMM capability: granted by name, enforced on the AST.

Two things are being defended here. One is that an operator which did not ask
for a primitive cannot reach it -- the grant is per run, not a global loosening.
The other is that the grant buys exactly one fixed function and nothing else:
the whole-operator spellings, autograd and ``torch.compile`` stay forbidden
whether or not a capability was given, and a contraction performed anywhere
outside the trusted primitive is a violation either way.
"""

from __future__ import annotations

import re
import textwrap
import unittest

import torch

from evograd.pipelines.shared.primitives import (
    REGISTRY,
    VENDOR_GEMM_SOURCE,
    PrimitiveViolation,
    _denied_torch_path,
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
        self._rejects("from evograd.benchmark import get_task\nreturn x, (x,)", "evograd")

    def test_an_aliased_import_does_not_launder_the_call(self):
        self._rejects("from torch import matmul\nreturn matmul(x, w), (x,)",
                      "matmul")

    def test_an_alias_or_attribute_read_does_not_launder_the_namespace(self):
        # The rules bind to what a name reaches, not to the spelling at the call.
        self._rejects("import torch.nn.functional as fn\nreturn fn.silu(x), (x,)",
                      "torch.nn.functional")
        self._rejects("_s = torch.nn.functional.silu\nreturn _s(x), (x,)", "torch.nn.functional")
        self._rejects("_mm = torch.matmul\nreturn _mm(x, w), (x,)", "torch.matmul")
        self._rejects("_mm = getattr(torch, 'matmul')\nreturn _mm(x, w), (x,)", "getattr")
        self._rejects("_f = __import__('torch.nn.functional')\nreturn x, (x,)", "__import__")
        self._rejects("import importlib\nreturn x, (x,)", "importlib")
        self._rejects("_m = x.matmul\nreturn _m(w), (x,)", "matmul")

    def test_triton_dot_is_the_kernels_own_arithmetic_and_stays_allowed(self):
        # `tl.dot` is what an evolved matmul kernel is made of; denying it would
        # forbid the search space rather than police it.
        usage = check_source(PURE_TRITON, allowed=())
        self.assertFalse(usage["uses_trusted_primitives"])

    def _reject_all(self, names, call="(x, w)"):
        for name in names:
            self._rejects(f"return {name}{call}, (x,)", name)

    def test_library_convolutions(self):
        self._reject_all(("torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv_tbc",
                          "torch.conv_transpose2d", "torch.convolution", "torch.cudnn_convolution",
                          "torch.cudnn_convolution_relu", "torch.miopen_depthwise_convolution",
                          "torch.mkldnn_convolution", "torch.nn.grad.conv2d_input",
                          "torch.nn.grad.conv2d_weight", "torch.nn.Conv2d"))

    def test_library_normalizations(self):
        self._reject_all(("torch.native_layer_norm", "torch.native_group_norm",
                          "torch.native_batch_norm", "torch.instance_norm",
                          "torch.cudnn_batch_norm", "torch.layer_norm", "torch.nn.LayerNorm"))

    def test_embeddings_losses_and_recurrent_layers(self):
        self._reject_all(("torch.embedding", "torch.embedding_bag", "torch.nn.EmbeddingBag",
                          "torch.kl_div", "torch.ctc_loss", "torch.binary_cross_entropy_with_logits",
                          "torch.lstm", "torch.gru_cell"))

    def test_attention_in_any_case(self):
        self._reject_all(("torch.nn.MultiheadAttention", "torch.ao.nn.quantizable.MultiheadAttention"))

    def test_vector_bilinear_and_sparse_contractions(self):
        self._reject_all(("torch.mv", "torch.addmv", "torch.addr", "torch.ger", "torch.vdot",
                          "torch.bilinear", "torch.cdist", "torch.linalg.multi_dot",
                          "torch.linalg.vecdot", "torch.sparse.sampled_addmm",
                          "torch.fbgemm_linear_fp16_weight", "torch.Tensor.addmm_", "x.mv", "x.addbmm",
                          "x.outer", "x.kron", "x.__matmul__"))
        self._rejects("return type(x).__rmatmul__(w, x), (x,)", ".__rmatmul__")
        self._rejects("import operator\nreturn operator.__matmul__(x, w), (x,)", "operator.__matmul__")
        self._rejects("from operator import matmul\nreturn matmul(x, w), (x,)",
                      "from operator import matmul")
        self._rejects("x @= w\nreturn x, (x,)", "'@='")

    def test_autograd_compilation_and_extension_escapes(self):
        self._reject_all(("torch.func.grad", "torch.func.vjp", "torch.utils.checkpoint.checkpoint",
                          "torch.utils.cpp_extension.load_inline", "torch.compiler.disable",
                          "torch.export.export", "torch.Tensor.backward"))
        self._rejects("import ctypes\nreturn x, (x,)", "ctypes")
        self._rejects("import cffi\nreturn x, (x,)", "cffi")

    def test_private_torch_modules_at_any_depth(self):
        self._reject_all(("torch._inductor.select_algorithm.extern_kernels.convolution",
                          "torch._C._nn.linear", "torch.cuda._lazy_call"))
        self._rejects("from torch._inductor.select_algorithm import extern_kernels\n"
                      "return extern_kernels.convolution(x, w), (x,)", "torch._inductor")

    def test_from_imports_are_judged_by_the_full_path(self):
        for body, needle in (
                ("from torch import conv2d\nreturn conv2d(x, w), (x,)", "torch.conv2d"),
                ("from torch import embedding_bag\nreturn embedding_bag(x, w), (x,)",
                 "torch.embedding_bag"),
                ("from torch import nn\nreturn nn.Linear(2, 2)(x), (x,)", "torch.nn"),
                ("from torch.nn import grad\nreturn grad.conv2d_input(x.shape, w, x), (x,)",
                 "torch.nn.grad"),
                ("import torch.nn.grad as g\nreturn g.conv2d_input(x.shape, w, x), (x,)",
                 "torch.nn.grad"),
                ("from torch.linalg import multi_dot\nreturn multi_dot([x, w]), (x,)",
                 "torch.linalg.multi_dot")):
            self._rejects(body, needle)

    def test_the_archived_dit_cudnn_program_is_rejected(self):
        # OpenEvolve's dit_s_2 spatial_conv2d candidate of 2026-10-28, scored
        # correct (1.109) under the rules of the time: its block, verbatim.
        block = (
            "import torch\n\n\n"
            "def spatial_conv2d_forward_with_saved(x, weight, bias, stride=2, padding=0, groups=1):\n"
            "    out = torch.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)\n"
            "    return out, (x, weight)\n\n\n"
            "def spatial_conv2d_backward_from_saved(dout, saved_tensors, stride=2, padding=0, groups=1):\n"
            "    x, weight = saved_tensors\n"
            "    dx = torch.nn.grad.conv2d_input(x.shape, weight, dout, stride=stride, padding=padding, groups=groups)\n"
            "    dweight = torch.nn.grad.conv2d_weight(x, weight.shape, dout, stride=stride, padding=padding, groups=groups)\n"
            "    dbias = dout.sum((0, 2, 3))\n"
            "    return dx, dweight, dbias\n")
        source = ("import torch\nimport triton\nimport triton.language as tl\n\n"
                  "# EVOLVE-BLOCK-START\n" + block + "# EVOLVE-BLOCK-END\n" + VENDOR_GEMM_SOURCE)
        with self.assertRaises(PrimitiveViolation) as caught:
            check_source(source, allowed=("vendor_gemm",))
        for name in ("torch.conv2d", "torch.nn.grad.conv2d_input", "torch.nn.grad.conv2d_weight"):
            self.assertIn(name, str(caught.exception))


class TestTheRulesCoverThisTorch(unittest.TestCase):
    """The deny rules against the installed torch, by a spelling of their own."""

    #: A library-looking name, written independently of the checker's tables.
    LIBRARY = re.compile(r"conv|norm|embedding|attention|softmax|loss|entropy|lstm|gru|rnn"
                         r"|mm($|_)|^mv|dot|matmul|gemm|linear|kl_div|cdist|^ger$|outer|inner|kron")
    #: Names that look like it and are ordinary tensor math, a sampler or a setting.
    ORDINARY = frozenset({"norm", "native_norm", "frobenius_norm", "nuclear_norm", "renorm",
                          "norm_except_dim", "normal", "vector_norm", "matrix_norm", "mvlgamma",
                          "get_float32_matmul_precision", "set_float32_matmul_precision",
                          "set_flush_denormal"})

    def test_every_library_entry_point_of_this_torch_is_denied(self):
        # A name a new torch adds to a library family is denied without a new rule.
        for prefix, namespace in (((), torch), (("linalg",), torch.linalg)):
            for name in dir(namespace):
                if name.startswith("_") or not callable(getattr(namespace, name, None)):
                    continue
                if self.LIBRARY.search(name.lower()):
                    with self.subTest(name=".".join(("torch", *prefix, name))):
                        self.assertEqual(_denied_torch_path([*prefix, name]),
                                         name not in self.ORDINARY)

    def test_ordinary_tensor_ops_stay_allowed(self):
        body = """
        y = torch.empty_like(x)
        z = torch.empty_strided((4, 4), (4, 1), device=x.device, dtype=x.dtype)
        s = torch.where(x > 0, torch.exp(x), torch.cat([x, w]))
        t = torch.erf(x) + torch.sigmoid(x) + torch.sum(x) + torch.cumsum(x, 0)
        u = torch.index_select(x, 0, torch.argsort(x[:, 0])) + torch.bincount(x.flatten()).sum()
        v = torch.linalg.vector_norm(x, dim=-1) + torch.norm(x) + x.unfold(0, 2, 1).sum()
        a = torch.as_strided(x, (2,), (1,)).float().contiguous().view(-1)
        stream = torch.cuda.current_stream()
        with torch.no_grad():
            return y, (x, z, s, t, u, v, a, stream)
        """
        usage = check_source(_with_body(body), allowed=())
        self.assertFalse(usage["uses_trusted_primitives"])


class TestVendorSdpaCapability(unittest.TestCase):
    """The attention grant: the same opt-in discipline as the GEMM one."""

    PAIR = ("o, lse = vendor_sdpa_forward(x, w, w)\nreturn o, (x, w, o, lse)")

    def test_the_pair_is_accepted_with_the_grant_and_rejected_without(self):
        from evograd.pipelines.shared.primitives import VENDOR_SDPA_SOURCE

        used = check_source(_with_body(self.PAIR, trusted=VENDOR_SDPA_SOURCE),
                            allowed=("vendor_sdpa",))
        self.assertTrue(used["uses_trusted_primitives"])
        with self.assertRaises(PrimitiveViolation):
            check_source(_with_body(self.PAIR), allowed=())

    def test_the_grant_does_not_open_attention_spellings(self):
        from evograd.pipelines.shared.primitives import VENDOR_SDPA_SOURCE

        for body in ("return torch.nn.functional.scaled_dot_product_attention(x, w, w), (x,)",
                     "return torch.ops.aten._scaled_dot_product_cudnn_attention(x, w, w, None, True)[0], (x,)"):
            with self.assertRaises(PrimitiveViolation):
                check_source(_with_body(body, trusted=VENDOR_SDPA_SOURCE), allowed=("vendor_sdpa",))

    def test_both_grants_render_in_registry_order(self):
        layer = render_primitive_layer(("vendor_sdpa", "vendor_gemm"))
        self.assertLess(layer.index("def vendor_gemm"), layer.index("def vendor_sdpa_forward"))
        self.assertIn("def vendor_sdpa_backward", layer)


class TestExistingArtifactsAreUnaffected(unittest.TestCase):
    def test_a_pure_triton_artifact_validates_with_no_capability(self):
        usage = check_source(PURE_TRITON, allowed=())
        self.assertEqual(usage["allowed_primitives"], [])
        self.assertEqual(usage["primitive_call_sites"], {})

    def test_rendering_without_a_grant_adds_nothing(self):
        self.assertEqual(render_primitive_layer(()), "")
        self.assertIn("def vendor_gemm", render_primitive_layer(("vendor_gemm",)))

    def test_the_deployment_layer_records_the_grant_either_way(self):
        from evograd.benchmark import get_task
        from evograd.pipelines.shared.artifact import render_deployment_layer

        op = get_task("qwen3_qkv_norm_rope")
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
