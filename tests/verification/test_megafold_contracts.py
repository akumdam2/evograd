"""Every MegaFold Level-2 contract against a CPU copy of the module it stands for.

For each contract, each static setting and both dtypes: the reference gives the
module's output and every gradient bit for bit, through the site adapter that
deploys it, and consumes RNG as the module does. The recorder's arguments
reproduce the live output too, which is ``fidelity`` on CPU. GPU rounding is
what ``fidelity`` and ``site_check`` add (``docs/AF3.md`` section 7).
"""

from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from evograd.benchmark import get_task
from evograd.benchmark.topdown.megafold.precision import result_precision
from evograd.evaluation.tier3.patch import ModulePatch
from evograd.evaluation.tier3.workloads.megafold import sites as S
from evograd.evaluation.tier3.workloads.megafold import workload as W
from evograd.opdecl.activity import Workload, bind_shape
from evograd.opdecl.inputs import arg_dtype_name, make_case_inputs, resolve_dtype
from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
from evograd.verification.integrations.megafold.capture import MegaFoldCapture, _dims
from evograd.verification.integrations.megafold.reference import reference_op

from tests._small_cases import SMALL

from . import megafold_transcription as T

BF16, FP32 = torch.bfloat16, torch.float32
#: (the autocast in force, the dtypes an input can arrive in there)
REGIONS = ((None, (FP32,)), (BF16, (FP32, BF16)))


def is_float(value) -> bool:
    return torch.is_tensor(value) and value.is_floating_point()


class Holder(nn.Module):
    """A one-module tree, for the recorder's paths."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


class Recorded(MegaFoldCapture):
    """The recorder, keeping the live tensors of its own site's calls."""

    def __init__(self, model, site):
        super().__init__(model)
        self.site, self.seen = site, []

    def _observe(self, op, site, path, values, out, layout):
        if site == self.site:
            self.seen.append((values, layout(out.detach())))


class FusedGemms(TorchFunctionMode):
    """The input shape of every biased ``F.linear`` that takes the fused ``addmm``.

    CPU rounding agrees on the two paths where the GPU's does not, so a wrong path
    shows in no value here, and the paths themselves are compared. A no-grad replay
    is left out: it is the same call again.
    """

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__name__", "") == "linear" and torch.is_grad_enabled():
            x, bias = args[0], args[2] if len(args) > 2 else kwargs.get("bias")
            if bias is not None and (x.ndim == 2 or (x.ndim == 3 and x.is_contiguous())):
                self.shapes.append(tuple(x.shape))
        return func(*args, **kwargs)


class ContractCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(S, "_sp_size", lambda: 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def abi(self, op, values):
        """The arguments are the task's, in order, at shapes and dtypes that bind one case."""
        self.assertEqual(list(values), [a.name for a in op.args])
        dims, dtype = _dims(op, values)
        self.assertEqual(sorted(dims), sorted(op.dims))
        case = mock.Mock(dims=dims, dtype=str(dtype).removeprefix("torch."))
        for arg in op.args:
            if torch.is_tensor(values[arg.name]):
                with self.subTest(argument=arg.name):
                    self.assertEqual(tuple(values[arg.name].shape), bind_shape(arg.shape, dims))
                    self.assertEqual(values[arg.name].dtype, resolve_dtype(arg_dtype_name(arg, case)))
        return dims, dtype

    def drop_in(self, site, make, inputs, kwargs=None, region=None):
        """The adapter with the reference is the module, bit for bit, gradients and RNG included."""
        kwargs, boundary = kwargs or {}, S.BOUNDARIES[site]
        op = reference_op(boundary.task)
        forward = resolve_runtime_forward(op)

        def kernel(*args):
            values = dict(zip((a.name for a in op.args), args))
            self.abi(op, values)
            out = forward(*args)
            with torch.no_grad():  # a no-grad replay takes the same path
                self.assertTrue(torch.equal(forward(*args), out))
            return out

        def step(module):
            leaves = tuple(t.clone().requires_grad_() for t in inputs)
            named = {k: v.clone().requires_grad_() if is_float(v) else v for k, v in kwargs.items()}
            torch.manual_seed(5)
            with FusedGemms() as fused, T.autocast(region):
                out = module(*leaves, **named)
            after = torch.rand(1)
            torch.manual_seed(6)
            out.backward(torch.randn(out.shape).to(out.dtype))
            floats = tuple(v for v in named.values() if is_float(v))
            return out.detach(), after, (*leaves, *floats), sorted(fused.shapes)

        live, patched = make(), S.CompositeAdapter(make(), kernel, boundary)
        self.assertEqual([n for n, _ in live.named_parameters()],
                         [n for n, _ in patched.named_parameters()])
        ours, theirs = step(patched), step(live)
        self.assertEqual(ours[0].dtype, theirs[0].dtype)
        self.assertTrue(torch.equal(ours[0], theirs[0]), "output")
        self.assertTrue(torch.equal(ours[1], theirs[1]), "the step's RNG")
        self.assertEqual(ours[3], theirs[3], "biased GEMMs on the fused path")
        for mine, others in zip(ours[2], theirs[2]):
            self.assertTrue(torch.equal(mine.grad, others.grad), "input gradient")
        for (name, mine), (_n, others) in zip(patched.named_parameters(), live.named_parameters()):
            with self.subTest(gradient=name):
                self.assertEqual(mine.grad.dtype, others.grad.dtype)
                self.assertTrue(torch.equal(mine.grad, others.grad), name)

    def recorded(self, site, make, matches, inputs, kwargs=None, region=None):
        """What the recorder keeps from the plain module reproduces its live output."""
        kwargs, boundary = kwargs or {}, S.BOUNDARIES[site]
        tree = Holder(make())
        spec = ModulePatch(site, matches, None)
        capture = Recorded(tree, site)
        with mock.patch.object(W, "_module_patches", lambda _m: (spec,)):
            capture.install()
        leaves = tuple(t.clone().requires_grad_() for t in inputs)
        try:
            with T.autocast(region):
                tree(*leaves, **kwargs)
        finally:
            capture.remove()
        self.assertTrue(capture.seen, "the recorder saw no call")
        forward = resolve_runtime_forward(reference_op(boundary.task))
        for values, live in capture.seen:
            self.assertTrue(torch.equal(forward(*values.values()), live))

    def fixtures(self, task):
        """The declaration is valid and its small cases run at their declared shapes and dtypes."""
        op, reference = get_task(task), reference_op(task)
        forward = resolve_runtime_forward(reference)
        for workload in SMALL[task]:
            with self.subTest(task=task, dims=workload.dims, dtype=workload.dtype):
                values = make_case_inputs(op, workload, device="cpu")
                out = forward(*(values[a.name] for a in op.args))
                self.assertEqual(tuple(out.shape), bind_shape(op.output.shape, workload.dims))
                self.assertEqual(out.dtype, resolve_dtype(arg_dtype_name(op.output, workload)))
                self.assertTrue(torch.isfinite(out.float()).all())
        names = {*op.output_names, *op.grad_names()}
        for dtype in ("float32", "bfloat16"):
            self.assertEqual(set(result_precision(op, dtype)), names)


def pair_inputs(n=9, c=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(1, n, generator=g) > 0.2
    mask[:, 0] = True
    return torch.randn(1, n, n, c, generator=g), mask


class TriangleMultiplication(ContractCase):
    def make(self, mix):
        def build():
            torch.manual_seed(3)
            return T.TriangleMultiplication(dim=16, mix=mix, dropout=0.25, dropout_type="row")

        return build

    def test_each_mix_in_each_region(self):
        x, mask = pair_inputs()
        for mix in ("outgoing", "incoming"):
            for region, dtypes in REGIONS:
                for dtype in dtypes:
                    with self.subTest(mix=mix, region=region, input=dtype):
                        site = f"triangle_multiplication_{mix}"
                        self.drop_in(site, self.make(mix), (x.to(dtype),), {"mask": mask}, region)
                        self.recorded(site, self.make(mix),
                                      lambda m: type(m) is T.TriangleMultiplication,
                                      (x.to(dtype),), {"mask": mask}, region)

    def test_the_mixes_differ(self):
        x, mask = pair_inputs()
        torch.manual_seed(0)
        one = self.make("outgoing")()(x, mask=mask)
        torch.manual_seed(0)
        other = self.make("incoming")()(x, mask=mask)
        self.assertFalse(torch.equal(one, other))

    def test_a_missing_mask_is_every_token(self):
        x, _mask = pair_inputs()
        self.drop_in("triangle_multiplication_outgoing", self.make("outgoing"), (x,), {})

    def test_declarations(self):
        for mix in ("outgoing", "incoming"):
            self.fixtures(f"megafold_triangle_multiplication_{mix}")


class TriangleAttention(ContractCase):
    def make(self, node):
        def build():
            torch.manual_seed(4)
            dropout_type = "row" if node == "starting" else "col"
            return T.PreLayerNorm(T.TriangleAttention(
                dim=16, heads=2, node_type=node, dropout=0.25, dropout_type=dropout_type), dim=16)

        return build

    def test_each_node_in_each_region(self):
        x, mask = pair_inputs()
        for node in ("starting", "ending"):
            for region, dtypes in REGIONS:
                for dtype in dtypes:
                    with self.subTest(node=node, region=region, input=dtype):
                        site = f"triangle_attention_{node}"
                        self.drop_in(site, self.make(node), (x.to(dtype),), {"mask": mask}, region)
                        self.recorded(site, self.make(node), lambda m: type(m) is T.PreLayerNorm,
                                      (x.to(dtype),), {"mask": mask}, region)

    def test_the_ending_node_is_not_the_starting_one(self):
        x, mask = pair_inputs()
        torch.manual_seed(0)
        one = self.make("starting")()(x, mask=mask)
        torch.manual_seed(0)
        other = self.make("ending")()(x, mask=mask)
        self.assertFalse(torch.equal(one, other))

    def test_a_missing_mask_attends_to_every_key(self):
        x, _mask = pair_inputs()
        self.drop_in("triangle_attention_starting", self.make("starting"), (x,), {})

    def test_declarations(self):
        for node in ("starting", "ending"):
            self.fixtures(f"megafold_triangle_attention_{node}")


class Transition(ContractCase):
    #: Each region's rank: pair and MSA are 4D, single is 3D.
    SHAPES = ((1, 9, 9, 16), (1, 5, 9, 16), (1, 9, 16))

    def make(self):
        torch.manual_seed(5)
        return T.FusedTransition(16, expansion_factor=2)

    def test_every_region(self):
        for shape in self.SHAPES:
            for region, dtypes in REGIONS:
                for dtype in dtypes:
                    with self.subTest(shape=shape, region=region, input=dtype):
                        x = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(dtype)
                        self.drop_in("transition_block", self.make, (x,), {}, region)
                        self.recorded("transition_block", self.make,
                                      lambda m: type(m) is T.FusedTransition, (x,), {}, region)

    def test_the_output_dtype_of_a_bf16_transition_is_bf16(self):
        with T.autocast(BF16):
            out = self.make()(torch.randn(1, 9, 16))
        self.assertEqual(out.dtype, BF16)

    def test_declaration(self):
        self.fixtures("megafold_transition")


def msa_inputs(s=6, n=8, m=16, seed=4):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(1, n, generator=g) > 0.2
    mask[:, 0] = True
    msa_mask = torch.rand(1, s, generator=g) > 0.3
    msa_mask[:, 0] = True
    return torch.randn(1, s, n, m, generator=g), mask, msa_mask


class OuterProductMean(ContractCase):
    def make(self):
        torch.manual_seed(7)
        return T.OuterProductMean()

    def test_in_each_region(self):
        msa, mask, msa_mask = msa_inputs()
        for region, dtypes in REGIONS:
            for dtype in dtypes:
                with self.subTest(region=region, input=dtype):
                    # MSAModule's own sum, float32 under autocast.
                    count = msa_mask.float().sum(-1)
                    kwargs = dict(mask=mask, msa_mask=(msa_mask, count))
                    self.drop_in("outer_product_mean", self.make, (msa.to(dtype),), kwargs, region)
                    self.recorded("outer_product_mean", self.make,
                                  lambda m: type(m) is T.OuterProductMean, (msa.to(dtype),), kwargs, region)

    def test_the_checkpointed_path_passes_the_msa_by_name(self):
        msa, mask, msa_mask = msa_inputs()
        kwargs = dict(msa=msa, mask=mask, msa_mask=(msa_mask, msa_mask.float().sum(-1)))
        self.drop_in("outer_product_mean", self.make, (), kwargs)

    def test_a_bf16_trunk_gets_a_float32_result(self):
        msa, mask, msa_mask = msa_inputs()
        with T.autocast(BF16):
            out = self.make()(msa.bfloat16(), mask=mask, msa_mask=(msa_mask, msa_mask.float().sum(-1)))
        self.assertEqual(out.dtype, FP32)

    def test_the_branch_without_an_msa_mask_is_refused(self):
        msa, mask, _ = msa_inputs()
        patched = S.CompositeAdapter(self.make(), lambda *a: None, S.BOUNDARIES["outer_product_mean"])
        with self.assertRaisesRegex(ValueError, "MSA mask"):
            patched(msa, mask=mask)

    def test_declaration(self):
        self.fixtures("megafold_outer_product_mean")


class MSAPairWeightedAveraging(ContractCase):
    def make(self):
        torch.manual_seed(8)
        return T.MSAPairWeightedAveraging(dropout=0.15, dropout_type="row")

    def test_in_each_region(self):
        msa, mask, _ = msa_inputs()
        pair = torch.randn(1, 8, 8, 8, generator=torch.Generator().manual_seed(9))
        for region, dtypes in REGIONS:
            for dtype in dtypes:
                with self.subTest(region=region, input=dtype):
                    kwargs = dict(msa=msa.to(dtype), pairwise_repr=pair.to(dtype), mask=mask)
                    self.drop_in("msa_pair_weighted_averaging", self.make, (), kwargs, region)
                    self.recorded("msa_pair_weighted_averaging", self.make,
                                  lambda m: type(m) is T.MSAPairWeightedAveraging, (), kwargs, region)

    def test_declaration(self):
        self.fixtures("megafold_msa_pair_weighted_averaging")


class TheKeptContracts(ContractCase):
    """``megafold_layernorm_linear`` and ``megafold_evoattention`` still are the eager computation.

    The adapters are checked in ``test_megafold``. This is the recorder's side, on
    the same copies of MegaFold's modules the new contracts are held to: what it
    keeps at each site reproduces the live output, in both regions.
    """

    def test_layernorm_linear(self):
        def make():
            torch.manual_seed(11)
            return T.LayernormLinear(16, 24, has_linear_bias=False)

        for shape in ((4, 16), (1, 5, 16), (1, 3, 5, 16)):
            for region, dtypes in REGIONS:
                for dtype in dtypes:
                    with self.subTest(shape=shape, region=region, input=dtype):
                        x = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(dtype)
                        self.recorded("layernorm_linear", make,
                                      lambda m: type(m) is T.LayernormLinear, (x,), {}, region)

    def test_evoattention_on_the_bias_each_caller_passes(self):
        def triangle():
            torch.manual_seed(12)
            return T.TriangleAttention(dim=16, heads=2, node_type="starting")

        def pair_bias():
            torch.manual_seed(12)
            return T.AttentionPairBias(heads=2, dim_pairwise=8, dim=16)

        x, mask = pair_inputs()
        single = torch.randn(1, 9, 16, generator=torch.Generator().manual_seed(2))
        pair = torch.randn(1, 9, 9, 8, generator=torch.Generator().manual_seed(3))
        for region, dtypes in REGIONS:
            for dtype in dtypes:
                with self.subTest(region=region, input=dtype):
                    self.recorded("triangle_attention", triangle,
                                  lambda m: type(m) is T.TriangleAttention, (x.to(dtype),),
                                  {"mask": mask}, region)
                    self.recorded("pair_bias_attention", pair_bias,
                                  lambda m: type(m) is T.AttentionPairBias, (single.to(dtype),),
                                  dict(pairwise_repr=pair, mask=mask), region)


class TheWholeTree(unittest.TestCase):
    """``site_check`` on CPU: the real site table on a tree of copied modules."""

    def setUp(self):
        fake = SimpleNamespace(
            AttentionPairBias=T.AttentionPairBias,
            FusedTransition=T.FusedTransition, LayernormLinear=T.LayernormLinear,
            MSAPairWeightedAveraging=T.MSAPairWeightedAveraging, OuterProductMean=T.OuterProductMean,
            PreLayerNorm=T.PreLayerNorm, TriangleAttention=T.TriangleAttention,
            TriangleMultiplication=T.TriangleMultiplication)
        for patcher in (mock.patch.object(S, "_sp_size", lambda: 1),
                        mock.patch.object(W, "_require_model_module", lambda: fake)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tree(self):
        torch.manual_seed(13)
        tree = T.Tree().train()
        for module in tree.modules():  # the site matches a real LayerNorm; its input here is float32
            if type(module) is T.PreLayerNorm:
                norm = nn.LayerNorm(16)
                norm.load_state_dict(module.norm.state_dict())
                module.norm = norm
        return tree

    def inputs(self):
        g = torch.Generator().manual_seed(14)
        mask = torch.rand(1, 8, generator=g) > 0.2
        mask[:, 0] = True
        msa_mask = torch.rand(1, 5, generator=g) > 0.2
        msa_mask[:, 0] = True
        return (torch.randn(1, 5, 8, 16, generator=g), torch.randn(1, 8, 8, 16, generator=g),
                torch.randn(1, 8, 16, generator=g), mask, msa_mask)

    def step(self, tree, region):
        torch.manual_seed(15)
        with T.autocast(region):
            loss = tree(*self.inputs())
        after = torch.rand(1)
        loss.backward()
        return loss.detach(), after, {n: p.grad for n, p in tree.named_parameters() if p.grad is not None}

    def test_every_group_of_sites_is_the_plain_step(self):
        from evograd.evaluation.tier3.patch import patch_modules
        from evograd.verification.integrations.megafold.site_check import reference_kernels

        for region in (None, BF16):
            plain = self.step(self.tree(), region)
            for sites in W.MEGAFOLD_SITES.groups():
                with self.subTest(region=region, sites=sites):
                    tree = self.tree()
                    kernels = reference_kernels(W.MEGAFOLD_SITES, sites, reference_op,
                                                resolve_runtime_forward)
                    provenance = patch_modules(tree, W._module_patches(tree), kernels)
                    self.assertEqual(sorted(provenance.actual_sites), sorted(sites))
                    loss, after, grads = self.step(tree, region)
                    self.assertTrue(torch.equal(loss, plain[0]), "loss")
                    self.assertTrue(torch.equal(after, plain[1]), "the step's RNG")
                    self.assertEqual(sorted(grads), sorted(plain[2]))
                    for name, grad in grads.items():
                        self.assertTrue(torch.equal(grad, plain[2][name]), name)

    def test_every_group_through_the_path_a_candidate_takes(self):
        """The identity control: each kernel through ``bind`` and an ``autograd.Function``.

        The kernels above are plain functions. A candidate arrives as a Function, with
        masks and keep masks as Inactive tensors and scalars passed on to its backward.
        """
        from evograd.benchmark import TASKS
        from evograd.evaluation.tier3.patch import identity_control_kernels, patch_modules

        for region in (None, BF16):
            plain = self.step(self.tree(), region)
            for sites in W.MEGAFOLD_SITES.groups():
                with self.subTest(region=region, sites=sites):
                    tree = self.tree()
                    kernels = identity_control_kernels(TASKS, sites, registry=W.MEGAFOLD_SITES)
                    patch_modules(tree, W._module_patches(tree), kernels)
                    loss, after, grads = self.step(tree, region)
                    self.assertEqual(sorted(grads), sorted(plain[2]))
                    self.assertTrue(torch.isclose(loss, plain[0], rtol=1e-3), "loss")
                    self.assertTrue(torch.equal(after, plain[1]), "the step's RNG")
                    for name, grad in grads.items():
                        self.assertTrue(grad.isfinite().all(), name)

    def test_a_nested_pair_is_refused_by_the_real_table(self):
        from evograd.evaluation.tier3.patch import OverlappingSites, patch_modules
        from evograd.verification.integrations.megafold.site_check import reference_kernels

        tree = self.tree()
        kernels = reference_kernels(W.MEGAFOLD_SITES, ("transition_block", "layernorm_linear"),
                                    reference_op, resolve_runtime_forward)
        with self.assertRaises(OverlappingSites):
            patch_modules(tree, W._module_patches(tree), kernels)

    def test_every_nesting_in_the_tree_is_declared(self):
        """Read off the modules, so a new site cannot nest without saying so."""
        tree = self.tree()
        boundaries = {}
        for spec in W._module_patches(tree):
            boundary = S.BOUNDARIES[spec.site]
            boundaries[spec.site] = [boundary.module(m) for m in tree.modules() if spec.match(m)]
        for outer in boundaries:
            inside = {id(m) for module in boundaries[outer] for m in module.modules()} \
                - {id(module) for module in boundaries[outer]}
            for inner, modules in boundaries.items():
                if inner != outer and any(id(m) in inside for m in modules):
                    with self.subTest(outer=outer, inner=inner):
                        self.assertTrue(W.MEGAFOLD_SITES.conflicts((outer, inner)),
                                        f"{outer} holds {inner} and the pair is not declared")

    def test_what_holds_no_site_runs_stock(self):
        """AdaLN and its gate, pair-bias attention and the LayerNorm-free transition stay in the tree."""
        from evograd.evaluation.tier3.patch import patch_modules
        from evograd.verification.integrations.megafold.site_check import reference_kernels

        for sites in W.MEGAFOLD_SITES.groups():
            with self.subTest(sites=sites):
                tree = self.tree()
                kernels = reference_kernels(W.MEGAFOLD_SITES, sites, reference_op,
                                            resolve_runtime_forward)
                provenance = patch_modules(tree, W._module_patches(tree), kernels)
                self.assertTrue(provenance.actual_sites)
                self.assertIs(type(tree.conditioned_attn), T.ConditionWrapper)
                self.assertIs(type(tree.conditioned_transition), T.ConditionWrapper)
                self.assertIs(type(tree.conditioned_transition.adaptive_norm), T.AdaptiveLayerNorm)
                self.assertIs(type(tree.pair_bias_attn.fn), T.AttentionPairBias)
                self.assertIs(type(tree.pair_bias_attn.fn.to_attn_bias), nn.Sequential)
                if "transition" not in sites:  # that site's SwiGLU swap replaces the module
                    self.assertIs(type(tree.conditioned_transition.fn), T.FusedTransition)

    def test_the_recorder_sees_every_site_once_per_call(self):
        tree = self.tree()
        capture = MegaFoldCapture(tree).install()
        try:
            self.step(tree, BF16)
        finally:
            capture.remove()
        seen = {c["site"] for c in capture.calls} - {"primitive"}
        self.assertEqual(seen, set(W.MEGAFOLD_SITES.names))
        tasks = {c["task"] for c in capture.calls}
        from evograd.benchmark.topdown.megafold.harvest.harvest import LEVEL1, LEVEL2

        self.assertEqual(tasks, (LEVEL1 | LEVEL2) - {"relu"})  # the small tree has no ReLU


class Boundaries(unittest.TestCase):
    def test_every_boundary_is_a_registered_site_with_its_task(self):
        from evograd.benchmark.topdown.megafold import MEGAFOLD_SITE_OPS

        self.assertEqual({k: b.task for k, b in S.BOUNDARIES.items()}, MEGAFOLD_SITE_OPS)

    def test_an_adapter_refuses_sequence_parallelism(self):
        with mock.patch.object(S, "_sp_size", lambda: 2):
            with self.assertRaisesRegex(ValueError, "sp_size"):
                S.CompositeAdapter(nn.Linear(2, 2), lambda *a: None,
                                   S.BOUNDARIES["triangle_multiplication_outgoing"])


class SeedsLower(unittest.TestCase):
    """A task whose forward a seed cannot lower gets no unified seed, so no fallback."""

    TASKS = ("megafold_msa_pair_weighted_averaging", "megafold_triangle_multiplication_outgoing",
             "megafold_triangle_multiplication_incoming")

    def test_the_forward_lowers_in_both_dtypes(self):
        from dataclasses import replace

        import torch._inductor.config as inductor

        from evograd.pipelines.unified.capture import capture_unified_pair
        from evograd.pipelines.unified.codegen import generate_unified_seed

        # A login node's CXX can be a wrapper that cannot build Inductor's CPU headers.
        with inductor.patch({"cpp.cxx": (None, "g++")}):
            self.lower_each_task(replace, capture_unified_pair, generate_unified_seed)

    def lower_each_task(self, replace, capture_unified_pair, generate_unified_seed):
        for name in self.TASKS:
            op = get_task(name)
            dims = dict(SMALL[name][0].dims)
            for dtype in ("float32", "bfloat16"):
                with self.subTest(task=name, dtype=dtype):
                    workload = Workload(dims=dims, dtype=dtype)
                    captured = capture_unified_pair(op, workload, device="cpu")
                    # Only the generator's device check refuses CPU; the source is not run.
                    relabelled = replace(captured, pair=replace(captured.pair, device="cuda:0"))
                    source = generate_unified_seed(op, workload, relabelled)
                    self.assertIn("EVOLVE-BLOCK", source)
                    self.assert_graphs_release_and_never_reuse(source)

    def assert_graphs_release_and_never_reuse(self, source):
        """Each graph drops its intermediates, and reads no name after dropping it."""
        import ast

        graphs = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                  and node.name.startswith("_unified_") and node.name.endswith("_graph")]
        self.assertEqual(len(graphs), 2)
        for graph in graphs:
            dropped, count = set(), 0
            for statement in graph.body:
                if isinstance(statement, ast.Delete):
                    names = {target.id for target in statement.targets}
                    self.assertFalse(names & dropped)
                    dropped |= names
                    count += len(names)
                    continue
                read = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load)}
                self.assertFalse(read & dropped, graph.name)
            self.assertGreater(count, 0, graph.name)

    def test_every_reference_can_be_written_out_for_a_task_pack(self):
        from evograd.benchmark.topdown.megafold import MEGAFOLD_SITE_OPS
        from evograd.opdecl.importing import reference_closure
        from evograd.verification.integrations.megafold.reference import reference_op

        for name in sorted({op for op in MEGAFOLD_SITE_OPS.values() if op.startswith("megafold_")}):
            with self.subTest(task=name):
                task, reference = get_task(name), reference_op(name)
                self.assertTrue(reference_closure(task.forward)[0])
                self.assertTrue(reference_closure(reference.forward)[0])

    def test_the_spelled_glu_is_the_models(self):
        for name in self.TASKS[1:]:
            op = get_task(name)
            inputs = op.make_inputs(torch, op, SMALL[name][0], device="cpu")
            values = [inputs.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
            with self.subTest(task=name):
                self.assertTrue(torch.allclose(resolve_forward(op)(*values),
                                               resolve_runtime_forward(op)(*values)))


if __name__ == "__main__":
    unittest.main()
