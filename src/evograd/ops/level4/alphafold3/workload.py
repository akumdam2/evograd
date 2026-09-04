"""The AlphaFold3 training-step workload: the torch side of the declaration.

Implements ``bench.tier3_model.TrainingWorkload`` over `alphafold3-pytorch`_,
patching evolved kernels in by module surgery — the route for a model this
repository did not write, and the same thing ``apply_liger_kernel_to_llama``
does to a HuggingFace model.

Everything importing the model implementation is deferred into the functions
that need it: the declaration in ``__init__`` must stay importable without
torch, and this module without ``alphafold3-pytorch`` (whose absence makes the
task *uncovered*, not broken).

Two determinism decisions worth knowing about before touching this file:

* **Batches are seeded synthetic features.** The tier-3 protocol compares
  providers on identical data; an MSA/template pipeline would inject variance
  that is not the kernels'.
* **The forward itself is stochastic** — the diffusion module samples noise and
  the pairformer uses structured dropout — so :meth:`AF3Workload.loss` reseeds
  the global RNG from the batch's seed before every call. Without that, two
  providers would train on different noise and the loss-agreement metric would
  compare nothing. The reseed is identical work for every provider, inside the
  timed region for all of them alike.

.. _alphafold3-pytorch: https://pypi.org/project/alphafold3-pytorch/
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn

from evograd.bench.tier3_patch import (
    KernelSet,
    ModulePatch,
    PatchProvenance,
    Site,
    SiteRegistry,
    patch_modules,
)
from evograd.opdecl.activity import Workload
from evograd.opdecl.models import ALPHAFOLD3, ALPHAFOLD3_2L, AlphaFoldConfig
from evograd.ops.level4.alphafold3 import AF3_SITE_OPS


def _surgery_only(site: str):
    """The 'default' of a surgery-route site, which is not a callable path.

    An unpatched site here is the original module in the built tree; nothing
    ever calls through the kernel set for it. Reaching this is a wiring bug,
    and saying so beats returning something that silently computes.
    """

    def refuse(*_args, **_kwargs):
        raise RuntimeError(
            f"site {site!r} is patched by module surgery; the unpatched model "
            "does not call through the kernel set"
        )

    return refuse


#: The AF3 patchable surface, owned by this workload as every registry is.
AF3_SITES = SiteRegistry(
    name="alphafold3",
    sites=tuple(
        Site(site, op_name, _surgery_only(site))
        for site, op_name in AF3_SITE_OPS.items()
    ),
)


def _require_model_package():
    try:
        import alphafold3_pytorch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the alphafold3 workload needs alphafold3-pytorch; "
            "install the 'af3' extra (pip install 'evograd[af3]')"
        ) from exc


# ── replacement modules, one per site ────────────────────────────────────────


class _PatchedLayerNorm(nn.Module):
    """An affine ``nn.LayerNorm``, rerouted through an evolved ``layernorm`` pair."""

    def __init__(self, original: nn.LayerNorm, kernel):
        super().__init__()
        # The original's parameters, not copies: surgery must carry trained
        # state across, or the model is silently reinitialized.
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.kernel = kernel

    def forward(self, x):
        return self.kernel(x, self.weight, self.bias, self.eps)


class _PatchedSwiGLU(nn.Module):
    """alphafold3-pytorch's ``SwiGLU``, rerouted through an evolved ``swiglu`` pair.

    Their spelling chunks one tensor — ``x, gates = chunk(2)`` then
    ``silu(gates) * x`` — while the declaration is ``swiglu(a, b) = silu(a) * b``,
    so ``a`` is their ``gates`` half and ``b`` their ``x`` half. The halves are
    made contiguous because chunking leaves strided views, and the pair was
    declared, verified and evolved on contiguous rows.
    """

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return self.kernel(gates.contiguous(), x.contiguous())


class _PatchedAttend(nn.Module):
    """alphafold3-pytorch's ``Attend`` core, rerouted through ``evoattention``.

    ``Attend`` is the pure attention mathematics after projection — exactly the
    boundary the ``evoattention`` declaration draws. Shapes differ by
    convention: ``Attend`` runs ``[b, h, n, d]`` with an additive ``[b, h, n, n]``
    bias and a boolean key mask, the declaration runs ``[B, S, N, H, D]`` with a
    float32 additive ``res_mask`` and ``pair_bias``; the MSA axis ``S`` is 1
    here, as it is for pair-bias attention over the single representation.

    Configurations the declaration does not model are refused rather than
    silently computed differently: a custom scale, softclamp, attention
    dropout, windowing, or memory key/values would make the patched model a
    different function, and a fast wrong answer is worse than a loud one.
    """

    def __init__(self, original, kernel):
        super().__init__()
        if original.is_local_attn:
            raise ValueError("evoattention site cannot take a windowed Attend")
        if original.scale is not None:
            raise ValueError("evoattention assumes the default 1/sqrt(d) scale")
        if original.enable_attn_softclamp:
            raise ValueError("evoattention does not implement logit softclamp")
        if original.dropout:
            raise ValueError("evoattention does not implement attention dropout")
        self.kernel = kernel

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        windowed_mask=None,
        attn_bias=None,
        memory_kv=None,
    ):
        if windowed_mask is not None or memory_kv is not None:
            raise ValueError(
                "evoattention site received windowed/memory-kv attention inputs"
            )
        batch, heads, n, _ = q.shape

        def to_declared(t):  # [b, h, n, d] -> [B, S=1, N, H, D]
            return t.permute(0, 2, 1, 3).unsqueeze(1).contiguous()

        if attn_bias is None:
            pair_bias = q.new_zeros((batch, 1, heads, n, n), dtype=torch.float32)
        else:
            # [b, h, n, n] (or head-broadcast) -> [B, 1, H, N, N], float32 as
            # MegaFold keeps it.
            pair_bias = attn_bias.expand(batch, heads, n, n).unsqueeze(1).float()
        if mask is None:
            res_mask = q.new_zeros((batch, 1, 1, 1, n), dtype=torch.float32)
        else:
            res_mask = torch.where(
                mask.view(batch, 1, 1, 1, n),
                q.new_zeros((), dtype=torch.float32),
                q.new_full((), -1e9, dtype=torch.float32),
            )

        out = self.kernel(
            to_declared(q), to_declared(k), to_declared(v), res_mask, pair_bias
        )
        return out.squeeze(1).permute(0, 2, 1, 3)  # -> [b, h, n, d]


def _module_patches() -> tuple[ModulePatch, ...]:
    """The AF3 site table, bound to the real module classes."""
    _require_model_package()
    from alphafold3_pytorch.alphafold3 import (
        AttentionPairBias,
        SwiGLU,
        TriangleAttention,
    )

    def affine_layer_norm(module: nn.Module) -> bool:
        return (
            type(module) is nn.LayerNorm
            and module.elementwise_affine
            and module.bias is not None
        )

    def swap_inner_attend(module: nn.Module, kernel) -> nn.Module:
        # The swap point is the attention core inside the module's Attention;
        # projections, gating and output stay the original's, weights intact.
        module.attn.attend = _PatchedAttend(module.attn.attend, kernel)
        return module

    return (
        ModulePatch("layer_norm", affine_layer_norm, _PatchedLayerNorm),
        ModulePatch(
            "transition",
            lambda m: type(m) is SwiGLU,
            lambda _original, kernel: _PatchedSwiGLU(kernel),
        ),
        ModulePatch(
            "pair_bias_attention",
            lambda m: type(m) is AttentionPairBias and m.window_size is None,
            swap_inner_attend,
        ),
        ModulePatch(
            "triangle_attention",
            lambda m: type(m) is TriangleAttention,
            swap_inner_attend,
        ),
    )


# ── the workload ─────────────────────────────────────────────────────────────


class AF3Workload:
    """AlphaFold3 training, as a workload the tier-3 harness can drive.

    Implements the :class:`~evograd.bench.tier3_model.TrainingWorkload`
    protocol. Patching is by module surgery on the built tree; the paths
    actually replaced are recorded and reported, so a run states what it
    swapped rather than what it meant to.
    """

    unit_name = "residues"
    site_registry = AF3_SITES

    def __init__(
        self,
        config: AlphaFoldConfig,
        *,
        batch: int,
        residues: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.config = config
        self.batch = batch
        self.residues = residues
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.name = f"{config.name}[batch={batch},residues={residues}]"
        self._loss_seed = seed

    def units_per_step(self) -> int:
        return self.batch * self.residues

    def build(self, kernels: KernelSet) -> nn.Module:
        model, _provenance = self.build_patched(kernels)
        return model

    def build_patched(self, kernels: KernelSet) -> tuple[nn.Module, PatchProvenance]:
        _require_model_package()
        from alphafold3_pytorch import Alphafold3

        config = self.config
        torch.manual_seed(self.seed)
        model = Alphafold3(
            dim_atom_inputs=config.dim_atom_inputs,
            dim_atompair_inputs=config.dim_atompair_inputs,
            dim_template_feats=config.dim_template_feats,
            atoms_per_window=config.atoms_per_window,
            dim_single=config.c_s,
            dim_pairwise=config.c_z,
            confidence_head_kwargs=dict(
                pairformer_depth=config.confidence_pairformer_depth
            ),
            template_embedder_kwargs=dict(
                pairformer_stack_depth=config.template_pairformer_depth
            ),
            msa_module_kwargs=dict(depth=config.msa_module_depth, dim_msa=config.c_m),
            pairformer_stack=dict(
                depth=config.pairformer_depth,
                pair_bias_attn_dim_head=config.pair_bias_attn_dim_head,
                pair_bias_attn_heads=config.pair_bias_attn_heads,
            ),
            diffusion_module_kwargs=dict(
                atom_encoder_depth=config.diffusion_atom_encoder_depth,
                token_transformer_depth=config.diffusion_token_transformer_depth,
                atom_decoder_depth=config.diffusion_atom_decoder_depth,
            ),
        ).to(device=self.device, dtype=self.dtype)

        # Attention-weight dropout off, for EVERY provider — the pairformer's
        # AttentionPairBias carries the pair-stack dropout inside the attention
        # core, which no fused-attention contract models (MegaFold's
        # EvoAttention and DS4Sci alike). Zeroing it uniformly keeps all
        # providers training the same function, and keeps their RNG streams
        # aligned: a dropout that fires in the eager model but not inside a
        # fused kernel would shift every later random draw. Structured
        # row/column dropout sits outside the kernels and stays on.
        from alphafold3_pytorch.attention import Attend

        for module in model.modules():
            if isinstance(module, Attend):
                module.dropout = 0.0
                module.attn_dropout.p = 0.0

        # patch_modules touches only patched sites, records what it replaced,
        # and raises on a requested site that matches nothing — the case where
        # the model implementation's class names have moved out from under the
        # matchers.
        provenance = patch_modules(model, _module_patches(), kernels)
        return model, provenance

    def batch_for(self, *, seed: int) -> Any:
        _require_model_package()
        batch = _synthetic_batch(
            self.config, batch=self.batch, residues=self.residues, seed=seed
        )
        self._loss_seed = seed
        moved = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.dict().items()
        }
        return moved

    def loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        # The forward samples diffusion noise and dropout masks; reseeding here
        # is what makes every provider train on the same randomness.
        torch.manual_seed(self._loss_seed)
        return model(**batch)

    def describe(self) -> dict[str, Any]:
        return {
            "workload": "alphafold3_train_step",
            "model": self.config.name,
            "pairformer_depth": self.config.pairformer_depth,
            "diffusion_depth": self.config.diffusion_token_transformer_depth,
            "batch": self.batch,
            "residues": self.residues,
            "dtype": str(self.dtype).removeprefix("torch."),
        }


def _synthetic_batch(config: AlphaFoldConfig, *, batch: int, residues: int, seed: int):
    """Seeded synthetic training inputs at a fixed crop length.

    Mirrors alphafold3-pytorch's own ``MockAtomDataset`` — the package's
    statement of what a training example contains — with the two properties a
    benchmark batch needs and the mock lacks: the crop length is exactly
    ``residues`` rather than sampled, and everything is generated under a
    caller-supplied seed so every provider sees the same bytes.
    """
    from alphafold3_pytorch.inputs import collate_inputs_to_batched_atom_input
    from alphafold3_pytorch.mocks import MockAtomDataset

    dataset = MockAtomDataset(
        data_length=batch,
        max_seq_len=residues + 1,
        atoms_per_window=config.atoms_per_window,
        dim_atom_inputs=config.dim_atom_inputs,
    )
    examples = []
    for index in range(batch):
        random.seed(seed + 7919 * index)
        torch.manual_seed(seed + 7919 * index)
        examples.append(_fixed_length_example(dataset, residues))
    return collate_inputs_to_batched_atom_input(
        examples, atoms_per_window=config.atoms_per_window
    )


def _fixed_length_example(dataset, residues: int):
    """One mock example at exactly ``residues`` tokens.

    ``MockAtomDataset`` draws ``seq_len`` with ``random.randrange``; pinning the
    draw pins the crop while leaving every other generated feature to the
    seeded RNGs.
    """
    original = random.randrange
    random.randrange = lambda *args, **kwargs: residues
    try:
        return dataset[0]
    finally:
        random.randrange = original


def make_workload(
    case: Workload,
    *,
    device: str = "cuda",
    seed: int = 0,
    config: AlphaFoldConfig | None = None,
) -> AF3Workload:
    """Build the training workload for one declared benchmark case.

    ``config`` defaults to whichever registered AlphaFold3 configuration the
    case's dims re-derive from — the full model for declared benchmark cases,
    the reduced ``alphafold3_2l`` for iteration cases built from it — and the
    dims are checked against the configuration rather than trusted, so a case
    cannot quietly claim an architecture it does not describe.
    """
    dims = case.dims
    if config is None:
        for candidate in (ALPHAFOLD3, ALPHAFOLD3_2L):
            expected = candidate.train_step_dims(
                batch=dims["batch"], residues=dims["residues"]
            )
            if expected == dims:
                config = candidate
                break
        else:
            raise ValueError(
                f"case dims {dims} re-derive from no registered AlphaFold3 "
                "configuration"
            )
    else:
        expected = config.train_step_dims(
            batch=dims["batch"], residues=dims["residues"]
        )
        if expected != dims:
            raise ValueError(
                f"case dims {dims} do not re-derive from {config.name}: "
                f"expected {expected}"
            )
    return AF3Workload(
        config,
        batch=dims["batch"],
        residues=dims["residues"],
        device=device,
        dtype=getattr(torch, case.dtype),
        seed=seed,
    )
