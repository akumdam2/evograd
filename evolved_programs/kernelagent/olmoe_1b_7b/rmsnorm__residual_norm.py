"""adbench-tools artifact: task olmoe_1b_7b.rmsnorm.residual_norm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
import torch
import triton
import triton.language as tl


@triton.jit
def _load_eps(eps, EPS_IS_TENSOR: tl.constexpr):
    if EPS_IS_TENSOR:
        return tl.load(eps + 0).to(tl.float32)
    else:
        return eps


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    w_ptr,
    eps,
    y_ptr,
    inv_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = row * N + offs

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    eps_v = _load_eps(eps, EPS_IS_TENSOR)
    mean_sq = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(mean_sq + eps_v)

    y = x * inv * w

    tl.store(y_ptr + idx, y, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _rmsnorm_backward_dx_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    inv_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = row * N + offs

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)

    v = dy * w
    dot = tl.sum(v * x, axis=0)

    dx = inv * v - x * (inv * inv * inv) * dot / N
    tl.store(dx_ptr + idx, dx, mask=mask)


@triton.jit
def _rmsnorm_backward_dw_kernel(
    x_ptr,
    dy_ptr,
    inv_ptr,
    dw_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DW_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_DW_N + tl.arange(0, BLOCK_DW_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_DW_N,), dtype=tl.float32)

    for start_m in tl.range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        ptrs = offs_m[:, None] * N + offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]

        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * inv[:, None], axis=0)

    tl.store(dw_ptr + offs_n, acc, mask=n_mask)


@triton.jit
def _fused_forward_dx_pack_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    eps,
    packed_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused per-row RMSNorm forward + backward-dx + packing.

    Stores:
      packed[0:NUMEL]           = forward output flattened as fp32
      packed[NUMEL:2*NUMEL]     = dx flattened as fp32

    dweight is a cross-row reduction and is computed by a separate Triton
    reduction kernel.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = row * N + offs

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    eps_v = _load_eps(eps, EPS_IS_TENSOR)
    mean_sq = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(mean_sq + eps_v)

    y = x * inv * w

    v = dy * w
    dot = tl.sum(v * x, axis=0)
    dx = inv * v - x * (inv * inv * inv) * dot / N

    tl.store(packed_ptr + idx, y, mask=mask)
    tl.store(packed_ptr + NUMEL + idx, dx, mask=mask)


@triton.jit
def _dw_pack_recompute_kernel(
    x_ptr,
    dy_ptr,
    eps,
    packed_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DW_N: tl.constexpr,
):
    """
    Computes dweight directly into the packed fp32 return buffer.

    dweight[j] = sum_rows dy[row, j] * x[row, j] * inv_rms[row]
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_DW_N + tl.arange(0, BLOCK_DW_N)
    n_mask = offs_n < N

    eps_v = _load_eps(eps, EPS_IS_TENSOR)
    acc = tl.zeros((BLOCK_DW_N,), dtype=tl.float32)

    for start_m in tl.range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        # Recompute inv_rms for this tile of rows.
        offs_all = tl.arange(0, BLOCK_DW_N)
        # This kernel intentionally uses BLOCK_DW_N for the column tile being
        # reduced, but RMS requires all N columns.  Therefore, below we compute
        # the variance with a loop over the hidden dimension in chunks.
        inv_vals = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Accumulate row-wise sum of squares over the full hidden dimension.
        ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in tl.range(0, N, BLOCK_DW_N):
            k = k0 + offs_all
            kmask = k < N
            x_var = tl.load(
                x_ptr + offs_m[:, None] * N + k[None, :],
                mask=m_mask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float32)
            ss += tl.sum(x_var * x_var, axis=1)

        inv_vals = tl.rsqrt(ss / N + eps_v)

        ptrs = offs_m[:, None] * N + offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]

        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * inv_vals[:, None], axis=0)

    tl.store(packed_ptr + 2 * NUMEL + offs_n, acc, mask=n_mask)


def _check_inputs(x, weight):
    assert x.is_cuda, "x must be a CUDA tensor"
    assert weight.is_cuda, "weight must be a CUDA tensor"
    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert weight.dim() == 1, "weight must be 1D over the normalized dimension"
    assert x.numel() % weight.numel() == 0, "x.numel() must be divisible by weight.numel()"


def _block_n(n: int) -> int:
    return triton.next_power_of_2(n)


def _warps(block: int) -> int:
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    return 1


def _eps_is_tensor(eps) -> bool:
    return isinstance(eps, torch.Tensor)


def _output_dtype(x, weight):
    # Mirrors ordinary PyTorch promotion for the dtypes used by this benchmark.
    if x.dtype == torch.float32 or weight.dtype == torch.float32:
        return torch.float32
    if x.dtype != weight.dtype:
        return torch.float32
    return x.dtype


def rmsnorm_forward_with_saved(x, weight, eps):
    """
    RMSNorm forward implemented entirely in Triton.

    Computes y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight.
    The saved state contains x, weight, inv_rms, and eps for the backward pass.
    """
    _check_inputs(x, weight)

    n = weight.numel()
    m = x.numel() // n
    block = _block_n(n)

    y = torch.empty(x.shape, device=x.device, dtype=_output_dtype(x, weight))
    inv = torch.empty((m,), device=x.device, dtype=torch.float32)

    _rmsnorm_forward_kernel[(m,)](
        x,
        weight,
        eps,
        y,
        inv,
        M=m,
        N=n,
        EPS_IS_TENSOR=_eps_is_tensor(eps),
        BLOCK_N=block,
        num_warps=_warps(block),
    )

    saved = (x, weight, inv, eps)
    return y, saved


def rmsnorm_backward_from_saved(upstream, saved):
    """
    RMSNorm backward implemented entirely in Triton.

    Returns gradients with respect to x and weight:
      dx[j] = inv * dy[j] * weight[j]
              - x[j] * inv^3 / N * sum_k(dy[k] * weight[k] * x[k])
      dweight[j] = sum_rows dy[row, j] * x[row, j] * inv[row]
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, weight, inv, _eps = saved
    _check_inputs(x, weight)
    assert dy.is_cuda, "upstream gradient must be a CUDA tensor"
    assert dy.is_contiguous(), "upstream gradient must be contiguous"

    n = weight.numel()
    m = x.numel() // n
    block = _block_n(n)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)

    _rmsnorm_backward_dx_kernel[(m,)](
        x,
        weight,
        dy,
        inv,
        dx,
        M=m,
        N=n,
        BLOCK_N=block,
        num_warps=_warps(block),
    )

    block_dw_n = 64
    block_m = 32
    grid_dw = (triton.cdiv(n, block_dw_n),)
    _rmsnorm_backward_dw_kernel[grid_dw](
        x,
        dy,
        inv,
        dw,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_DW_N=block_dw_n,
        num_warps=4,
    )

    return dx, dw


def kernel_function(x, weight, eps, *upstream):
    """
    Forward+backward fused entry point required by the evaluator.

    This wrapper returns a single fp32 flattened tensor matching:
      cat([forward_output.float().reshape(-1),
           dx.float().reshape(-1),
           dweight.float().reshape(-1)])

    Fusion strategy:
      1. One Triton kernel fuses RMSNorm forward, per-row backward dx, and
         direct fp32 packing of both tensors.
      2. dweight is a cross-row reduction, so it is computed in a second Triton
         reduction kernel and written directly into the packed output buffer.

    The Python wrapper performs only validation, allocation, and launch setup.
    """
    _check_inputs(x, weight)
    assert len(upstream) >= 1, "kernel_function expects an upstream gradient"
    dy = upstream[0]
    if isinstance(dy, (tuple, list)):
        dy = dy[0]
    assert dy.is_cuda, "upstream gradient must be a CUDA tensor"
    assert dy.is_contiguous(), "upstream gradient must be contiguous"

    n = weight.numel()
    m = x.numel() // n
    numel = x.numel()
    total = 2 * numel + n

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)

    block = _block_n(n)
    _fused_forward_dx_pack_kernel[(m,)](
        x,
        weight,
        dy,
        eps,
        packed,
        M=m,
        N=n,
        NUMEL=numel,
        EPS_IS_TENSOR=_eps_is_tensor(eps),
        BLOCK_N=block,
        num_warps=_warps(block),
    )

    block_dw_n = 64
    block_m = 32
    grid_dw = (triton.cdiv(n, block_dw_n),)
    _dw_pack_recompute_kernel[grid_dw](
        x,
        dy,
        eps,
        packed,
        M=m,
        N=n,
        NUMEL=numel,
        EPS_IS_TENSOR=_eps_is_tensor(eps),
        BLOCK_M=block_m,
        BLOCK_DW_N=block_dw_n,
        num_warps=4,
    )

    return packed
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'rmsnorm', 'arguments': ['x', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight'], 'gradient_return_order': ['dx', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class RmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        y, saved = rmsnorm_forward_with_saved(x, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dy = dy.contiguous() if (dy is not None and not dy.is_contiguous()) else dy
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dweight = rmsnorm_backward_from_saved(
            dy, saved, **_backward_scalars(rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, None


def rmsnorm_deployment(x, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = RmsnormFunction.apply(x, weight, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class RmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-05):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return rmsnorm_deployment(x, self.weight, self.eps)


DEPLOYMENT_ENTRY = "rmsnorm_deployment"
