"""adbench-tools artifact: task qwen2_vl_2b.layernorm.model.visual.blocks.norm1, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton LayerNorm forward + backward implementation.
#
# Fused path used by kernel_function:
#   1. Per row: compute LayerNorm mean/rstd, forward output, and dx.
#   2. Separately reduce dweight/dbias across rows using a two-stage Triton
#      reduction.
#   3. Write the required oracle format directly:
#        [output.flatten(), dx.flatten(), dweight.flatten(), dbias.flatten()]
#
# Python wrappers only allocate tensors, validate metadata, and launch kernels.
# All numerical computation is performed in Triton kernels.
# -----------------------------------------------------------------------------


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N

    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    xhat = (x - mean) * rstd
    dyw = tl.where(mask, dy * w, 0.0)

    sum_dyw = tl.sum(dyw, axis=0)
    sum_dyw_xhat = tl.sum(dyw * xhat, axis=0)

    dx = rstd * (dyw - (sum_dyw + xhat * sum_dyw_xhat) / N)
    tl.store(dx_ptr + base, dx, mask=mask)


@triton.jit
def _layernorm_param_grad_stage1_kernel(
    dy_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = offs_m[:, None] * N + offs_n[None, :]

    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)

    xhat = (x - mean[:, None]) * rstd[:, None]
    dy_masked = tl.where(mask, dy, 0.0)

    dw = tl.sum(dy_masked * xhat, axis=0)
    db = tl.sum(dy_masked, axis=0)

    row_block = pid_m
    tl.store(partial_dw_ptr + row_block * N + offs_n, dw, mask=offs_n < N)
    tl.store(partial_db_ptr + row_block * N + offs_n, db, mask=offs_n < N)


@triton.jit
def _layernorm_param_grad_stage2_kernel(
    partial_dw_ptr,
    partial_db_ptr,
    dw_ptr,
    db_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_r = tl.arange(0, BLOCK_R)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_r[:, None] < NUM_ROW_BLOCKS) & (offs_n[None, :] < N)
    ptrs = offs_r[:, None] * N + offs_n[None, :]

    vals_dw = tl.load(partial_dw_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    vals_db = tl.load(partial_db_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    dw = tl.sum(vals_dw, axis=0)
    db = tl.sum(vals_db, axis=0)

    tl.store(dw_ptr + offs_n, dw, mask=offs_n < N)
    tl.store(db_ptr + offs_n, db, mask=offs_n < N)


@triton.jit
def _layernorm_fused_row_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    dy_ptr,
    out_flat_ptr,
    mean_ptr,
    rstd_ptr,
    TOTAL_X: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = row * N + offs

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N

    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    xhat = diff * rstd

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    # Forward output.
    y = xhat * w + b
    tl.store(out_flat_ptr + idx, y, mask=mask)

    # Backward dx.
    dyw = tl.where(mask, dy * w, 0.0)
    sum_dyw = tl.sum(dyw, axis=0)
    sum_dyw_xhat = tl.sum(dyw * xhat, axis=0)
    dx = rstd * (dyw - (sum_dyw + xhat * sum_dyw_xhat) / N)

    tl.store(out_flat_ptr + TOTAL_X + idx, dx, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def _row_block_n(N: int) -> int:
    return triton.next_power_of_2(N)


def _num_warps_for_block(block: int) -> int:
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    if block >= 256:
        return 4
    return 1


def layernorm_forward_with_saved(x, weight, bias, eps):
    """
    LayerNorm forward over the last dimension.

    Returns:
        y: normalized output with affine scale/bias
        saved: tuple consumed by layernorm_backward_from_saved

    All arithmetic is performed by Triton. Python only allocates and launches.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()

    N = weight.numel()
    total = x.numel()
    assert N > 0
    assert total % N == 0
    M = total // N
    assert bias.numel() == N

    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = _row_block_n(N)
    num_warps = _num_warps_for_block(BLOCK_N)

    _layernorm_forward_kernel[(M,)](
        x,
        weight,
        bias,
        y,
        mean,
        rstd,
        M,
        N,
        float(eps),
        BLOCK_N,
        num_warps=num_warps,
    )

    saved = (x, weight, bias, mean, rstd, float(eps))
    return y, saved


def layernorm_backward_from_saved(upstream, saved):
    """
    Backward pass for LayerNorm.

    Computes gradients with respect to:
      - x
      - weight
      - bias

    All reductions and elementwise math are performed in Triton kernels.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, weight, bias, mean, rstd, eps = saved

    assert dy.is_cuda and x.is_cuda and weight.is_cuda and bias.is_cuda
    assert dy.is_contiguous()
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()

    N = weight.numel()
    total = x.numel()
    assert N > 0
    assert total % N == 0
    M = total // N

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)

    BLOCK_N_ROW = _row_block_n(N)
    num_warps_row = _num_warps_for_block(BLOCK_N_ROW)

    _layernorm_backward_dx_kernel[(M,)](
        dy,
        x,
        weight,
        mean,
        rstd,
        dx,
        M,
        N,
        BLOCK_N_ROW,
        num_warps=num_warps_row,
    )

    BLOCK_M = 32
    BLOCK_N = 64
    if N < BLOCK_N:
        BLOCK_N = triton.next_power_of_2(N)

    num_row_blocks = triton.cdiv(M, BLOCK_M)
    partial_dw = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)
    partial_db = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)

    grid_stage1 = (triton.cdiv(N, BLOCK_N), num_row_blocks)
    _layernorm_param_grad_stage1_kernel[grid_stage1](
        dy,
        x,
        mean,
        rstd,
        partial_dw,
        partial_db,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=4,
    )

    BLOCK_R = triton.next_power_of_2(num_row_blocks)
    grid_stage2 = (triton.cdiv(N, BLOCK_N),)
    _layernorm_param_grad_stage2_kernel[grid_stage2](
        partial_dw,
        partial_db,
        dw,
        db,
        num_row_blocks,
        N,
        BLOCK_R,
        BLOCK_N,
        num_warps=4,
    )

    return dx, dw, db


def kernel_function(x, weight, bias, eps, upstream):
    """
    Fused forward+backward oracle step for LayerNorm.

    This returns a single flat float32 tensor matching:
        torch.cat([
            output.float().reshape(-1),
            dx.float().reshape(-1),
            dweight.float().reshape(-1),
            dbias.float().reshape(-1),
        ])

    Fusion performed:
      - A row-wise Triton kernel computes LayerNorm mean/rstd, forward output,
        and dx in one pass over each row.
      - dweight/dbias require reductions across rows, so they are computed by
        a two-stage Triton reduction and written directly into the tail of the
        flat output tensor.

    The wrapper performs only validation, allocation, grid setup, and launches.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    assert x.is_cuda and weight.is_cuda and bias.is_cuda and dy.is_cuda
    assert x.is_contiguous()
    assert dy.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()

    N = weight.numel()
    total_x = x.numel()
    assert N > 0
    assert total_x % N == 0
    assert bias.numel() == N
    assert dy.numel() == total_x

    M = total_x // N
    total_out = total_x * 2 + N * 2

    out = torch.empty((total_out,), device=x.device, dtype=torch.float32)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N_ROW = _row_block_n(N)
    num_warps_row = _num_warps_for_block(BLOCK_N_ROW)

    _layernorm_fused_row_kernel[(M,)](
        x,
        weight,
        bias,
        dy,
        out,
        mean,
        rstd,
        total_x,
        M,
        N,
        float(eps),
        BLOCK_N_ROW,
        num_warps=num_warps_row,
    )

    BLOCK_M = 32
    BLOCK_N = 64
    if N < BLOCK_N:
        BLOCK_N = triton.next_power_of_2(N)

    num_row_blocks = triton.cdiv(M, BLOCK_M)
    partial_dw = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)
    partial_db = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)

    grid_stage1 = (triton.cdiv(N, BLOCK_N), num_row_blocks)
    _layernorm_param_grad_stage1_kernel[grid_stage1](
        dy,
        x,
        mean,
        rstd,
        partial_dw,
        partial_db,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=4,
    )

    dw_offset = total_x * 2
    db_offset = total_x * 2 + N

    BLOCK_R = triton.next_power_of_2(num_row_blocks)
    grid_stage2 = (triton.cdiv(N, BLOCK_N),)
    _layernorm_param_grad_stage2_kernel[grid_stage2](
        partial_dw,
        partial_db,
        out[dw_offset:],
        out[db_offset:],
        num_row_blocks,
        N,
        BLOCK_R,
        BLOCK_N,
        num_warps=4,
    )

    return out
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'layernorm', 'arguments': ['x', 'weight', 'bias', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias', 'None'], 'scalars': ['eps'], 'deployment_entry': 'layernorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class LayernormFunction(torch.autograd.Function):
    """Static autograd wiring for `layernorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, saved = layernorm_forward_with_saved(x, weight, bias, eps)
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
        dx, dweight, dbias = layernorm_backward_from_saved(
            dy, saved, **_backward_scalars(layernorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias, None


def layernorm_deployment(x, weight, bias, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = LayernormFunction.apply(x, weight, bias, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class LayernormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return layernorm_deployment(x, self.weight, self.bias, self.eps)


DEPLOYMENT_ENTRY = "layernorm_deployment"
