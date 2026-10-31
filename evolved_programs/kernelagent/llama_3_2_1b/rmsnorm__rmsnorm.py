"""adbench-tools artifact: task llama_3_2_1b.rmsnorm.rmsnorm, produced by kernelagent.

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
def _rmsnorm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward RMSNorm, one Triton program per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(mean_sq + eps)

    y = x * rstd * w
    tl.store(y_ptr + row * D + offs, y, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_backward_dx_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    rstd_ptr,
    dx_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward RMSNorm wrt x, one Triton program per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    # z_i = dy_i * weight_i
    # dx_i = rstd * z_i - x_i * rstd^3 / D * sum_j(z_j * x_j)
    z = dy * w
    dot = tl.sum(z * x, axis=0)
    dx = z * rstd - x * (rstd * rstd * rstd) * (dot / D)

    tl.store(dx_ptr + row * D + offs, dx, mask=mask)


@triton.jit
def _rmsnorm_backward_dw_kernel(
    x_ptr,
    dy_ptr,
    rstd_ptr,
    dw_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward RMSNorm wrt weight: dw[col] = sum_rows dy*x*rstd."""
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_m in tl.range(0, M, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & col_mask[None, :]

        x = tl.load(
            x_ptr + rows[:, None] * D + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            dy_ptr + rows[:, None] * D + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * rstd[:, None], axis=0)

    tl.store(dw_ptr + cols, acc, mask=col_mask)


@triton.jit
def _rmsnorm_flat_all_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Single-launch forward + backward step for kernel_function.

    Program ids [0, M) compute row-local forward y and dx:
      out[0:N]  = y.flatten()
      out[N:2N] = dx.flatten()

    Program id M computes the cross-row dweight reduction:
      out[2N:2N+D] = dweight

    dweight cannot reuse row-program rstd values inside the same launch because
    Triton has no grid-wide synchronization, so this program recomputes rstd
    while staying in the same fused kernel launch.
    """
    pid = tl.program_id(0)

    if pid < M:
        row = pid
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        flat = row * D + offs

        x = tl.load(x_ptr + flat, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + flat, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / D
        rstd = tl.rsqrt(mean_sq + eps)

        y = x * rstd * w

        z = dy * w
        dot = tl.sum(z * x, axis=0)
        dx = z * rstd - x * (rstd * rstd * rstd) * (dot / D)

        tl.store(out_ptr + flat, y, mask=mask)
        tl.store(out_ptr + N + flat, dx, mask=mask)
    else:
        cols = tl.arange(0, BLOCK_D)
        col_mask = cols < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for start_m in tl.range(0, M, BLOCK_M):
            rows = start_m + tl.arange(0, BLOCK_M)
            mask = (rows[:, None] < M) & col_mask[None, :]

            x = tl.load(
                x_ptr + rows[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            dy = tl.load(
                dy_ptr + rows[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            mean_sq = tl.sum(x * x, axis=1) / D
            rstd = tl.rsqrt(mean_sq + eps)

            acc += tl.sum(dy * x * rstd[:, None], axis=0)

        tl.store(out_ptr + 2 * N + cols, acc, mask=col_mask)


def _block_d(D: int) -> int:
    return triton.next_power_of_2(D)


def _num_warps(block_d: int) -> int:
    if block_d >= 2048:
        return 8
    if block_d >= 1024:
        return 4
    return 1


def _block_m(block_d: int) -> int:
    # Keep the 2-D reduction tile size moderate.
    if block_d >= 8192:
        return 4
    if block_d >= 4096:
        return 8
    return 16


def _forward_output_dtype(x, weight):
    # Metadata-only dtype selection; no tensor computation.
    if x.dtype == torch.float32 or weight.dtype == torch.float32:
        return torch.float32
    return x.dtype


def rmsnorm_forward_with_saved(x, weight, eps):
    """
    RMSNorm forward implemented in Triton.

    Computes:
      y = x * rsqrt(mean(x^2, last_dim) + eps) * weight

    Saves per-row reciprocal RMS for the Triton backward.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()

    D = weight.numel()
    N = x.numel()
    assert D > 0 and N % D == 0
    M = N // D

    block_d = _block_d(D)
    y_dtype = _forward_output_dtype(x, weight)

    y = torch.empty(x.shape, device=x.device, dtype=y_dtype)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    _rmsnorm_forward_kernel[(M,)](
        x,
        weight,
        y,
        rstd,
        M,
        D,
        eps,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
    )

    saved = (x, weight, rstd, eps)
    return y, saved


def _is_saved_tuple(obj):
    return isinstance(obj, (tuple, list)) and len(obj) >= 4


def _normalize_saved_arg(saved, extra):
    """
    Accept multiple evaluator deployment conventions:
      backward_from_saved(upstream, saved)
      backward_from_saved(upstream, output, saved)
      backward_from_saved(upstream, x, weight, rstd, eps)

    This fixes the previous interface failure where the deployment path passed
    three positional arguments.
    """
    if extra:
        last = extra[-1]
        if _is_saved_tuple(last):
            return last

    if _is_saved_tuple(saved):
        return saved

    if extra:
        packed = (saved,) + tuple(extra)
        if _is_saved_tuple(packed):
            return packed

    return saved


def rmsnorm_backward_from_saved(upstream, saved, *extra):
    """
    RMSNorm backward implemented in Triton.

    Returns gradients for differentiable tensor inputs:
      (dx, dweight)

    The variadic tail is intentional for evaluator compatibility; some
    deployment paths call this as backward_from_saved(upstream, output, saved).
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    saved = _normalize_saved_arg(saved, extra)

    x, weight, rstd, eps = saved[:4]

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous() and weight.is_contiguous() and dy.is_contiguous()

    D = weight.numel()
    N = x.numel()
    assert D > 0 and N % D == 0
    M = N // D

    block_d = _block_d(D)
    bm = _block_m(block_d)

    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)

    _rmsnorm_backward_dx_kernel[(M,)](
        x,
        weight,
        dy,
        rstd,
        dx,
        M,
        D,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
    )

    _rmsnorm_backward_dw_kernel[(1,)](
        x,
        dy,
        rstd,
        dweight,
        M,
        D,
        BLOCK_M=bm,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
    )

    return dx, dweight


def kernel_function(x, weight, eps, upstream):
    """
    Forward+backward RMSNorm step required by the evaluator.

    Returns one flat float32 tensor:
      concat(forward_output.flatten(), dx.flatten(), dweight.flatten())

    All computation is performed in Triton. The wrapper only validates metadata,
    allocates the output tensor, and launches the fused kernel.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous() and weight.is_contiguous() and dy.is_contiguous()

    D = weight.numel()
    N = x.numel()
    assert D > 0 and N % D == 0
    M = N // D

    block_d = _block_d(D)
    bm = _block_m(block_d)

    out = torch.empty((2 * N + D,), device=x.device, dtype=torch.float32)

    _rmsnorm_flat_all_kernel[(M + 1,)](
        x,
        weight,
        dy,
        out,
        M,
        D,
        eps,
        N,
        BLOCK_M=bm,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'rmsnorm', 'arguments': ['x', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight'], 'gradient_return_order': ['dx', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'rmsnorm_deployment', 'allowed_primitives': []}


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
            dy, saved, ctx.eps
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
