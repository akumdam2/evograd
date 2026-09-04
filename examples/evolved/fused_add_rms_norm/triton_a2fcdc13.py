import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START


@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    X,
    R,
    W,
    OUT,
    SUMMED,
    RSTD,
    N_COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N_COLS

    x = tl.load(X + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

    s = x + r
    ss = tl.sum(s * s, axis=0)
    rstd = tl.rsqrt(ss / N_COLS + EPS)
    out = s * rstd * w

    tl.store(SUMMED + row * N_COLS + offs, s, mask=mask)
    tl.store(OUT + row * N_COLS + offs, out, mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _zero_1d_kernel(
    X,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    z = tl.zeros((BLOCK_N,), dtype=tl.float32)
    tl.store(X + offs, z, mask=mask)


@triton.jit
def _fused_add_rms_norm_bwd_kernel(
    DOUT,
    DSUMMED,
    SUMMED,
    RSTD,
    W,
    DX,
    DWEIGHT,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DOUT: tl.constexpr,
    HAS_DSUMMED: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    ptrs = rows[:, None] * N_COLS + cols[None, :]

    s = tl.load(SUMMED + ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]

    if HAS_DOUT:
        dout = tl.load(DOUT + ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        dout = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if HAS_DSUMMED:
        dsummed = tl.load(DSUMMED + ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        dsummed = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    dnorm = dout * w
    mean_dot = tl.sum(dnorm * s, axis=1)[:, None] / N_COLS
    dtotal = dsummed + rstd * (
        dnorm - s * mean_dot * rstd * rstd
    )
    tl.store(DX + ptrs, dtotal, mask=mask)

    dw = tl.sum(dout * s * rstd, axis=0)
    tl.atomic_add(DWEIGHT + cols, dw, mask=col_mask, sem="relaxed")


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _num_warps(block_n: int) -> int:
    if block_n >= 2048:
        return 8
    if block_n >= 1024:
        return 4
    return 1


def _check_supported_tensor(name, t, ndim=None):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and t.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"{name} has unsupported dtype {t.dtype}")


def _check_cols_supported(cols: int):
    if cols <= 0:
        raise ValueError("last dimension must be non-empty")
    block_n = _next_power_of_2(cols)
    if block_n > 131072:
        raise ValueError("last dimension is too large for the single-block Triton RMSNorm kernel")
    return block_n


def fused_add_rms_norm_forward_with_saved(x, r, weight, eps=1e-6):
    _check_supported_tensor("x", x, ndim=2)
    _check_supported_tensor("r", r, ndim=2)
    _check_supported_tensor("weight", weight, ndim=1)

    if r.shape != x.shape:
        raise ValueError("r must have the same shape as x")
    if r.device != x.device:
        raise ValueError("r must be on the same device as x")
    if weight.device != x.device:
        raise ValueError("weight must be on the same device as x")
    if r.dtype != x.dtype:
        raise TypeError("r must have the same dtype as x")
    if weight.shape[0] != x.shape[1]:
        raise ValueError("weight length must match x.shape[1]")
    if weight.dtype not in (x.dtype, torch.float32):
        raise TypeError("weight dtype must match x dtype or be torch.float32")

    rows, cols = x.shape
    if rows <= 0:
        raise ValueError("row dimension must be non-empty")
    block_n = _check_cols_supported(cols)
    eps_f = float(eps)

    out = torch.empty_like(x)
    summed = torch.empty_like(x)
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)

    _fused_add_rms_norm_fwd_kernel[(rows,)](
        x,
        r,
        weight,
        out,
        summed,
        rstd,
        cols,
        eps_f,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
    )

    saved_tensors = (summed, rstd, weight)
    return (out, summed), saved_tensors


def fused_add_rms_norm_backward_from_saved(output_grads, saved_tensors, eps=1e-6):
    if not isinstance(output_grads, (tuple, list)) or len(output_grads) != 2:
        raise TypeError("output_grads must be a tuple/list of length 2: (dout, dsummed)")
    if not isinstance(saved_tensors, (tuple, list)) or len(saved_tensors) != 3:
        raise TypeError("saved_tensors must be a tuple/list of length 3")

    dout, dsummed = output_grads
    summed, rstd, weight = saved_tensors

    _check_supported_tensor("summed", summed, ndim=2)
    _check_supported_tensor("rstd", rstd, ndim=1)
    _check_supported_tensor("weight", weight, ndim=1)

    rows, cols = summed.shape
    if rows <= 0:
        raise ValueError("row dimension must be non-empty")
    block_n = _check_cols_supported(cols)

    if rstd.shape[0] != rows:
        raise ValueError("rstd length must match summed.shape[0]")
    if rstd.dtype != torch.float32:
        raise TypeError("rstd must have dtype torch.float32")
    if weight.shape[0] != cols:
        raise ValueError("weight length must match summed.shape[1]")
    if weight.device != summed.device or rstd.device != summed.device:
        raise ValueError("saved tensors must be on the same device")
    if weight.dtype not in (summed.dtype, torch.float32):
        raise TypeError("weight dtype must match summed dtype or be torch.float32")

    has_dout = dout is not None
    has_dsummed = dsummed is not None

    if has_dout:
        _check_supported_tensor("dout", dout, ndim=2)
        if dout.shape != summed.shape:
            raise ValueError("dout must have the same shape as summed")
        if dout.device != summed.device:
            raise ValueError("dout must be on the same device as summed")
        if dout.dtype != summed.dtype:
            raise TypeError("dout must have the same dtype as summed")

    if has_dsummed:
        _check_supported_tensor("dsummed", dsummed, ndim=2)
        if dsummed.shape != summed.shape:
            raise ValueError("dsummed must have the same shape as summed")
        if dsummed.device != summed.device:
            raise ValueError("dsummed must be on the same device as summed")
        if dsummed.dtype != summed.dtype:
            raise TypeError("dsummed must have the same dtype as summed")

    dx = torch.empty_like(summed)
    dweight_accum = torch.empty((cols,), device=summed.device, dtype=torch.float32)

    zero_block = 1024
    _zero_1d_kernel[(triton.cdiv(cols, zero_block),)](
        dweight_accum,
        cols,
        BLOCK_N=zero_block,
        num_warps=4,
    )

    dout_ptr = dout if has_dout else summed
    dsummed_ptr = dsummed if has_dsummed else summed

    block_m = 2 if block_n <= 1024 else 1
    bwd_warps = min(8, block_m * _num_warps(block_n))
    _fused_add_rms_norm_bwd_kernel[(triton.cdiv(rows, block_m),)](
        dout_ptr,
        dsummed_ptr,
        summed,
        rstd,
        weight,
        dx,
        dweight_accum,
        rows,
        cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_DOUT=has_dout,
        HAS_DSUMMED=has_dsummed,
        num_warps=bwd_warps,
    )

    if weight.dtype == torch.float32:
        dweight = dweight_accum
    else:
        dweight = dweight_accum.to(weight.dtype)

    # Both derivatives are the same dtotal tensor. Aliasing is safe here:
    # callers receive the exact gradient of each independent input.
    return dx, dx, dweight


# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'fused_add_rms_norm', 'arguments': ['x', 'r', 'weight', 'eps'], 'outputs': ['out', 'summed'], 'upstream_grads': ['dout', 'dsummed'], 'input_grads': ['dx', 'dr', 'dweight'], 'gradient_return_order': ['dx', 'dr', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'fused_add_rms_norm_deployment'}


class FusedAddRmsNormFunction(torch.autograd.Function):
    """Static autograd wiring for `fused_add_rms_norm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, r, weight, eps):
        (out, summed), saved = fused_add_rms_norm_forward_with_saved(x, r, weight, eps)
        ctx.save_for_backward(*saved)
        ctx.eps = eps
        return out, summed

    @staticmethod
    def backward(ctx, dout, dsummed):
        dx, dr, dweight = fused_add_rms_norm_backward_from_saved(
            (dout, dsummed), ctx.saved_tensors, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dr, dweight, None


def fused_add_rms_norm_deployment(x, r, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        r = r.reshape(-1, r.shape[-1])
    out, summed = FusedAddRmsNormFunction.apply(x, r, weight, eps)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return out, summed


class FusedAddRmsNormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, r):
        return fused_add_rms_norm_deployment(x, r, self.weight, self.eps)


DEPLOYMENT_ENTRY = "fused_add_rms_norm_deployment"
