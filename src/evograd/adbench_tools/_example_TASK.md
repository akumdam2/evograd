# Task qwen3_0_6b.qwen3_swiglu_mlp.swiglu_mlp

Write a training-ready forward/backward pair for `qwen3_swiglu_mlp` that is
faster than PyTorch eager autograd at the configuration below, and correct
within the tolerances in section 2.

## 1. Instructions (identical to the evograd agent's)

You are an expert Triton GPU programmer optimizing an autograd-pair kernel.

Task:
Improve the code inside the EVOLVE-BLOCK. The public API is a flexible
forward/backward pair:

    def qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
        return out, saved_tensors

    def qwen3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
        return dx, dgate_weight, dup_weight, ddown_weight

You may modify which tensors the forward saves, as long as backward consumes
the same saved_tensors contract and correctness is preserved.

Operator semantics:
- Qwen3's gated MLP block. gate = x @ gate_weight.T; up = x @ up_weight.T; hidden = silu(gate) * up computed with float32 accumulation and cast back to x's dtype; out = hidden @ down_weight.T. x is [B, T, H], gate_weight and up_weight are [I, H], down_weight is [H, I], out is [B, T, H]. silu(v) = v * sigmoid(v). Accumulate every GEMM in float32. Do not call F.linear, torch.matmul, @, F.silu, or autograd in the generated math.
- Return gradients for x, gate_weight, up_weight, and down_weight IN THIS ORDER. With g = silu(gate) * up (the float32 intermediate) and dh = dout @ down_weight: ddown_weight = dout^T @ hidden (shape [H, I]); dup = dh * silu(gate); dgate = dh * up * dsilu(gate) where dsilu(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v))); dgate_weight = dgate^T @ x and dup_weight = dup^T @ x (both [I, H], summed over B and T); dx = dgate @ gate_weight + dup @ up_weight (shape [B, T, H]). Accumulate every reduction and matmul in float32 before casting each gradient to its input's dtype.
- Activity: x: Active, gate_weight: Active, up_weight: Active, down_weight: Active (Inactive inputs never receive gradients).
- Additional constraints: Observed from a real Qwen3-0.6B training step, not chosen: the timed shape is the [2, 2048, 1024] BF16 activation that all 28 decoder layers pass to their MLP. B and T are kept as separate axes because that is the shape the module actually received; a kernel may flatten them internally. gate_weight and up_weight are separate tensors, as Qwen3 stores them -- not one fused [2I, H] matrix. No biases. All floating tensors are contiguous CUDA tensors.

The evaluator is memory-aware:
- It measures backward_from_saved latency.
- It measures forward+backward full-step latency.
- It reports saved_memory_ratio = saved_bytes / input_bytes.
- The combined score uses a weighted speedup with a small penalty for saved
  tensor memory.

Saved tensor policy:
- Prefer compact saved state such as per-row/per-block statistics over full
  activation-sized tensors when the backward can cheaply reconstruct the
  larger intermediate.
- Avoid saving tensors with the same shape as a large activation unless the
  latency benefit clearly outweighs the memory cost.
- Saved tensors are part of the optimization tradeoff, not free inputs.

Correctness is a hard requirement. Support the benchmark's CUDA dtypes,
preserve output dtypes, and use fp32 accumulation for reductions. Do not
call PyTorch autograd in generated math.

## 2. What you are given

There is no working starting program (at most an unimplemented skeleton):
where section 1 says "the code inside the EVOLVE-BLOCK", read "the module you
write" (section 3). The operator is
defined by the PyTorch forward reference below and nothing else; the
gradients you return are checked against `torch.autograd` applied to it.

```python
"""The PyTorch forward reference this task is defined by.

Source: evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference:qwen3_swiglu_mlp_forward_hf
The oracle differentiates this function with torch.autograd; candidates
must not call it.
"""

import torch
import torch.nn.functional as F


def gated_mlp_forward_model_dtype(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """The spelling ``Qwen3MLP.forward`` actually executes: no float32 upcast.

    Not the declared reference -- kept so the verification can report what the
    upcast costs, rather than leaving the two contracts silently different.
    """
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)

reference_forward = gated_mlp_forward_model_dtype
```

Fixed configuration: B=2, T=2048, H=1024, I=3072, dtype bfloat16, one CUDA device.

- `x`: [B, T, H] = [2, 2048, 1024], bfloat16 (active, receives gradient `dx`)
- `gate_weight`: [I, H] = [3072, 1024], bfloat16 (active, receives gradient `dgate_weight`)
- `up_weight`: [I, H] = [3072, 1024], bfloat16 (active, receives gradient `dup_weight`)
- `down_weight`: [H, I] = [1024, 3072], bfloat16 (active, receives gradient `ddown_weight`)
- output `out`: [B, T, H] = [2, 2048, 1024], bfloat16 (upstream gradient `dout`)

Acceptance compares every output and gradient with the reference, element by
element in float64, on real recorded training calls held out from you:
`|candidate - reference| <= max(atol + rtol * |reference|, envelope)`.
`atol` and `rtol` are the `torch.testing.assert_close` defaults for the result's
dtype (below); `envelope` is 2x the largest elementwise deviation of
`torch.compile` from the reference on that result, measured on other recorded
calls. So each element may be off by the dtype default or by up to 2x
torch.compile's own error, whichever is larger.

- `out` (bfloat16): atol=1e-05, rtol=0.016
- `dx` (bfloat16): atol=1e-05, rtol=0.016
- `dgate_weight` (bfloat16): atol=1e-05, rtol=0.016
- `dup_weight` (bfloat16): atol=1e-05, rtol=0.016
- `ddown_weight` (bfloat16): atol=1e-05, rtol=0.016

## 3. Deliverable and rules

Return ONE self-contained Python module (source code only) that defines, at
module level:

```python
def qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    return out, saved_tensors

def qwen3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
    return dx, dgate_weight, dup_weight, ddown_weight
```

- `saved_tensors` is a tuple of tensors and/or immutable Python scalars; the
  backward receives exactly what the forward saved. Gradients are returned in
  the order `dx, dgate_weight, dup_weight, ddown_weight`, each with its input's shape and
  dtype; inactive arguments get no gradient.
- Write the math as Triton kernels (`@triton.jit`) launched from these two
  functions and/or PyTorch tensor operations not listed below. Allocation,
  shape inspection, views and launches in plain PyTorch are fine.
- The harness wires autograd and the deployment entry from the declaration.
  Do not write a `torch.autograd.Function` or an `nn.Module`, and do not define
  these reserved names: `ARTIFACT_CONTRACT`, `DEPLOYMENT_ENTRY`, `Qwen3SwigluMlpFunction`, `Qwen3SwigluMlpModule`, `_vendor_sdpa_deterministic`, `qwen3_swiglu_mlp_deployment`, `vendor_gemm`, `vendor_sdpa_backward`, `vendor_sdpa_forward`.
- Every line of your module is checked statically. Rejected:
  - calls `torch.<name>` for: addbmm, addmm, baddbmm, batch_norm, bmm, chain_matmul, compile, dot, einsum, group_norm, inner, kron, layer_norm, log_softmax, matmul, matrix_power, mm, outer, rms_norm, scaled_dot_product_attention, softmax, tensordot;
  - any private `torch._*` entry point, and anything under torch._C, torch._VF, torch._dynamo, torch._ops, torch.autograd, torch.fx, torch.jit, torch.nn.attention, torch.nn.functional, torch.ops;
  - imports of or calls into: F, evograd, functional, liger_kernel, transformers;
  - methods `.<name>()` on any object for: addmm, backward, baddbmm, bmm, einsum, linear, log_softmax, matmul, mm, softmax;
  - the `@` operator.
- Library primitives this task grants:
  - `vendor_gemm`: 2-D row-major GEMM through cuBLAS (torch.mm), no autograd. The harness defines it outside your module; call it by name, do not define or import it.
  - It is the only way to reach a vendor contraction or fused attention: every other contraction must be written in Triton (`tl.dot`).
  - To run your module on its own, add `from adbench_primitives import vendor_gemm` (the task pack ships that file); the harness removes that line and supplies its own definition.

## 4. How it is scored

The same evaluator scores every agent. A candidate that fails the static
rules, the import, any output or gradient check, or a run at the timed shape
scores as incorrect. A correct candidate is timed at the configuration above
against the `pytorch_autograd` baseline: forward, backward from
saved state, and the forward+backward step, with saved-tensor bytes reported
relative to input bytes. `combined_score` is evograd's `speed_memory`
policy over those measurements -- higher is better.
