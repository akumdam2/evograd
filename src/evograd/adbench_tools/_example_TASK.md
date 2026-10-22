# Task llama3_swiglu_mlp.bf16.B2_H2048_I8192_T2048

Write a training-ready forward/backward pair for `llama3_swiglu_mlp` that is
faster than PyTorch eager autograd at the configuration below, and exactly as
correct as the declared tolerances allow.

## 1. Instructions (identical to the evograd agent's)

You are an expert Triton GPU programmer optimizing an autograd-pair kernel.

Task:
Improve the code inside the EVOLVE-BLOCK. The public API is a flexible
forward/backward pair:

    def llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
        return out, saved_tensors

    def llama3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
        return dx, dgate_weight, dup_weight, ddown_weight

You may modify which tensors the forward saves, as long as backward consumes
the same saved_tensors contract and correctness is preserved.

Operator semantics:
- Qwen3's gated MLP block. gate = x @ gate_weight.T; up = x @ up_weight.T; hidden = silu(gate) * up computed with float32 accumulation and cast back to x's dtype; out = hidden @ down_weight.T. x is [B, T, H], gate_weight and up_weight are [I, H], down_weight is [H, I], out is [B, T, H]. silu(v) = v * sigmoid(v). Accumulate every GEMM in float32. Do not call F.linear, torch.matmul, @, F.silu, or autograd in the generated math.
- Return gradients for x, gate_weight, up_weight, and down_weight IN THIS ORDER. With g = silu(gate) * up (the float32 intermediate) and dh = dout @ down_weight: ddown_weight = dout^T @ hidden (shape [H, I]); dup = dh * silu(gate); dgate = dh * up * dsilu(gate) where dsilu(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v))); dgate_weight = dgate^T @ x and dup_weight = dup^T @ x (both [I, H], summed over B and T); dx = dgate @ gate_weight + dup @ up_weight (shape [B, T, H]). Accumulate every reduction and matmul in float32 before casting each gradient to its input's dtype.
- Activity: x: Active, gate_weight: Active, up_weight: Active, down_weight: Active (Inactive inputs never receive gradients).
- Additional constraints: Derived from the published Llama-3-8B configuration, not chosen. The boundary is the whole gated MLP: gate and up projections, the SwiGLU activation, and the down projection. No bias anywhere. All floating tensors are CUDA tensors.

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

Source: evograd.benchmark.topdown.llama3_2_1b.levels.level2.swiglu_mlp.reference:llama3_swiglu_mlp_forward_ref
The oracle differentiates this function with torch.autograd; candidates
must not call it.
"""

import torch
import torch.nn.functional as F


def gated_mlp_forward_ref(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] != gate_weight.shape[-1]:
        raise ValueError(
            f"x's last dim {x.shape[-1]} must match gate_weight's {gate_weight.shape[-1]}"
        )
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape [I, H]")
    if down_weight.shape != (gate_weight.shape[1], gate_weight.shape[0]):
        raise ValueError(
            f"down_weight must be [H, I] = "
            f"{(gate_weight.shape[1], gate_weight.shape[0])}, got {tuple(down_weight.shape)}"
        )

    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = F.silu(gate.float()) * up.float()
    hidden = hidden.to(x.dtype)
    return F.linear(hidden, down_weight)

reference_forward = gated_mlp_forward_ref
```

Fixed configuration: B=2, T=2048, H=2048, I=8192, dtype bfloat16, one CUDA device.

- `x`: [B, T, H] = [2, 2048, 2048], bfloat16 (active, receives gradient `dx`)
- `gate_weight`: [I, H] = [8192, 2048], bfloat16 (active, receives gradient `dgate_weight`)
- `up_weight`: [I, H] = [8192, 2048], bfloat16 (active, receives gradient `dup_weight`)
- `down_weight`: [H, I] = [2048, 8192], bfloat16 (active, receives gradient `ddown_weight`)
- output `out`: [B, T, H] = [2, 2048, 2048], bfloat16 (upstream gradient `dout`)

Correctness tolerances at this configuration, per output and gradient
(`|candidate - oracle| <= atol + rtol * |oracle|`, elementwise):

- `out`: atol=0.0488, rtol=0.024
- `dx`: atol=0.0468, rtol=0.01
- `dgate_weight`: atol=1.17, rtol=0.01
- `dup_weight`: atol=1.55, rtol=0.01
- `ddown_weight`: atol=2.05, rtol=0.01

## 3. Deliverable and rules

Return ONE self-contained Python module (source code only) that defines, at
module level:

```python
def llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    return out, saved_tensors

def llama3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
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
  these reserved names: `ARTIFACT_CONTRACT`, `DEPLOYMENT_ENTRY`, `Llama3SwigluMlpFunction`, `Llama3SwigluMlpModule`, `llama3_swiglu_mlp_deployment`, `vendor_gemm`.
- Every line of your module is checked statically. Rejected:
  - calls `torch.<name>` for: addbmm, addmm, baddbmm, batch_norm, bmm, chain_matmul, compile, dot, einsum, group_norm, inner, kron, layer_norm, log_softmax, matmul, matrix_power, mm, outer, rms_norm, scaled_dot_product_attention, softmax, tensordot;
  - any private `torch._*` entry point, and anything under torch._C, torch._VF, torch._dynamo, torch._ops, torch.autograd, torch.fx, torch.jit, torch.nn.attention, torch.nn.functional, torch.ops;
  - imports of or calls into: F, evograd, functional, liger_kernel, transformers;
  - methods `.<name>()` on any object for: addmm, backward, baddbmm, bmm, einsum, linear, log_softmax, matmul, mm, softmax;
  - the `@` operator.
- Library primitives this task grants:
  - `vendor_gemm`: 2-D row-major GEMM through cuBLAS (torch.mm), no autograd. The harness defines it outside your module; call it by name, do not define or import it.
  - It is the only way to reach a vendor contraction: every other contraction must be written in Triton (`tl.dot`).
  - To run your module on its own, add `from adbench_primitives import vendor_gemm` (the task pack ships that file); the harness removes that line and supplies its own definition.

## 4. How it is scored

The same evaluator scores every agent. A candidate that fails the static
rules, the import, any output or gradient check, or a run at the timed shape
scores as incorrect. A correct candidate is timed at the configuration above
against the `pytorch_autograd` baseline: forward, backward from
saved state, and the forward+backward step, with saved-tensor bytes reported
relative to input bytes. `combined_score` is evograd's `speed_memory`
policy over those measurements -- higher is better.
