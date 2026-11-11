# Level 2 — the three fused boundaries one Llama-3.2-1B decoder layer decomposes into

A level-2 task is a **boundary**: a cut through the running computation where
every tensor crossing it can be named, captured and differentiated, so that
span can be replaced by one kernel. Level 2 asks whether an implementation can
optimize *across* an operator boundary while preserving autograd semantics.

Where the cuts fall. One decoder layer, in execution order:

```
1.  s1 = r + mlp_prev;   x1 = RMSNorm(s1)                 llama3_residual_rmsnorm
2.  attn  = LlamaAttention(x1)                            llama3_attention_block
          = o_proj(causal_gqa_sdpa(RoPE(q_proj(x1)), RoPE(k_proj(x1)), v_proj(x1)))
3.  s2 = s1 + attn;      x2 = RMSNorm(s2)                 llama3_residual_rmsnorm
4.  mlp   = down(silu(gate(x2)) * up(x2))                 llama3_swiglu_mlp
5.  s2 and mlp are carried into the next layer's step 1
```

Steps 1 and 3 are the same operator: a residual add fused with the RMSNorm
that consumes its result. That is why it returns `summed` (the un-normalized
`s1`/`s2`) alongside `out` — the residual stream needs it, and the fusion hands it
over rather than forcing a second pass over the same memory.

`residual_rmsnorm` fires twice per layer except for layer 0's input norm (32
calls per step); the other two once per layer (16 each).

## The three operators

`SITE_TASKS` in `manifest.py` maps each site package here to the task key it
serves. Every key is this model's own.

| site package here | task key | contract and reference |
| --- | --- | --- |
| `attention_block/` | `llama3_attention_block` | `attention_block/{task,reference}.py` |
| `swiglu_mlp/` | `llama3_swiglu_mlp` | `swiglu_mlp/{task,reference}.py` |
| `residual_rmsnorm/` | `llama3_residual_rmsnorm` | `residual_rmsnorm/{task,reference}.py` |

The MLP and residual mathematics, and the SDPA plus output projection inside
the attention block, are shared with Qwen3-0.6B and live once in
`benchmark/topdown/common/level2_references.py`. Judging an implementation is
not done here — that is the verification package (`evograd.verification`).

### `llama3_attention_block` — the whole `LlamaAttention` module

| | |
| --- | --- |
| args | `x [B,T,H]`, `q_weight [QO,H]`, `k_weight [KVO,H]`, `v_weight [KVO,H]`, `o_weight [H,QO]`, and **Inactive** `cos`/`sin` `[1,T,D]` |
| output | `out [B,T,H]` |
| gradients | `dx, dq_weight, dk_weight, dv_weight, do_weight` |

Project three ways, reshape to head-major, rotate `q` and `k` with
`t*cos + rotate_half(t)*sin` (`v` unrotated), causal grouped-query SDPA
(4:1 query-to-kv heads), merge the heads and apply `o_proj`. There is no
per-head normalization and no bias. The runtime spelling is the module's own
arithmetic (RoPE in the model dtype, the SDPA call Transformers makes with
`attn_mask=None, is_causal=True, enable_gqa=True`), which is what the model
reference and the eager baseline run.

### `llama3_swiglu_mlp` — the MLP boundary

| | |
| --- | --- |
| args | `x [B,T,H]`, `gate_weight [I,H]`, `up_weight [I,H]`, `down_weight [H,I]` |
| output | `out [B,T,H]` |

`down(silu(gate(x)) * up(x))`, three biasless projections. Identical in
structure to Qwen3's; only the widths differ.

### `llama3_residual_rmsnorm` — the residual fusion, with two outputs

| | |
| --- | --- |
| args | `x [rows,cols]`, `r [rows,cols]`, `weight [cols]`, **Inactive** `eps` |
| outputs | `out [rows,cols]`, `summed [rows,cols]` |
| gradients | `dx, dr, dweight` |

`s = x + r`; `out = s * rstd * weight` with the reduction in float32;
`summed = s`. The second output is the *un-normalized sum itself*, not a copy
or a recomputation — the next block consumes it, which is why the fusion
returns it rather than forcing a second pass. The backward receives
`(dout, dsummed)` and both paths reach `s`.

## Where the cases come from

The cases are the calls the real-payload capture records
(`evograd.verification.integrations.llama3_2_1b`): every call of each boundary
in a real training step, at the shapes and calls per step the harvest snapshot
records.

**Calibration artifacts must be regenerated** for any task whose key or
boundary changed. Identity validation is not bypassed to load an old one.
