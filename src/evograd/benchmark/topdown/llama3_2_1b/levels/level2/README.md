# Level 2 — the four fused boundaries one Llama-3 decoder layer decomposes into

A level-2 task is a **boundary**: a cut through the running computation where
every tensor crossing it can be named, captured and differentiated, so that
span can be replaced by one kernel. Level 2 asks whether an implementation can
optimize *across* an operator boundary while preserving autograd semantics.

Where the four cuts fall. One decoder layer, in execution order:

```
1.  s1 = r + mlp_prev;   x1 = RMSNorm(s1)                 llama3_residual_rmsnorm
2.  q,k,v = RoPE(q_proj(x1), k_proj(x1)), v_proj(x1)      llama3_qkv_rope
3.  attn  = o_proj(causal_gqa_sdpa(q, k, v))              llama3_attention
4.  s2 = s1 + attn;      x2 = RMSNorm(s2)                 llama3_residual_rmsnorm
5.  mlp   = down(silu(gate(x2)) * up(x2))                 llama3_swiglu_mlp
6.  s2 and mlp are carried into the next layer's step 1
```

Steps 1 and 4 are the same operator: a residual add fused with the RMSNorm
that consumes its result. That is why it returns `summed` (the un-normalized
`s1`/`s2`) alongside `out` — the residual stream needs it, and the fusion hands it
over rather than forcing a second pass over the same memory.

`residual_rmsnorm` fires twice per layer; the other three once. 32 layers →
**160 invocations** per canonical step.

## The four operators

`SITE_TASKS` in `manifest.py` maps each site package here to the task key it
serves, so the correspondence is looked up rather than inferred from
directory names. Every key is this model's own.

| site package here | task key | contract, reference and capture |
| --- | --- | --- |
| `qkv_rope/` | `llama3_qkv_rope` | `qkv_rope/{task,reference,capture}.py` |
| `attention/` | `llama3_attention` | `attention/{task,reference,capture}.py` |
| `swiglu_mlp/` | `llama3_swiglu_mlp` | `swiglu_mlp/{task,reference,capture}.py` |
| `residual_rmsnorm/` | `llama3_residual_rmsnorm` | `residual_rmsnorm/{task,reference,capture}.py` |

The mathematics of the last three is shared with Qwen3-0.6B and lives once in
`benchmark/topdown/common/level2_references.py`; each model re-exports it from
its own `reference.py`. Judging an implementation is not done here — that is
the verification package (`evograd.verification`).

### `llama3_qkv_rope` — Llama's own boundary

The prefix of `LlamaAttention`: three projections and the rotation, ending with
`(q, k, v)` ready for SDPA.

| | |
| --- | --- |
| args | `x [B,T,H]`, `q_weight [QO,H]`, `k_weight [KVO,H]`, `v_weight [KVO,H]`, and **Inactive** `cos`/`sin` `[1,T,D]` |
| outputs | `q [B,HQ,T,D]`, `k [B,HK,T,D]`, `v [B,HK,T,D]` |
| gradients | `dx, dq_weight, dk_weight, dv_weight` |

Project three ways, reshape to head-major, rotate `q` and `k` with
`t*cos + rotate_half(t)*sin`; `v` is returned unrotated. **There is no
per-head normalization and no `eps`** — this operator contains no
normalization to have one. It does not contain SDPA, `o_proj`, or the RMSNorm
that produced `x`.

**This is the one declaration Llama had to introduce.** Qwen3's
`qwen3_qkv_norm_rope` applies per-head RMSNorm to q and k before RoPE, which
`LlamaAttention` does not — a different computation, not a different shape, so
no parameterization reaches across it.

The outputs are non-contiguous head-major views, because the model reaches them
by view-then-transpose. A kernel may work in any internal layout but must
return that one.

### `llama3_attention` — the observed attention boundary

| | |
| --- | --- |
| args | `q [B,HQ,T,D]`, `k [B,HK,T,D]`, `v [B,HK,T,D]`, `o_weight [H,QO]` |
| output | `out [B,T,H]` |

Causal grouped-query SDPA followed by the output projection. Picks up exactly
where `llama3_qkv_rope` stops. The same mathematics Qwen3-0.6B runs, and the
same shared implementation — Llama's 4:1 query-to-kv grouping and Qwen3's 2:1
are one operator at different widths — but a separate task, carrying this
model's cases and provenance.

### `llama3_swiglu_mlp` — the MLP boundary

| | |
| --- | --- |
| args | `x [B,T,H]`, `gate_weight [I,H]`, `up_weight [I,H]`, `down_weight [H,I]` |
| output | `out [B,T,H]` |

`down(silu(gate(x)) * up(x))`, three biasless projections. Identical in
structure to Qwen3's; only the widths differ (4096/14336 against 1024/3072).
That width is why this task carries an `out` tolerance multiplier of
`(2.4, 2.4)` and `qwen3_swiglu_mlp` carries none: the reduction over a
14336-wide intermediate is the measured cost, and it is charged to the task
that runs it.

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

No `qwen3_` prefix on this one: it is a generic level-2 operator that predates
both workloads.

**On the `qwen3_` prefixes.** Three of the four operator names carry it. That
records which workload first harvested them, not which model runs them — a
rename this repository owes. The declarations are dimension-parameterized and
shared.

## What this package owns

The site packages here own the contracts and, through
`benchmark/topdown/common/level2_references.py`, the reference implementations.
Each `capture.py` derives its invocation by **replaying `layer16.pt`** and
hooking the two points that bracket its boundary. `layer16.pt` stays the only
tensor artifact: nothing but JSON is written here.

## Task identities after the integration

The upstream work reached three of these four boundaries through Qwen3's task
keys and noted that the rename was owed. This is that rename. The logical site
names and the model identifier `llama_3_2_1b` are unchanged.

| site | upstream task key | integrated task key |
| --- | --- | --- |
| `qkv_rope` | `llama3_qkv_rope` | `llama3_qkv_rope` (unchanged) |
| `attention` | `qwen3_attention` | `llama3_attention` |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | `llama3_swiglu_mlp` |
| `residual_rmsnorm` | `fused_add_rms_norm` | `llama3_residual_rmsnorm` |

Three of the four compute the same mathematics Qwen3-0.6B does. The
implementation is shared and lives once in
`benchmark/topdown/common/level2_references.py`; each model re-exports it under
its own symbol. What is *not* shared is the task: the cases, the widths, the
provenance and the tolerances are this model's.

The `out` tolerance multiplier of `(2.4, 2.4)` on `llama3_swiglu_mlp` is the
measured shortfall at Llama-3-8B's 14336-wide intermediate. It belongs to this
task and not to Qwen3's, whose 3072-wide intermediate measured 1.00 and keeps
no entry.

**Calibration artifacts must be regenerated.** Any tolerance report produced
against the old keys — `qwen3_attention`, `qwen3_swiglu_mlp` or
`fused_add_rms_norm` — describes a different task identity than these sites now
resolve to. Identity validation is not bypassed to load one: rerun the
calibration under the new key. Historical reports are left as they were
written.
