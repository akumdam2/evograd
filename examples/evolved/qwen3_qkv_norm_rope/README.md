# Evolved hybrid `qwen3_qkv_norm_rope`

`hybrid_40abfaca.py` — one selected candidate, kept byte-for-byte as it was
evaluated. It is here as evidence, not as a dependency: nothing in `src/`
imports it, and it is not on any measured path.

    sha256  40abfacaccd30f653b187de0ab6a87dbb801194e2bdda8b48f69e11d01764bcb

Verify with `sha256sum hybrid_40abfaca.py`. Do not reformat it — the hash is how
a report ties a number to a program.

## What it is

An **evolved hybrid Level-2 operator**, not one fully evolved Triton kernel. The
dense contractions go to cuBLAS through the granted `vendor_gemm` primitive; the
per-head Q/K RMSNorm, RoPE, their fused backward, the layout changes and the
reductions are evolved Triton. Calling it "an evolved kernel" would overclaim.

| | |
|---|---|
| Pipeline | A (AtenIR-grounded synthesis), GPT-5.5 |
| Capability | `vendor_gemm` (2-D cuBLAS GEMM via `torch.mm`, no autograd) |
| Selection | iteration 9 of 10, combined score 4.6355, chosen on the search score alone |
| Composition | 9 Triton kernels, 12 `vendor_gemm` call sites |
| Contract | `evograd-artifact/1`, `allowed_primitives = ['vendor_gemm']` |
| Deployment | `qwen3_qkv_norm_rope_deployment`, static `torch.autograd.Function`, no `opdecl.bind` |

Seed: `bccce375c1318383479c7f2ec98e87444ad58bec485521a5b26721b84e334af9`
(accepted on codegen attempt 2 of a maximum 3). Selection used no tier-3 result.

## Environment the numbers came from

NVIDIA GH200 120GB (sm_90), driver 595.71.05 · CUDA 12.8 · torch 2.11.0+cu128 ·
triton 3.6.0 · transformers 5.16.1 · Python 3.12.13. Every figure below is from
that machine; none of it transfers to another GPU without remeasuring.

## Tier 1 — correctness and fair paired timing

All four declared correctness workloads pass, and so does the exact observed
Qwen3-0.6B workload (`B2 T2048 H1024 HQ16 HK8 D128 QO2048 KVO1024`, bfloat16) on
every output and gradient, with the required head-major non-contiguous q/k/v
strides reproduced exactly — `q (4194304, 128, 2048, 1)`,
`k`/`v (2097152, 128, 1024, 1)`.

Paired fair runner, 5 blocks × 50 reps, 10 warm-ups, shuffled order,
20 000 bootstrap resamples:

| vs | pair ms (this) | pair ms (other) | speedup | 95% CI |
|---|---:|---:|---:|---|
| native eager | 0.4933 | 1.7224 | 3.4914× | [3.3951, 3.8110] |
| `torch.compile` | 0.5014 | 0.8465 | 1.6882× | [1.6434, 1.7249] |
| Triton-only candidate | 0.4863 | 1.0653 | 2.1905× | [2.1213, 2.2498] |

## Tier 2 — strict, one process per provider

Exact observed shapes and layouts, identical seeded inputs and independent
`dq/dk/dv`, correctness and compilation outside timing, ten untimed warm-ups,
`do_bench(rep=500, quantiles=[0.5, 0.2, 0.8])`, five blocks, provider order
shuffled per block.

| provider | forward ms | full step ms | correct |
|---|---:|---:|---|
| native eager | 0.5630 | 1.6251 | 9/9 |
| `torch.compile` | 0.2511 | 0.8178 | 9/9 |
| Triton-only candidate | 0.3299 | 1.0860 | 9/9 |
| **this artifact** | **0.2056** | **0.7221** | 9/9 |

Block-bootstrap speedups, 95% CI: **2.2505× [2.1818, 2.8040]** over eager,
**1.1325× [1.0685, 1.2173]** over `torch.compile`, **1.5040× [1.4600, 1.5756]**
over the Triton-only candidate. No interval crosses 1.0.

Per iteration it issues 24 kernel launches — fewer than `torch.compile`'s 27 and
eager's 98 — for 396.3 µs of device time, split 172.8 µs evolved Triton /
143.3 µs vendor GEMM / 80.2 µs ATen. The Triton-only candidate spends 1049.2 µs,
98.3% of it in its own hand-written matmuls. Peak allocated 0.659 GiB.

## Tier 3 is still under review — do not treat this as accepted

**This artifact has not passed the Qwen3 whole-model gate.** Patched into the
live `qkv_norm_rope` site it clears site preflight, provider purity, the full
live-boundary validation (4/4 patched and 4/4 carried invocations, zero
failures) and finiteness, then fails the numerical envelope:

    logits rel_l2 = 1.097e-02  >  6.975e-03   (1.57x)

The loss-trajectory and counts stages were never reached. **No end-to-end or
whole-model performance claim should be made from the tier-1 and tier-2 numbers
above.**

Two things are worth knowing before reading that failure as a defect in this
program. `torch.compile` of the declared reference implementation produces
1.0971e-02 on the same measurement, and the Triton-only candidate produces
1.0971e-02 — three structurally unrelated implementations agreeing to three
digits, which points at the characteristic size of any bfloat16 re-association
of this operator rather than at any one kernel. And the threshold itself is
under review: it is derived from a bound-pair reference patched at *every* site,
whose drift is entirely `residual_rmsnorm`'s, while the matched reference for a
QKV-only replacement drifts by exactly zero. See
`src/evograd/bench/workloads/qwen3/README.md`.

Neither of those makes this artifact accepted. It is a candidate whose
model-level acceptance is open.
