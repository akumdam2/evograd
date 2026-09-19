# Llama report-first evaluation, evolved candidates — 2026-09-18

**None of the evolved configurations beats whole-model `torch.compile`. The
residual candidate is the best single site at 1.099× eager (0.953× compile);
the MLP candidate halves throughput, and it turns the three-site combination
from the 2026-09-17 run's 1.20× into 0.51×.** All eight configurations
completed timing and 200 training steps; every patched configuration has a
complete model-level verdict, all within limits.

This repeats the Llama half of
[benchmark_run_20260917.md](benchmark_run_20260917.md) with the four kernels
from the 2026-09-12 evolution run in place of that report's candidates. No
kernel was generated or modified for this run. Raw data are local under
`llama_mine/20260918/` (`RUN` below).

## What differs from 2026-09-17

| | 2026-09-17 | This run |
| --- | --- | --- |
| Llama candidates | `c32041fbb8e1`, `dda5b0065854`, `b0e5414d6e4a`, `537071c5e144` | `1bc20342a06e`, `161dced02de8`, `403542f7da9b`, `781744cb6731` (evolved 2026-09-12) |
| Software | PyTorch 2.11.0+cu128, Triton 3.6.0, Transformers 5.16.1 | PyTorch 2.14.0+cu130, Triton 3.8.0, Transformers 5.17.0 |
| Model-level verdict | none bound for Llama | bound model gate (`environment_hash 2e986434ce93`) for all six patched rows |
| Block calibration controls | structural and bound-pair | structural, bound-pair **and** site `torch.compile` |
| Target-shape Tier 2 | not run (listed as remaining work) | run for all four operators |
| Training "compile" row | whole-model compile | all four sites compiled separately |
| Training peak memory | recorded | not recorded |

Same in both: GH200, BF16, batch 2, sequence 2048, SDPA, seeded weights and
synthetic tokens, `report-first` enforcement, whole-model timing over
forward/loss + backward + AdamW + gradient reset with 5 warmups and 5 blocks ×
20 steps, and 200-step training with validation at steps 0/50/100/200.

Because the stacks differ, absolute times and losses are **not** comparable
between the two runs. Ratios against each run's own eager and compile are.

## Whole-model results

| Configuration | Step ms | × eager | × compile | 95% CI (× eager) | Model verdict |
| --- | ---: | ---: | ---: | --- | --- |
| Eager | 107.05 | 1.000 | 0.868 | — | unpatched, not gated |
| Whole-model compile | 92.88 | **1.153** | 1.000 | 1.149–1.156 | unpatched, not gated |
| QKV | 104.38 | 1.026 | 0.890 | 1.023–1.029 | within limits |
| Attention | 432.71 | 0.247 | 0.215 | 0.247–0.248 | within limits |
| MLP | 225.54 | 0.475 | 0.412 | 0.473–0.476 | within limits |
| Residual | 97.41 | **1.099** | 0.953 | 1.096–1.102 | within limits |
| QKV + MLP + residual | 211.98 | 0.505 | 0.438 | 0.504–0.506 | within limits |
| All four | 527.46 | 0.203 | 0.176 | 0.202–0.203 | within limits |

Every provider ran in its own process. Per-block step times vary by under
1%, so the intervals are narrow and every row differs from eager.

### Against 2026-09-17

| Configuration | 09-17 × eager | This run × eager | 09-17 × compile | This run × compile |
| --- | ---: | ---: | ---: | ---: |
| Whole-model compile | 1.144 | 1.153 | 1.000 | 1.000 |
| QKV | 1.052 | 1.026 | 0.919 | 0.890 |
| Attention | 0.690 | 0.247 | 0.603 | 0.215 |
| MLP | 1.015 | **0.475** | 0.887 | 0.412 |
| Residual | 1.114 | 1.099 | 0.973 | 0.953 |
| QKV + MLP + residual | **1.201** | 0.505 | **1.049** | 0.438 |
| All four | 0.778 | 0.203 | 0.679 | 0.176 |

- **The MLP site decides the combination.** The 09-17 MLP runs at eager speed;
  the evolved one costs 118 ms per step. The three-site combination loses its
  lead over compile entirely because of it.
- **Residual and QKV land close to the 09-17 kernels**: 1.099× against 1.114×,
  and 1.026× against 1.052×.
- **Whole-model compile moved by under 1%** between stacks (1.144× and 1.153×),
  so the gap between the two runs' candidates is the candidates, not the stack.
- **Attention is 2.8× slower than the 09-17 attention** (0.247× against 0.690×
  eager).

## How block timings predict the model

The block run times one decoder layer; the model has 16. For the sites that
occur once per layer, the block's per-layer cost times 16 predicts the model's
per-step cost:

| Configuration | Block: extra ms per layer | × 16 | Model: extra ms per step |
| --- | ---: | ---: | ---: |
| MLP | +7.55 | +120.8 | +118.5 |
| Attention | +20.65 | +330.5 | +325.7 |
| All four | +26.99 | +431.8 | +420.4 |

The MLP and attention losses are GPU execution, and the prediction holds to
within 3%.

**Residual does not follow this rule, and should not.** The block contains one
of its two calls per layer, the post-attention one; the model has 32 calls,
including the cross-layer fusion and the final norm. The block saving is
0.08 ms per layer with an interval of 0 to 0.42 ms; even its upper bound times
16 (6.7 ms) is below the 9.6 ms the model saves. At least 30% of the residual
candidate's model-level gain therefore comes from calls the block cannot see.

## Block results (layer 8, captured)

| Configuration | fwd+bwd ms | × native | 95% CI | Block output check |
| --- | ---: | ---: | --- | --- |
| Native | 4.478 | 1.000 | — | reference |
| Residual | 4.402 | 1.017 | 1.000–1.104 | **exceeds**: rel L2 2.71e-3 against 8.58e-5 (31.6×) |
| QKV | 4.529 | 0.989 | 0.964–1.097 | within limits |
| QKV + MLP + residual | 11.564 | 0.387 | 0.381–0.414 | within the combination limit |
| MLP | 12.026 | 0.372 | 0.366–0.398 | within limits |
| Attention | 25.132 | 0.178 | 0.175–0.191 | **exceeds**: rel L2 1.96e-3 against 2.0e-5 (97.8×) |
| All four | 31.463 | 0.142 | 0.140–0.152 | within the combination limit |

One round (order seed 11); 5 warmups, 5 blocks × 10 samples. Under
`report-first` the two mismatches were recorded and the rows were still timed.
Live per-invocation checks passed for every row. These results reproduce the
2026-09-16 strict-mode block run: the same two sites fail by the same margins,
because they are the same kernels.

The two combinations pass only because the combination limit (4.69e-3) is set
by the compiled QKV site's own deviation and is wide enough to admit both
failing sites. A passing combination here says nothing about its members.

**This table is not comparable with 09-17's block verdicts.** That run
calibrated without the site-compile control and reports all 12 patched rows
over the output envelope; this run includes the control, which widens every
limit on a patch set containing QKV or MLP.

**The gates disagree.** Residual and attention fail the block output check by
31.6× and 97.8× and pass the model gate. The model gate's envelope (E/E plus
S/B per parameter group) is wider than the block's per-site limits; neither
verdict overrides the other.

## Target-shape Tier 2

The operator alone at the canonical shape (B=2, T=2048), through its deployment
module and autograd. All four pass the declared correctness gate.

| Operator | Candidate ms | Eager ms | Compile ms | × eager | × compile | Candidate peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3_qkv_rope` | 0.504 | 1.022 | 0.686 | **2.027** | **1.360** | 220 MiB |
| `llama3_attention` | 21.346 | 0.632 | 0.631 | 0.030 | 0.030 | 6,033 MiB |
| `llama3_swiglu_mlp` | 9.917 | 2.423 | 2.324 | 0.244 | 0.234 | 816 MiB |
| `llama3_residual_rmsnorm` | 0.290 | 0.199 | 0.358 | 0.684 | 1.233 | 160 MiB |

The two rows to read against the model table:

- **QKV is 2.03× eager at Tier 2 and 1.026× at model scope.** Tier 2 times the
  operator with nothing queued on the GPU, where the candidate's fewer launches
  count; inside a training step they mostly do not.
- **Residual is 0.684× eager at Tier 2 and 1.099× at model scope**: the
  reverse. Its per-call autograd wrapper outweighs a 0.03 ms kernel on its own,
  while the model gains from fusing across layers.

Tier 2 does not predict the model result in either direction for these two
kernels.

## Training (200 steps, synthetic data)

| Configuration | Val NLL @0 | @50 | @100 | @200 | Δ vs eager @200 | Nonfinite steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 12.17430 | 12.17180 | 12.10526 | 11.92804 | — | 0 |
| Per-site compile | 12.17430 | 12.17332 | 12.10520 | 11.92766 | −0.00037 | 0 |
| QKV | 12.17430 | 12.17290 | 12.10468 | 11.92747 | −0.00056 | 0 |
| Attention | 12.17430 | 12.17395 | 12.10503 | 11.92786 | −0.00017 | 0 |
| MLP | 12.17430 | 12.17195 | 12.10671 | 11.92879 | +0.00075 | 0 |
| Residual | 12.17430 | 12.17384 | 12.10474 | 11.92778 | −0.00025 | 0 |
| QKV + MLP + residual | 12.17430 | 12.17332 | 12.10422 | 11.92768 | −0.00036 | 0 |
| All four | 12.17430 | 12.17308 | 12.10459 | 11.92779 | −0.00025 | 0 |

Validation scores the trained weights in a fresh unpatched model on 8 held-out
synthetic batches (32,752 tokens), so a candidate's own forward cannot affect
its validation number.

- **Every configuration trains without a nonfinite step**, and the candidates
  end within −0.00056 to +0.00075 of eager, against −0.00087 to +0.00020 in the
  09-17 run. The per-site compile lands inside the same band (−0.00037).
- **Per-step training NLL stays within 0.009 of eager** for every configuration.
- **These are execution and optimization diagnostics, not quality results.**
  The tokens are synthetic and the weights random, and 200 steps says nothing
  about long-run equivalence.

This run's eager validation NLL at step 200 is 11.928; the 09-17 run's is
11.918. The two runs used different software stacks, so only the within-run
deltas compare.

## Memory

The timing processes peak at 19.54 GiB (eager), 14.96 GiB (whole-model compile)
and 29.7–35.6 GiB for the patched rows. **The patched peaks include
allocations the model gate retains while checking,** which the unpatched rows
skip, so they are not a kernel memory cost. The 09-17 report hit the same
effect (32.87 GiB timing against 20.03 GiB training for one configuration) and
reported training-process peaks instead. This run's training processes did not
record peak memory, so no clean comparison is available.

The one attributable memory cost is attention's: 6.3 GiB at block scope against
native's 0.81 GiB, consistent with the candidate materializing the full
`[B, HQ, T, T]` score tensor.

## Findings

1. **No evolved configuration beats whole-model compile.** The best, residual,
   reaches 0.953× compile. The 09-17 three-site combination was the only row in
   either run to beat compile (1.049×), and it depended on an MLP kernel that
   runs at eager speed.
2. **Evolved MLP and attention should not be deployed.** Both lose at every
   scope by amounts consistent with their Tier-2 losses, and attention also
   fails the block output check by 97.8×.
3. **Residual is the most useful evolved kernel, and only whole-model timing
   shows it.** Tier 2 reports a slowdown and the block a gain inside its noise,
   while the model gains 9.6 ms per step, mostly from cross-layer calls. It
   still fails the block output check by 31.6×, because it rounds to BF16
   differently from native.
4. **Evolved QKV gives a small, real gain**: 1.026× eager at model scope,
   against 2.03× at Tier 2.

## Limits

- One session, one node, one block round; no cross-node repeats.
- Operator prechecks (`evograd verify`) were not saved with this run.
- Absolute times and losses cannot be compared with 09-17's because the
  software stacks differ.
- The training comparison row is per-site compile, not whole-model compile, and
  training peak memory was not recorded.
- The block combination verdicts pass members that fail on their own; read them
  per site.

## Reproduction

Under `RUN`: `model/step_timing.json` (whole model), `block/round11.json` and
`round11.policy.json` (block, policy `6d27e6f66a52`, case `c78223693e7c`),
`t2-llama3_*.json` (Tier 2) and `training/<configuration>.json` (training).

`scripts/run_llama_report_first.sh` runs all stages. The model stage needs
about 200 GiB of host memory; below that the patched workers are killed while
checking. Training uses `scripts/llama_training_study.py`.

| Site | Candidate | SHA256 prefix |
| --- | --- | --- |
| QKV | `best-llama3_qkv_rope.py` | `1bc20342a06e4ecc` |
| Attention | `best-llama3_attention.py` | `161dced02de86262` |
| MLP | `best-llama3_swiglu_mlp.py` | `403542f7da9b4353` |
| Residual | `best-llama3_residual_rmsnorm.py` | `781744cb6731520f` |

The result files do not record candidate hashes; these are the local copies in
`evolve/`, matching the hashes the evolution report recorded for the files it
evaluated.
