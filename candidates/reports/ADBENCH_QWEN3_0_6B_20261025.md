# ADBench on Qwen3-0.6B: first formal comparison (evograd-evaluation-v2), 2026-10-25

This report covers the first formal run of [`EVALUATION_PLAN.md`](../EVALUATION_PLAN.md). Three agents (evograd, Codex and GEPA) searched six Qwen3-0.6B cases with `gpt-5.5` and at most 10 model requests per case. A zero-LLM frontend seed is included as a reference. All frozen submissions then went through the v2 protocol: Gate 1, operator Gate 2, model Gate 2 and Gate 3 for correctness; T2 and T3 for performance. After the run, the model-scope Gate 2 was found too strict and amended. The frozen submissions were re-evaluated under the amended gate with no new model requests. Both evaluations are reported below.

## Summary

- **Completeness.** All 18 searches and 6 seeds ended with a submission and an explicit terminal status. Every metered case admitted at most 10 requests: 156 of 180 formal, plus 26 of 30 for the smoke run. All four evaluations are v2 with nothing suspended, and the completeness check passes 215 of 215 for both roots.
- **L1.** Every LLM agent beats eager at T2. Accepted geomean vs eager under the amended evaluation: evograd 1.68x, codex 1.76x, gepa 1.73x. GEPA's mlp_activation is rejected because it saves fp8 backward state. The seed alone runs at 0.34x.
- **Model Gate 2 was too strict.** Per-call boundary replay rejected rounding-level candidates on single bf16 ULP flips in one layer's weight gradient. It is now removed (details below). Under the amended gate, L2 accepted counts are evograd 4/4, codex 3/4 (its qkv_norm_rope is numerically wrong), gepa 2/4 (its residual fails Gate 3 with a late loss spike) and seed 3/4.
- **L3 (amended).** evograd {residual_rmsnorm}: 1.055x [1.034, 1.081] vs eager. codex {residual_rmsnorm}: 1.035x [1.029, 1.043]. gepa {swiglu_mlp}: 0.985x [0.979, 0.989]. All three are accepted, but every combination runs at only 0.57–0.60x of whole-model `torch.compile`.
- **Caveat.** The training step is host-bound (`cpu_bound_fraction` ≈ 0.996), and single-site T3 of the same program moves by ±5–8% between nodes. That is the size of the effects, so which candidates count as positive is fragile in a single-trial run.

## Setup (frozen)

| Item | Setting |
|---|---|
| Model / data | Qwen3-0.6B `c1899de2…`, WikiText-2 raw TRAIN `b08601e0…`; BF16, batch 2, seq 2048, SDPA, train mode |
| Cases | L1 `rmsnorm/rmsnorm`, `swiglu/mlp_activation`; L2 `fused_add_rms_norm/residual_rmsnorm`, `qwen3_qkv_norm_rope/qkv_norm_rope`, `qwen3_attention/causal_gqa_attention`, `qwen3_swiglu_mlp/swiglu_mlp`; separate pure-SDPA reference `causal_gqa_attention/causal_gqa_sdpa` |
| Agents | seed (no LLM), evograd (unified frontend + OpenEvolve), Codex CLI 0.149.1, GEPA 8f287de; one trial each |
| LLM / budget | `gpt-5.5` through the metering proxy, 10 admitted requests per agent × case (retries counted) |
| Code | commit 6ff215e run from a frozen export, plus the three patches below (manifests and diffs kept with the results) |
| Stack | torch 2.11.0+cu128, triton 3.6.0, transformers 5.16.1, openevolve 0.3.2, litellm 1.106.2, openai 2.54.0; NVIDIA GH200 120GB (DeltaAI) |
| Corpus | `evograd_payload_corpus/20261021_main`, manifest sha256 `8fd19a00…`; fidelity 2252 payloads, layouts ok, outputs bitwise |

**Deviations from the plan's command templates.** None of these changes a gate threshold, the timer, the request budget or the optimizer.

1. Allocations were 2 h and later 1 h on `ghx4-interactive`, re-requested in a loop, because 4 h `ghx4` jobs queued for about an hour. The time limits were therefore set to one common policy before any formal search: outer limit 6000 s (the plan has 10800 s) and Codex session 4800 s (the plan has 9600 s; the 1200 s margin is kept). The evaluator timeout stays at 900 s. No search came near these limits: evograd took 12–14 min per case, Codex 1–2 min, GEPA 9–11 min.
2. Every `evaluate` runs with `PYTHONHASHSEED=0`. Candidate module names are hash-based, so without a fixed seed a resumed Gate 3 discarded its checkpoint. A cross-node resume was later confirmed bitwise identical to an uninterrupted run.
3. Result and corpus paths are not exported to agent environments. Each search gets an empty node-local Triton cache. The task pack's search-time baseline cache is cleared before each Codex or GEPA search.
4. Formal evaluations started only after every Codex search had finished, because Codex has a shell.

## Code changes made during the run (included in this commit)

1. **Import check per submission** (`evaluation/protocol.py`). One submission that failed to import used to crash the agent's whole evaluation. Now each submission is imported in its own process first. A failure is recorded as submitted, not executable and not accepted, with an `import_error` field. `configure()` is re-applied after the imports so that module-level torch flag changes cannot alter verification settings. Nothing triggered this check in this run.
2. **Plain values in a pair's saved state** (`pipelines/shared/artifact.py`). The task statement allows `saved_tensors` to hold tensors and immutable Python scalars, but the generated deployment layer passed everything to `ctx.save_for_backward`, which raises TypeError on an int. Non-tensor values are now kept on `ctx` with their positions and restored in order for backward. Behaviour is unchanged for tensor-only state. Two GEPA submissions (swiglu_mlp, qkv_norm_rope) saved shape integers. They were repackaged with the fixed layer, which changes only the harness layer and not the agent's block. Their frozen-state evaluation is archived.
3. **Model-scope Gate 2 without per-call boundary replay** (`verification/integrations/gates.py`, `GATE2_MODEL_BOUNDARIES = False`). See the gate review below. Model Gate 2 now judges the whole model's outputs and parameter gradients, plus coverage and input/parameter side effects. A cached policy calibrated with the other setting is recalibrated, and the setting is bound into the stage key. The verification suite passes (355 tests) plus 3 new tests.

## Results under the amended model Gate 2 (current protocol)

| Case | Agent | Verdict | G1 / op-G2 / model-G2 / G3 | T2 vs eager [95% CI] | T2 vs compile | Single-site T3 vs eager [95% CI] | Positive |
|---|---|---|---|---|---|---|---|
| rmsnorm (L1) | seed | accepted | pass / pass / n/a / n/a | 0.34 [0.32, 0.35] | 0.27 | n/a | n/a |
| rmsnorm (L1) | evograd | accepted | pass / pass / n/a / n/a | 1.82 [1.80, 1.86] | 1.42 | n/a | n/a |
| rmsnorm (L1) | codex | accepted | pass / pass / n/a / n/a | 1.87 [1.72, 1.93] | 1.36 | n/a | n/a |
| rmsnorm (L1) | gepa | accepted | pass / pass / n/a / n/a | 1.73 [1.73, 1.81] | 1.35 | n/a | n/a |
| mlp_activation (L1) | seed | accepted | pass / pass / n/a / n/a | 0.34 [0.33, 0.34] | 0.34 | n/a | n/a |
| mlp_activation (L1) | evograd | accepted | pass / pass / n/a / n/a | 1.56 [1.37, 1.69] | 1.72 | n/a | n/a |
| mlp_activation (L1) | codex | accepted | pass / pass / n/a / n/a | 1.66 [1.44, 1.84] | 1.63 | n/a | n/a |
| mlp_activation (L1) | gepa | rejected (G1, op-G2) | FAIL / FAIL / n/a / n/a | 1.44 [1.26, 1.57]* | 1.60* | n/a | n/a |
| residual_rmsnorm (L2) | seed | accepted | pass / pass / pass / pass | 0.35 [0.33, 0.35] | 0.28 | 0.70 [0.70, 0.70] | no |
| residual_rmsnorm (L2) | evograd | accepted | pass / pass / pass / pass | 1.70 [1.64, 1.84] | 1.34 | 1.03 [1.01, 1.08] | yes |
| residual_rmsnorm (L2) | codex | accepted | pass / pass / pass / pass | 1.95 [1.80, 1.99] | 1.52 | 1.01 [1.01, 1.03] | yes |
| residual_rmsnorm (L2) | gepa | rejected (G3) | pass / pass / pass / FAIL | 1.78 [1.76, 1.81]* | 1.34* | 1.00 [1.00, 1.01]* | no |
| qkv_norm_rope (L2) | seed | accepted | pass / pass / pass / pass | 0.38 [0.38, 0.38] | 0.22 | 0.54 [0.54, 0.54] | no |
| qkv_norm_rope (L2) | evograd | accepted | pass / pass / pass / pass | 0.49 [0.48, 0.51] | 0.29 | 0.63 [0.62, 0.66] | no |
| qkv_norm_rope (L2) | codex | rejected (G1, op-G2, model-G2) | FAIL / FAIL / FAIL / skip | 1.83 [1.77, 1.84]* | 1.08* | 1.13 [1.13, 1.14]* | no |
| qkv_norm_rope (L2) | gepa | rejected (G1, op-G2, model-G2) | FAIL / FAIL / FAIL / skip | 2.56 [2.48, 2.59]* | 1.47* | 1.30 [1.23, 1.32]* | no |
| causal_gqa_attention (L2) | seed | accepted | pass / pass / pass / pass | 0.59 [0.58, 0.59] | 0.74 | 0.92 [0.92, 0.93] | no |
| causal_gqa_attention (L2) | evograd | accepted | pass / pass / pass / pass | 0.85 [0.84, 0.87] | 1.05 | 0.96 [0.94, 1.00] | no |
| causal_gqa_attention (L2) | codex | accepted | pass / pass / pass / pass | 0.99 [0.98, 1.00] | 1.22 | 0.99 [0.99, 1.00] | no |
| causal_gqa_attention (L2) | gepa | accepted | pass / pass / pass / pass | 1.03 [1.02, 1.04] | 1.26 | 0.95 [0.95, 0.96] | no |
| swiglu_mlp (L2) | seed | rejected (G3) | pass / pass / pass / FAIL | 0.37 [0.37, 0.38]* | 0.43* | 0.71 [0.71, 0.71]* | no |
| swiglu_mlp (L2) | evograd | accepted | pass / pass / pass / pass | 0.82 [0.81, 0.83] | 0.92 | 0.96 [0.95, 1.00] | no |
| swiglu_mlp (L2) | codex | accepted | pass / pass / pass / pass | 0.93 [0.92, 0.96] | 1.08 | 0.99 [0.98, 0.99] | no |
| swiglu_mlp (L2) | gepa | accepted | pass / pass / pass / pass | 1.12 [1.10, 1.12] | 1.25 | 1.04 [1.04, 1.05] | yes |

\* diagnostic timing of a candidate that is not accepted (not part of any aggregate).

| Agent | Level | Submitted | Executable | Accepted | Coverage (official denominator) | Pass rate (selected cases) | T2 geomean vs eager | vs compile |
|---|---|---|---|---|---|---|---|---|
| seed | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 0.34 | 0.30 |
| seed | L2 | 4/4 | 4 | 3 | 3/4 = 0.750 | 3/4 | 0.41 | 0.34 |
| evograd | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 1.68 | 1.56 |
| evograd | L2 | 4/4 | 4 | 4 | 4/4 = 1.000 | 4/4 | 0.97 | 0.88 |
| codex | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 1.76 | 1.49 |
| codex | L2 | 4/4 | 4 | 3 | 3/4 = 0.750 | 3/4 | 1.22 | 1.26 |
| gepa | L1 | 2/14 | 2 | 1 | 1/14 = 0.071 | 1/2 | 1.73 | 1.35 |
| gepa | L2 | 4/4 | 4 | 2 | 2/4 = 0.500 | 2/4 | 1.07 | 1.26 |

| Agent | L3 combination | model-G2 | G3 | Combined T3 vs eager [95% CI] | vs whole-model compile | Peak memory vs eager |
|---|---|---|---|---|---|---|
| seed | empty (no positive L2 candidate) | - | - | - | - | - |
| evograd | residual_rmsnorm (accepted) | pass | pass | 1.055 [1.03, 1.08] | 0.59 | 0.95 |
| codex | residual_rmsnorm (accepted) | pass | pass | 1.035 [1.03, 1.04] | 0.60 | 0.95 |
| gepa | swiglu_mlp (accepted) | pass | pass | 0.985 [0.98, 0.99] | 0.57 | 0.95 |

## Results as originally run (model Gate 2 with per-call boundary replay)

These are the formal results under the frozen code, after the GEPA deployment-layer fix (change 2). The L2 verdicts differ from the amended run only through model Gate 2. The timings are separate measurements on other nodes.

| Case | Agent | Verdict | G1 / op-G2 / model-G2 / G3 | T2 vs eager [95% CI] | T2 vs compile | Single-site T3 vs eager [95% CI] | Positive |
|---|---|---|---|---|---|---|---|
| rmsnorm (L1) | seed | accepted | pass / pass / n/a / n/a | 0.34 [0.34, 0.34] | 0.27 | n/a | n/a |
| rmsnorm (L1) | evograd | accepted | pass / pass / n/a / n/a | 1.79 [1.72, 1.86] | 1.41 | n/a | n/a |
| rmsnorm (L1) | codex | accepted | pass / pass / n/a / n/a | 1.76 [1.67, 1.86] | 1.33 | n/a | n/a |
| rmsnorm (L1) | gepa | accepted | pass / pass / n/a / n/a | 1.79 [1.72, 1.83] | 1.37 | n/a | n/a |
| mlp_activation (L1) | seed | accepted | pass / pass / n/a / n/a | 0.34 [0.32, 0.35] | 0.35 | n/a | n/a |
| mlp_activation (L1) | evograd | accepted | pass / pass / n/a / n/a | 1.69 [1.50, 1.81] | 1.79 | n/a | n/a |
| mlp_activation (L1) | codex | accepted | pass / pass / n/a / n/a | 1.36 [1.26, 1.48] | 1.54 | n/a | n/a |
| mlp_activation (L1) | gepa | rejected (G1, op-G2) | FAIL / FAIL / n/a / n/a | 1.56 [1.38, 1.69]* | 1.51* | n/a | n/a |
| residual_rmsnorm (L2) | seed | accepted | pass / pass / pass / pass | 0.35 [0.34, 0.35] | 0.28 | 0.71 [0.70, 0.71] | no |
| residual_rmsnorm (L2) | evograd | rejected (model-G2) | pass / pass / FAIL / skip | 1.72 [1.66, 1.74]* | 1.35* | 0.95 [0.93, 0.97]* | no |
| residual_rmsnorm (L2) | codex | rejected (model-G2) | pass / pass / FAIL / skip | 2.00 [1.82, 2.11]* | 1.47* | 1.09 [1.06, 1.11]* | no |
| residual_rmsnorm (L2) | gepa | rejected (model-G2) | pass / pass / FAIL / skip | 1.71 [1.60, 1.75]* | 1.35* | 1.10 [1.06, 1.14]* | no |
| qkv_norm_rope (L2) | seed | accepted | pass / pass / pass / pass | 0.38 [0.37, 0.38] | 0.22 | 0.54 [0.53, 0.54] | no |
| qkv_norm_rope (L2) | evograd | rejected (model-G2) | pass / pass / FAIL / skip | 0.50 [0.49, 0.50]* | 0.29* | 0.62 [0.62, 0.63]* | no |
| qkv_norm_rope (L2) | codex | rejected (G1, op-G2, model-G2) | FAIL / FAIL / FAIL / skip | 1.94 [1.91, 1.97]* | 1.06* | 1.26 [1.23, 1.29]* | no |
| qkv_norm_rope (L2) | gepa | rejected (G1, op-G2, model-G2) | FAIL / FAIL / FAIL / skip | 2.51 [2.48, 4.07]* | 1.47* | 1.47 [1.43, 1.50]* | no |
| causal_gqa_attention (L2) | seed | accepted | pass / pass / pass / pass | 0.58 [0.58, 0.58] | 0.73 | 0.94 [0.92, 0.95] | no |
| causal_gqa_attention (L2) | evograd | accepted | pass / pass / pass / pass | 0.84 [0.83, 0.84] | 1.06 | 0.92 [0.91, 0.93] | no |
| causal_gqa_attention (L2) | codex | accepted | pass / pass / pass / pass | 0.98 [0.97, 0.99] | 1.17 | 1.00 [0.98, 1.03] | no |
| causal_gqa_attention (L2) | gepa | accepted | pass / pass / pass / pass | 1.00 [1.00, 1.01] | 1.25 | 1.03 [0.98, 1.06] | no |
| swiglu_mlp (L2) | seed | rejected (G3) | pass / pass / pass / FAIL | 0.36 [0.36, 0.37]* | 0.42* | 0.73 [0.73, 0.74]* | no |
| swiglu_mlp (L2) | evograd | rejected (model-G2) | pass / pass / FAIL / skip | 0.81 [0.81, 0.82]* | 0.94* | 0.94 [0.93, 0.95]* | no |
| swiglu_mlp (L2) | codex | rejected (model-G2) | pass / pass / FAIL / skip | 0.93 [0.92, 0.94]* | 1.06* | 1.03 [1.02, 1.05]* | no |
| swiglu_mlp (L2) | gepa | accepted | pass / pass / pass / pass | 1.15 [1.13, 1.86] | 1.28 | 1.06 [1.03, 1.09] | yes |

\* diagnostic timing of a candidate that is not accepted (not part of any aggregate).

| Agent | Level | Submitted | Executable | Accepted | Coverage (official denominator) | Pass rate (selected cases) | T2 geomean vs eager | vs compile |
|---|---|---|---|---|---|---|---|---|
| seed | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 0.34 | 0.31 |
| seed | L2 | 4/4 | 4 | 3 | 3/4 = 0.750 | 3/4 | 0.40 | 0.33 |
| evograd | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 1.74 | 1.59 |
| evograd | L2 | 4/4 | 4 | 1 | 1/4 = 0.250 | 1/4 | 0.84 | 1.06 |
| codex | L1 | 2/14 | 2 | 2 | 2/14 = 0.143 | 2/2 | 1.55 | 1.43 |
| codex | L2 | 4/4 | 4 | 1 | 1/4 = 0.250 | 1/4 | 0.98 | 1.17 |
| gepa | L1 | 2/14 | 2 | 1 | 1/14 = 0.071 | 1/2 | 1.79 | 1.37 |
| gepa | L2 | 4/4 | 4 | 2 | 2/4 = 0.500 | 2/4 | 1.07 | 1.27 |

| Agent | L3 combination | model-G2 | G3 | Combined T3 vs eager [95% CI] | vs whole-model compile | Peak memory vs eager |
|---|---|---|---|---|---|---|
| seed | empty (no positive L2 candidate) | - | - | - | - | - |
| evograd | empty (no positive L2 candidate) | - | - | - | - | - |
| codex | empty (no positive L2 candidate) | - | - | - | - | - |
| gepa | swiglu_mlp (accepted) | pass | pass | 0.979 [0.92, 1.03] | 0.55 | 0.95 |

## Gate review: why model Gate 2 was too strict

- **Where the failures were.** Every model-Gate-2 rejection of a rounding-level candidate was in the **per-call boundary replay**. That check replays each replaced call of the site on its recorded inputs with native and candidate code, comparing local outputs and gradients (plus `~live` and `~repeat`). Its budgets are keyed by call index (`residual_rmsnorm#27/...`): 4 × the compile control's drift *at that call*, over four samples. The whole-model outputs and parameter gradients of the same candidates all passed.
- **How large the failures were.** 2–6 results out of about 700–1150 per sample, in 1–3 of 4 samples. All were weight gradients of one layer, with relative L2 0.1–0.6%. Every `diff_max` is an exact power of two, which means a single bf16 ULP flip on an outlier element. The `~live` failures repeat the same numbers, and `~repeat` never failed.
- **Duplication with operator Gate 2.** Operator Gate 2 already checks the same quantity on the same calls and data orders, pooling one budget over every call of the site. For evograd's residual_rmsnorm norm-weight gradient, the operator budget is Emax 0.443 and the candidate's worst value is 0.213, a pass. The per-call boundary budgets were 0.058–0.080, so the same candidate failed at 0.08–0.195.
- **What else the boundary checks added.** Coverage, input mutation and parameter modification are checked outside the boundary replay, and they remain in place.
- **Offline re-judgement from the stored metrics.** Re-judging the stored metrics with whole-model results only rejects every true failure. Codex's and GEPA's numerically wrong qkv_norm_rope exceed their budgets by 723–800x on about 1180 results. The x1.01 fault candidates exceed theirs by 2.73x (residual), 9200x (attention), 1.65x (qkv_norm_rope) and 2.74x (swiglu_mlp). Every rounding-level candidate passes, using at most 76% of its budget.
- **Confirmation on GPU.** The same check was repeated under the amended gate: all 7 x1.01 faults were rejected (operator Gate 2 7/7, model Gate 2 4/4), with the ratios predicted offline. The seed (positive control) gives the same verdicts as before.
- **Gate 3 is not too strict.** Every attention candidate reproduces native's loss trajectory bitwise. The other passing ratios are 0.02–0.78. Both Gate 3 failures are real:
  - The seed's swiglu_mlp drifts from step 2, with a loss spike of +1.42 at step 423.
  - GEPA's residual_rmsnorm tracks native for 800 steps, then spikes by +3.99 at step 902 and ends 0.22 nats higher.

  Each 1000-step run takes about 3 minutes.

## Costs

| Agent | Requests per case | Total admitted | Refused (402) | Evaluator calls | Search wall time | Termination |
|---|---|---|---|---|---|---|
| seed | 0/0/0/0/0/0 | 0 | 0 | 6 | 1.9 min | completed: frontend only (no LLM) |
| evograd | 10/10/10/10/10/10 | 60 | 0 | 72 | 76.0 min | completed: OpenEvolve reached 10 iterations |
| codex | 10/10/10/10/10/10 | 60 | 6 | 22 | 9.4 min | budget_exhausted: request 11 refused (402), session ended |
| gepa | 6/6/6/6/6/6 | 36 | 0 | 36 | 60.3 min | completed: native max_evals=11 reached |

Requests are the proxy's `api_calls.jsonl` lines, retries included. Evaluator calls differ in composition by agent:
- evograd counts the seed acceptance, the initial program and 10 iterations;
- Codex counts correctness-only and full evaluations;
- GEPA has one evaluation per candidate.

GEPA stops after 6 requests because its native `max_evals = budget + 1` also counts cached parent re-evaluations; this was left as configured. Token usage is unknown for all agents:
- Codex emits no usage after its 402;
- OpenEvolve records none;
- GEPA has only a litellm cost estimate, about $7.7 for six cases.

## Pure-SDPA reference (separate root, not in the core aggregates)

The vendor-only frontend seed of `causal_gqa_attention/causal_gqa_sdpa` is accepted (Gate 1 and operator Gate 2). T2 is 1.00x vs eager and 1.21x vs `torch.compile`.

## Observations

- **Single-site L2 replacement gives little whole-model benefit in this configuration.**
  - The training step is host-bound.
  - Each site is a fraction of the step: residual_rmsnorm about 19% across 56 calls, qkv_norm_rope about 44%.
  - Whole-model `torch.compile` is about 1.7x faster than eager.
  - Example: evograd's residual_rmsnorm is 1.70x at T2, which predicts an ideal T3 of about 1.08x; it measures 1.03x.
- **Why evograd is weak on the GEMM/SDPA cases (qkv_norm_rope 0.49x, swiglu_mlp 0.82x, attention 0.85x at T2).**
  - The unified frontend takes only the AOTAutograd cut and Inductor's extern decisions from the compiler. Every remaining ATen node is lowered to its own handwritten Triton kernel, so the seed launches 9–16 kernels where Inductor launches 2–9, and starts at 0.34–0.58x eager.
  - The unified contract also fixes the preserved GEMM/SDPA call sites (for example 9 `mm` wrappers in qkv_norm_rope, each called exactly once), so evolution can only fuse the glue between them.
  - On the elementwise cases, 10 iterations close the gap (1.6–1.8x). On the GEMM cases they do not, while Codex and GEPA restructure freely around `vendor_gemm`/`vendor_sdpa`.
  - A fused seed (the stored `compiler_prior`), or a relaxed extern-site contract, would be the next things to try. This is not tested here.
- **Timing noise.** Re-measuring the same program on another node changed:
  - evograd residual_rmsnorm: 0.947 → 1.033;
  - Codex swiglu_mlp: 1.033 → 0.989;
  - GEPA attention: 1.065 → 1.033 → 0.950.

  The protocol's intervals cover only within-run timing noise. Repeated T3 measurements, or measuring all providers in one allocation, would make positive selection more robust.

## Where the results are

On DeltaAI, under `/u/wzhan/.cache/adbench/`:

| Path | Contents |
|---|---|
| `20261025_v2_b10_trial01/qwen3_0_6b/` | Formal root: tasks, runs (ledgers, submissions), evaluation, summary, Chinese report |
| `20261025_v2_b10_trial01_g2amend/qwen3_0_6b/` | Amended evaluation of the same frozen submissions |
| `20261025_v2_sdpa_reference/`, `20261025_v2_smoke/` | Pure-SDPA reference root and smoke root |
| `20261025/` | Run record: freeze manifests and patches, audit JSONs, logs, chains/scripts, `STATUS.md`, `records/` (GEPA frozen-state evaluation, completeness checks, gate analyses) |

Large evaluation and state directories are on `/work/nvme/bcjw/wzhan/adbench_20261025/`.
