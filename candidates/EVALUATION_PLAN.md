# ADBench Qwen3-0.6B Evaluation Plan

Date: 2026-10-25. Protocol: `evograd-evaluation-v2`.

This experiment compares evograd, Codex, and GEPA on generating forward/backward implementations from the same task inputs, using `gpt-5.5` with at most 10 model requests per case. The main comparison covers six core cases: two L1 and four L2. A frontend-only seed provides a zero-LLM reference. Each agent independently selects L2 candidates with a demonstrated whole-model benefit, combines them into its own L3 deployment, and verifies and times that combination.

The experiment asks three questions: does the candidate pass correctness verification under the frozen protocol; is the deployed operator faster; and does that benefit carry into whole-model training? Report these separately, without an overall score. Pure SDPA is a separate reference and does not enter the core search tasks or aggregates.

## Scope and frozen configuration

| Item | Setting |
| --- | --- |
| Evaluated model | Qwen3-0.6B, model name `qwen3_0_6b` |
| Weights | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Data | WikiText-2 raw TRAIN, revision `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Model configuration | BF16, batch 2, sequence 2048, train mode, SDPA |
| Agents | `evograd`, `codex`, `gepa`, in this execution order |
| Reference | `seed`: evograd frontend only, without an LLM |
| LLM | `ADBENCH_MODEL=gpt-5.5`; do not change the model default in source code |
| Request budget | `ADBENCH_BUDGET=10`; an explicit `--budget` overrides the environment, and formal runs fix it at 10 |
| Trials | One search trial per agent and case in the first round, named `trial01` |
| Main comparison size | 3 agents x 6 cases = 18 searches, at most 180 upstream model requests in total; seed uses 0 |
| Protocol implementation | `src/evograd/evaluation/protocol.py`; results must use v2 |

This is not a coverage experiment over every observed Qwen configuration. Other L1 shapes, standalone GEMM, RoPE, CE, and other operators are outside the six search tasks. Raw OpenEvolve and KernelAgent can be added later under the same protocol. Claude Code and GEAK use different model and budget semantics and must be reported separately. A second model, additional search trials, and frontend ablations are outside this experiment.

### Core cases

| Level | case_id | Fixed configuration | Allowed vendor primitives |
| --- | --- | --- | --- |
| L1 | `rmsnorm/rmsnorm` | rows 4096, hidden 1024 | None |
| L1 | `swiglu/mlp_activation` | rows 4096, cols 3072 | None |
| L2 | `fused_add_rms_norm/residual_rmsnorm` | rows 4096, cols 1024 | None |
| L2 | `qwen3_qkv_norm_rope/qkv_norm_rope` | B 2, T 2048, H 1024, HQ 16, HK 8, D 128 | GEMM |
| L2 | `qwen3_attention/causal_gqa_attention` | B 2, T 2048, H 1024, HQ 16, HK 8, D 128 | GEMM, SDPA |
| L2 | `qwen3_swiglu_mlp/swiglu_mlp` | B 2, T 2048, H 1024, I 3072 | GEMM |

The separate reference is L1 `causal_gqa_attention/causal_gqa_sdpa`, with B 2, T 2048, HQ 16, HK 8, D 128 and vendor SDPA allowed. Generate only its frontend seed, then run Gate 1, operator Gate 2, and formal T2. Use a separate results root and label it as a vendor-only reference; do not mix it with the six core cases.

### Input fairness and submissions

Build one frozen task pack per case. All three systems receive the same forward reference, shapes/dtypes/scalars, required gradients, forward/backward ABI, allowed primitives, static rules, and shared evaluator. Record hashes for `TASK.md`, `task.json`, `declaration.py`, and every other task file. Acceptance payloads and final Gate/T3 results are not search feedback.

The shared information is the external task specification. evograd generates its own executable seed from those inputs; Codex starts from the skeleton; GEPA generates its initial candidate by default. Each system retains its own internal prompts and search procedure. Individual LLM prompts need not be identical, and other systems do not receive evograd's seed as extra input.

Each agent submits one final program per case, using the driver's final selection and implemented fallback rules. Budget exhaustion preserves an available candidate even if it has not passed correctness checks. `bench.py` repackages the evolve block into `submission.py`, recording its hash and `fixed_code_intact`. Freeze the submitted code; a formal verification failure must not be fed back to the agent for repairs within that trial.

### Request budget and randomness

Each model request admitted for attempted forwarding consumes one request, including initial generation, reflection, repairs, retries after upstream failures, and context compaction. Concurrent requests share the cap. Request 11 is not forwarded, and failed requests are not refunded. The admitted tenth request must be allowed to return normally rather than being killed early. A search may stop before using all 10 requests.

`api_calls.jsonl` is the request ledger. `evaluations.jsonl` records evaluator calls separately. `llm.jsonl` contains usage events exposed by the agent; its line count is not an API request count. Report tokens, wall time, and evaluator calls separately. Missing usage measurements are unknown, not zero. Equal request caps do not imply equal token consumption. Do not set an additional `--eval-budget` for Codex in this experiment. Keep each system's native iteration/evaluation stopping conditions without granting extra requests.

The first round has only one search trial and does not estimate variance across searches. The common `--seed` is not a shared LLM sampling seed: GEPA uses an engine seed, evograd retains OpenEvolve's native configuration, and Codex has no corresponding sampling-seed interface. Save the generated configurations and CLI versions. Verification and timing seeds are frozen separately below.

## Correctness and performance matrix

Level describes task granularity, Gate describes correctness evidence, and Tier describes the scope of timing.

| Object | Requirements for accepted | Formal performance results |
| --- | --- | --- |
| L1 candidate | Static rules pass; Gate 1 and operator Gate 2 pass | T2 against eager and operator compile |
| L2 candidate | Static rules pass; Gate 1, operator Gate 2, single-site model Gate 2, and single-site Gate 3 all pass | T2; single-site T3 against whole-model eager |
| L3 combination | Model Gate 2 and Gate 3 pass on the combined deployment | Combined T3 against eager and whole-model compile |

T1 serves search feedback and standalone kernel/pair diagnostics. Final deployment performance uses T2/T3; search-time `combined_score` and the best search-time measurement are not final scores. `score/compare`, standalone `tier2-bench`, and `tier3-bench` do not replace formal `bench.py evaluate`.

Distinguish `passed`, `failed`, `incomplete`, and `skipped`. A candidate cannot be accepted unless every required Gate passes. An invalid or unavailable control/policy leaves verification incomplete; do not relax thresholds. A static-rule violation prevents verification and timing. Other executable candidates that are not accepted may still receive diagnostic timings, with their verdicts shown alongside the numbers. They do not enter accepted performance aggregates or L3.

### Gate 1 and Gate 2

Gate 1 uses real acceptance payloads to check forward/backward structure, shapes, dtypes, finiteness, input mutation, outputs, and required gradients. Backward uses the candidate's own saved forward state. The elementwise rule is `abs(c-r) <= max(atol + rtol*abs(r), envelope)`. Tolerances come from the frozen table for each result's dtype; the envelope comes from a trusted compile control, never from candidate error.

| Check | Data and frozen parameters |
| --- | --- |
| Gate 1 | Real acceptance calls from orders 303/404; no additional generated-input Gate 1 |
| Operator Gate 2 | Calibration orders 101/202, acceptance orders 303/404; actual autograd deployment; gamma=2 |
| Model Gate 2 | One L2 candidate installed at its model site; the first TRAIN batch of each order 101/202/303/404; one whole-model forward/backward per sample, without an optimizer update; gamma=4 |
| Model error budget | Pool maxima per result tensor over the four samples, then verify fresh executions of the same sample contents; this is a matched-input check, not an independent holdout |
| Coverage and side effects | Every declared site call is actually replaced; inputs and parameters are not modified. No per-call boundary replay at model scope (amended 2026-10-25): its local check duplicated operator Gate 2 on the same calls with per-call budgets and rejected rounding-level candidates on single bf16 ULP flips |

Gate 2 checks both relative L2 error E2 and normalized maximum error Emax. Budgets are calibrated from native-repeat error, compile-native error, and fixed floors. Model Gate 2 judges the whole model's outputs and parameter gradients; the local checks of each call stay with operator Gate 2, which pools its budgets over every call of the site. Gate 3's scalar loss cannot replace output and gradient checks.

### Gate 3

Run single-site Gate 3 for an L2 candidate only after its preceding three Gates pass; otherwise mark it skipped. Run Gate 3 again after deploying the L3 combination. Passing the individual candidates does not establish that their combination passes.

| Parameter | Qwen3-0.6B setting |
| --- | --- |
| Updates | **1000 AdamW updates**, for both single-site and combined deployments |
| Optimizer | lr `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.01`, `foreach=False` |
| Training | Constant learning rate; no warmup or gradient clipping; accumulation 1 |
| Randomness and data | RNG seed 0, TRAIN data order 707; consecutive batches, cycling through the same permutation when exhausted |
| Objective | Explicit FP32 next-token CE on logits, sum / valid-token count; record the loss actually differentiated before each update |
| Controls | Native A, native B, and a compile control for the current patch set, with identical initial weights, optimizer, and data sequence |
| Acceptance | Both RMS and maximum absolute difference over the full loss trajectory must stay within their budgets; gamma=2, both floors `1e-4` |
| Checkpointing | Save every 100 steps and when the deadline is reached; the horizon remains 1000 updates |

This compile control compiles the corresponding replacement sites; it is not the whole-model compile baseline in the L3 performance table. Passing establishes acceptance under the frozen inputs, trajectory, and tolerances, not mathematical equivalence or correctness for arbitrary training. Keep these parameters fixed. A seed or agent failure does not justify changing gamma, learning rate, clipping, or the horizon for this experiment.

## Performance measurement and combination

### T2

Time the candidate's actual deployment forward followed by `torch.autograd.grad` backward, without an optimizer or `.grad` accumulation. Eager, operator compile, and available declared library baselines use the same autograd scope. Every provider receives identical seeded inputs at the observed case's shapes, dtypes, and scalars. Real-payload Gates independently determine numerical acceptance.

Freeze the fair timer at warmup 10, reps 50, 3 blocks, and seed 0, retaining its adaptive reduction of repetitions for expensive calls. Formal results currently retain full forward+backward candidate/baseline latency, speedup, and a 95% interval, plus forward/backward speedups. They do not retain separate forward/backward absolute latencies, the actual adaptive repetition counts, or raw T2 blocks. Use eager/compile in the main table and other library comparisons in supplementary tables. An unavailable library must not discard available main-baseline results.

### Single-site and combined T3

Single-site T3 installs one L2 candidate at every applicable invocation of that site, leaving other sites native. "Single-site" does not mean a single decoder layer. T3 measures the complete training step: forward/loss, backward, AdamW update, and gradient reset. Each provider runs in its own process.

Freeze warmup 3, 10 steps per block, 10 blocks, `loss_steps=5`, lr `1e-4`, and seed 0. Untimed loss, settling, and memory-probe steps precede timing. Timed steps repeat the first batch of order 0, and model state continues updating between steps. Synchronize each block, compute its mean step wall time, and report the median across blocks. Retain raw block data, 95% intervals, throughput, peak memory, and `cpu_bound_fraction`. Do not flush L2 cache inside a training step.

Gate 3's 1000-update loss verification and T3's short performance measurement are separate experiments. Each freezes its own data sequence, loss entry point, and optimizer execution choices. T3 uses the HF training loss; its five loss records do not replace Gate 3. Verification enables deterministic algorithms and timing disables them. Attention follows the resulting flash/cuDNN selection; do not change the protocol to force a common backend.

### Positive candidates and L3

```text
positive(case) = accepted(case)
                 AND single_site_T3.ok
                 AND lower_95pct_CI(speedup_vs_eager) > 1
```

For each agent, deploy all of its positive L2 candidates together and rerun model Gate 2, Gate 3, and combined T3. L1 artifacts are not independently installed in the whole model, and candidates from different agents are never mixed. Local T2 speedup is not a selection criterion. Single-site T3 gains must not be added to predict the combination's gain.

The combination's formal metrics are speedup against eager and whole-model compile, a 95% interval against eager, peak-memory ratio, and CPU submission/blocking fraction. Compile-relative speedup currently has only a point estimate; do not reuse the eager-relative interval. Timings for a combination that is not accepted are diagnostic only.

When no L2 candidate is positive, record L3 as empty with a reason. Do not force candidates into the combination, assign an empty combination a 1x speedup, or treat emptiness as a pipeline failure. Retain combination verification failures or performance regressions. Do not search subsets or request additional LLM calls based on the final results.

## Execution sequence

### Freeze code and inputs

Before formal runs, record the source revision, dependency versions, GPU/driver information, model and data revisions, and corpus manifest/hash. Pin a revision containing the v2 implementation; baseline `0512b69` alone does not include it. If running a dirty working tree, preserve the full diff and snapshots of untracked files so that the exact source can be reconstructed.

Generate task packs in a new root and verify the six cases and all file hashes. Existing packs receive only limited checks from `tasks`, and model packs contain absolute declaration paths. Copying an old task directory is not sufficient to establish a frozen, valid input set. Keep code and task packs unchanged throughout formal runs.

The existing Qwen corpus can be reused after confirming that its capture identity matches the current reference. Run fidelity or reuse an exactly matching fidelity report. Every agent uses the same corpus. Model or data updates invalidate earlier acceptance conclusions.

### Smoke validation

Use a separate smoke root, for example `/u/wzhan/.cache/adbench/20261025_v2_smoke`. Run `rmsnorm/rmsnorm` for each of the three agents, still with at most 10 model requests each. The smoke cap is 30 requests, accounted for separately from the main comparison's 180. Do not transfer smoke candidates into the formal trial.

Check model routing, actual request counts, budget-exhaustion recovery, and submission packaging. Run formal evaluation whenever a submission exists. Also use one L2 frontend seed to exercise model Gate 2, 1000-update Gate 3, checkpoint resume, and single-site T3 on a real GPU. Correctness failure can be a valid research result; infrastructure, routing, or accounting failures are pipeline problems. L3 may be empty when nothing is positive, but this does not validate the nonempty combination path on a GPU.

Complete smoke-driven configuration or infrastructure fixes before formal trials. CPU regressions and Codex tests against a local fake API do not replace a real GPT-5.5/GPU smoke run.

### Formal search and evaluation

1. Generate seed references for all six core cases. Old v1 seed results cannot enter the new main table; reevaluate them under v2.
2. Run the six cases for evograd, Codex, and GEPA in that order. Count each case's budget independently. Schedule search by agent and case so that a job wall-time limit does not interrupt later cases.
3. Collect each case's terminal status and submission hash. Freeze the code, then evaluate the agent's complete six-case set. Evaluation does not consume its 10-request search budget.
4. Resume a suspended evaluation with the same command until all applicable Gates, T2, single-site T3, combination selection, and combination evaluation finish. Then summarize seed and all three agents.
5. Run the pure-SDPA seed reference under a separate root and produce its supplementary table.

All agents share `state` under the formal root. Reuse native/compile calibration only when its bindings match. Keep candidate verification and performance results separate for each agent. A new root does not automatically read an old root's state. To reuse historical controls, make a complete copy into the new state directory and let the implementation validate bindings. Do not force acceptance of old policies or write into a directory used by another active session.

### Resources and execution safeguards

Before submitting a job, run `squeue -u wzhan` and check for work from other sessions. Keep at most one interactive job per user. Smoke uses `ghx4-interactive`, 1 hour, 8 CPUs, and 32g. Longer work uses `ghx4`, 4 hours, 8 CPUs, and 32g. Increase to 64g only after OOM and record the change. Manage long tasks in the `work` window of tmux session `code`. Do not run searches or timings concurrently on the same GPU.

Operational limits are 10800 seconds for the outer bench search, 9600 seconds for the inner Codex session, and 900 seconds per Codex/GEPA evaluator call. These are wall-time safeguards, not extra search budgets. Codex retains 1200 seconds of margin because the driver may wait for an in-flight evaluator after the CLI exits before collecting and packaging its candidate. Start a case only with enough allocation time remaining; the default schedule is one case per 4-hour job. If smoke indicates that these limits need adjustment, record a common policy before formal searches. Do not extend an individual agent's time or calls based on its results.

Run evaluation in separate long jobs. For a 4-hour allocation, begin with `--deadline-min 180` to leave room for the current stage. This takes effect at stage boundaries or Gate 3 checkpoints, not as a hard wall-time limit; monitor long compilation and verification stages. Use a correspondingly shorter deadline in a 1-hour allocation.

Store scripts, task packs, and checkpoints needed by compute nodes under `/u/wzhan/.cache`. Use `/work/nvme/bcjw/wzhan/` for corpora and large data; do not access `/work/hdd`. Do not depend on files in the login node's `/tmp`, and never print credentials in logs.

## Command templates

Run these templates inside the GPU allocations described above. They do not allocate resources themselves. Both `ADBENCH_ROOT` and `--root` name a base directory; bench appends the model name. Use a new root for every new trial, and after changes to the LLM, task packs, or search configuration.

```bash
cd /u/wzhan/evograd-main
export PATH=/u/wzhan/.conda/envs/evograd/bin:/u/wzhan/.local/bin:$PATH
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export EVOGRAD_HF_CACHE=/u/wzhan/.cache/evograd-mF/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/u/wzhan/.cache/adbench/triton-cache
mkdir -p "$TRITON_CACHE_DIR"
export CORPUS=/work/nvme/bcjw/wzhan/evograd_payload_corpus/20261021_main
set +x
set -a
source /u/wzhan/evograd/.env.local
set +a
export ADBENCH_MODEL=gpt-5.5
export ADBENCH_BUDGET=10
export ADBENCH_ROOT=/u/wzhan/.cache/adbench/20261025_v2_b10_trial01
MODEL=qwen3_0_6b
CASES=(
  rmsnorm/rmsnorm
  swiglu/mlp_activation
  fused_add_rms_norm/residual_rmsnorm
  qwen3_qkv_norm_rope/qkv_norm_rope
  qwen3_attention/causal_gqa_attention
  qwen3_swiglu_mlp/swiglu_mlp
)

python candidates/bench.py tasks --model "$MODEL" --case-id "${CASES[@]}"
python -m evograd.verification.integrations "$MODEL" fidelity \
  --corpus "$CORPUS/manifest.json" \
  --out "$ADBENCH_ROOT/$MODEL/fidelity.json"
```

Choose one `AGENT` and `CASE_ID` per search job, repeating across the frozen matrix. Do not start the next case near the end of an allocation:

```bash
AGENT=evograd                    # seed / evograd / codex / gepa
CASE_ID=rmsnorm/rmsnorm          # Choose from CASES
extra=()
case "$AGENT" in
  codex) extra=(--time-limit 9600 --eval-timeout 900) ;;
  gepa)  extra=(--eval-timeout 900) ;;
esac
python candidates/bench.py run --model "$MODEL" --agent "$AGENT" \
  --case-id "$CASE_ID" --budget "$ADBENCH_BUDGET" --time-limit 10800 \
  -- "${extra[@]}"
```

`seed` does not call an LLM. Do not pass `--budget 0`; the shared argument still requires a positive value. Smoke uses a separate root with the same task entry points and request budget, but shorter session limits suitable for a 1-hour allocation. Do not directly run the old six-agent `s1_smoke.sh`: its scope and root differ from this plan.

After freezing submissions, evaluate each agent on the full core case list:

```bash
AGENT=evograd                    # Repeat for seed / evograd / codex / gepa
python candidates/bench.py evaluate --model "$MODEL" --agent "$AGENT" \
  --case-id "${CASES[@]}" --corpus "$CORPUS/manifest.json" \
  --deadline-min 180 --tier3-blocks 10
# Exit 3: rerun this evaluate command unchanged in the next allocation.
```

Confirm that seed and all three agents have completed evaluation of their full case sets, with no `suspended` result, before running the summary separately. Do not continue unconditionally after an evaluation suspends or errors. Retain failure rows for agents with no submissions, as described below.

```bash
python candidates/bench.py summarize --model "$MODEL" \
  --agent seed evograd codex gepa --out "$ADBENCH_ROOT/$MODEL/summary.md"
```

Use another root for the pure-SDPA supplementary table so that cost scans cannot mix it into the core results:

```bash
REF_ROOT=/u/wzhan/.cache/adbench/20261025_v2_sdpa_reference
SDPA_CASE=causal_gqa_attention/causal_gqa_sdpa
python candidates/bench.py tasks --root "$REF_ROOT" --model "$MODEL" --case-id "$SDPA_CASE"
python candidates/bench.py run --root "$REF_ROOT" --model "$MODEL" --agent seed \
  --case-id "$SDPA_CASE" --budget 10 --time-limit 10800
python candidates/bench.py evaluate --root "$REF_ROOT" --model "$MODEL" --agent seed \
  --case-id "$SDPA_CASE" --corpus "$CORPUS/manifest.json" --deadline-min 180
```

Generate the supplementary table only after pure-SDPA evaluation returns 0 and its report is complete. If it returns 3, resume first.

```bash
python candidates/bench.py summarize --root "$REF_ROOT" --model "$MODEL" --agent seed
```

## Resume and failure records

| Situation | Action |
| --- | --- |
| `budget_exhausted` with a submission | Normal search termination; continue formal verification. It does not imply correctness. |
| Budget exhausted or search ended without a submission | Keep the failure row and all costs. Do not omit the case or grant a free rerun. |
| Finished run | `run` skips it, including finished failure states. Repeating the command does not repair it. |
| Unfinished run or outer timeout | Preserve the API ledger and files; LLM sessions are not currently resumed. Confirm that the old process tree has stopped before other GPU work. |
| `--redo` | Deletes the old run and resets its ledger, so it starts a new trial. Do not use it as if it resumed a formal search. |
| Evaluate exit 3 | Keep submissions, case set, settings, and environment unchanged; rerun the same command. |
| Evaluate exit 0 | The protocol finished. Read every verdict; this does not mean all candidates passed. |
| Gate failed / incomplete | Preserve the reason and diagnostic timings. Exclude it from accepted aggregates and combinations; do not tune thresholds to make it pass. |

The outer bench timeout does not currently guarantee termination of all descendant processes. After a timeout, handle the entire Slurm step/job and confirm the GPU is idle before timing. If a search must be rerun, create a new trial and retain the original ledger and reason. Do not select the best of several searches and report it as one 10-request run.

`evaluate --case-id` writes the same agent-level `evaluation.json` each time; it does not append cases automatically. After debugging one case, run evaluation on the full six-case set for the final result. `summarize` can skip a missing agent and still return success, so completion checks cannot rely on its exit code alone.

## Result tables and delivery checks

The main report contains four groups:

1. **L1/L2 T2:** per-case Gate statuses and runtime/speedup for seed, evograd, Codex, and GEPA; per-level geometric means over accepted configurations, coverage, and the funnel. Aggregation follows configuration -> task -> family. Read the geometric means together with coverage.
2. **L2 single-site T3:** each agent's step time, speedup against eager, 95% interval, all Gate statuses, positive flag, and actual selection flag for every L2 case. Do not average these speedups into a combination benefit.
3. **L3:** each agent's combination members, model Gate 2/Gate 3, performance against eager/whole-model compile, and memory. Explicitly distinguish empty combinations, verification failures, and timing failures.
4. **Costs and references:** actual API requests, evaluator calls, available token usage, wall time, and termination reason for all 18 searches; pure SDPA in a separate supplementary table.

**Always state the coverage denominator.** The model registry contains 14 observed L1 configurations and four L2 configurations. `--case-id` does not change the formal summary's denominator; its numerator counts `executable && accepted`. The maximum full-model coverage for the selected L1 cases is `2/14`; for L2 it is `4/4`. Also report a "selected-case pass rate" using the same numerator divided by 2 for L1 and 4 for L2. Do not conflate these two measures. A performance geometric mean additionally requires a valid speedup for the corresponding baseline. Do not replace missing timings with 0x or 1x. Pure SDPA does not enter the core numerator, and unselected L1 cases are not candidate verification failures.

Before delivery, check that all 18 searches and six core seeds have explicit terminal or failure records; every available candidate maps to a frozen submission; every metered run has at most 10 entries in `api_calls.jsonl`; seed and all three agents have been summarized over the full six-case set with no suspended v2 evaluation; and every L2 single-site and L3 verdict and timing is traceable. If an agent produces no submission for any of its six cases, evaluation will not produce a normal report. Add explicit failure rows to the research table, citing its run ledgers. Do not fabricate an evaluation file or omit the agent.

Retain `run.json`, `submission.py/json`, request and evaluation ledgers, `evaluation.json`, detailed Gate reports, emitted T2 statistics, raw timing blocks in T3 caches, summaries, source/dependency/task-pack/corpus identities, and Slurm logs. Preserve shared state as calibration evidence, not as an agent's candidate result.

Timing confidence intervals describe timing uncertainty under the frozen execution conditions; they do not cover variance across LLM search trials. If single-site T3 generally shows no benefit, interpret it alongside T2 and `cpu_bound_fraction`. Do not label the CPU submission/blocking fraction as pure dispatch overhead. Changes to the protocol, model set, or trial count require a new results root and experiment record.

## Implementation entry points

- [Benchmark driver](bench.py) and [run instructions](README.md)
- [Formal evaluation protocol](../src/evograd/evaluation/protocol.py) and [timing documentation](../src/evograd/evaluation/README.md)
- [Model Gates](../src/evograd/verification/integrations/gates.py) and [Qwen workload](../src/evograd/verification/integrations/qwen3_0_6b/capture.py)
- [API request budget](../src/evograd/adbench_tools/llm_budget.py) and [shared driver configuration](_common.py)
