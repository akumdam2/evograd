# ADBench Six-Agent Evaluation Plan

Date: 2026-10-27. Protocol: `evograd-evaluation-v3`. For rules not repeated here (request accounting, T2/T3 timing, the positive rule), see the [Qwen3-0.6B plan](EVALUATION_PLAN.md) and the [protocol](../src/evograd/evaluation/protocol.py).

## Goal

Six agents each search once on 58 cases from eight models, with a budget of 10 per case: OpenEvolve, GEPA, KernelAgent, Codex, Claude Code and GEAK. Two teammates divide the six agents between them and run the searches. wzhan runs evograd and the frontend-only seed. During search every agent gets Gate 1 as its correctness feedback; Gate 2, Gate 3 and the timings are run by wzhan on the submissions afterwards, once the verification settings are final, in one unified evaluation. Only that evaluation is reported.

## Cases

The 58 cases are:
- every L2 case (28);
- per model, one L1 case per L1 task, using the task's first configuration in the model's snapshot (30).

Tasks whose operator is itself a granted vendor primitive are left out: `linear`, `linear_no_bias` and `causal_gqa_attention/causal_gqa_sdpa`. The case list is `/work/hdd/bcjw/wzhan/ADBench/cases.json`. MegaFold is not in this round.

| Model (`--model`) | L1 | L2 | Corpus under `/work/hdd/bcjw/wzhan/ADBench_payload/` | Gate 3 settings (provisional) |
| --- | --- | --- | --- | --- |
| `qwen3_0_6b` | 4 | 4 | `qwen3_0_6b/20261021_main` | 1000 updates, lr 1e-4 |
| `qwen3_5_4b` | 8 | 7 | `qwen3_5_4b/qwen3_5_4b_20261021` | 200 updates, lr 1e-4 |
| `llama_3_2_1b` | 4 | 4 | `llama_3_2_1b/llama_3_2_1b_20261021` | 1000 updates, warmup 100, clip 1.0, lr 3e-5 |
| `qwen2_vl_2b` | 4 | 2 | `qwen2_vl_2b/qwen2_vl_2b_20261021` | 125 updates of 8 batches, warmup 100, clip 1.0, lr 3e-5 |
| `convnextv2_tiny` | 4 | 4 | `convnextv2_tiny/convnextv2_tiny_20261021` | 1000 updates, lr 1e-4 |
| `dit_s_2` | 1 | 2 | `dit_s_2/dit_s_2_20261021` | 1000 updates, lr 1e-4 |
| `torchrec_dlrm` | 1 | 1 | `torchrec_dlrm/torchrec_dlrm_20261021` | 1000 updates, lr 1e-4 |
| `olmoe_1b_7b` | 4 | 4 | `olmoe_1b_7b/olmoe_1b_7b_20261021` | none (no model-scope gates) |

The corpora are read-only; never copy or write them. Orders 101 and 202 are the calibration split and orders 303 and 404 the acceptance split; Tier 2 times order 303. The corpus index is `/work/hdd/bcjw/wzhan/ADBench_payload/README.md`.

## Verification and timing

| Level | Accepted when | Timing |
| --- | --- | --- |
| L1 | static rules, Gate 1, operator-scope Gate 2 | T2 |
| L2 | the L1 gates plus single-site model-scope Gate 2 and Gate 3 (OLMoE: Gate 1 and operator-scope Gate 2 only) | T2, single-site T3 |
| L3 | the positive L2 candidates of one agent on one model, deployed together: model-scope Gate 2 and Gate 3 | T3 against eager and whole-model `torch.compile` |

A candidate is positive when it is accepted, its single-site T3 succeeds, and the lower 95% bound of that speedup over eager is above 1. An executable candidate that is not accepted is still timed, but only as a diagnostic. The safety margins, the floors and the Gate 3 settings are provisional until the unified evaluation.

## Search feedback

All agents, evograd included, get their correctness feedback from one shared evaluator. It applies Gate 1's rule to real calibration payloads: the first, middle and last call of the case in orders 101 and 202. The compile envelope is precomputed and stored in the task pack. Acceptance payloads are never used as feedback. Timing feedback stays as it was: Tier 1 on generated inputs with the case's shapes.

The rule is stored in each task pack and is the same for every agent; evograd's seed acceptance uses it too. No check during search judges a model case by the declaration's tolerances, and there is no fallback rule.

## Agents and budget

| Agent | LLM | Budget 10 means | Flags after `--` |
| --- | --- | --- | --- |
| OpenEvolve | gpt-5.5 | 10 model requests | none |
| GEPA | gpt-5.5 | 10 model requests | `--eval-timeout 900` |
| KernelAgent | gpt-5.5 | 10 model requests | none |
| Codex | gpt-5.5 | 10 model requests | `--effort high --time-limit 4800 --eval-timeout 900` |
| Claude Code | claude-opus-4-8 | 10 model requests | `--claude-model claude-opus-4-8 --effort high --time-limit 4800 --eval-timeout 900` |
| GEAK | claude-opus-4-8 | 2 optimization directions; requests are counted, not capped | `--claude-model claude-opus-4-8 --effort high --geak-budget 2 --time-limit 4800` |

A local proxy counts model requests in `api_calls.jsonl`. Every admitted request counts, retries included, and request 11 is refused. Evaluator calls (`evaluations.jsonl`), tokens and wall time are recorded separately. GEAK's budget unit differs from the other agents', so it is reported in its own row.

## Setup

- Clone evograd from GitHub and use the latest `main`.
- Build the Python environment from [`requirements-adbench.txt`](requirements-adbench.txt).
- Install KernelAgent, GEAK (`bash candidates/geak/setup.sh`), codex-cli and Claude Code at the versions that file lists.
- Do not put the conda environment on `/work/hdd`: activating environments there has hung.
- Use your own OpenAI and Anthropic API keys. Claude Code and GEAK need `ANTHROPIC_API_KEY` (a `claude` login is not enough): the request proxy forwards with it.
- Check your access to the shared data once. The command should print only `ADBench writable`:

  ```bash
  find /work/hdd/bcjw/wzhan/{ADBench,ADBench_payload,hf_cache,evograd_verification/{hf_home,hf_cache,data}} /work/hdd/bcjw/kramesh/hub/models--meta-llama--Llama-3.2-1B ! -readable 2>&1 | head; test -w /work/hdd/bcjw/wzhan/ADBench && echo "ADBench writable"
  ```

## Running

Run on your own Slurm account, preferably with 2-hour `ghx4-interactive` allocations. Give each process its own GPU, run at most four searches per node, and give each process 8 CPUs and 32g of memory.

```bash
export EVOGRAD=/path/to/your/evograd PATH=/path/to/your/env/bin:$PATH        # edit
AGENT=codex MODEL=qwen3_0_6b                                                  # edit
export ADBENCH_ROOT=/work/hdd/bcjw/wzhan/ADBench/$AGENT ADBENCH_MODEL=gpt-5.5 ADBENCH_BUDGET=10
export PYTHONPATH=$EVOGRAD/src CUBLAS_WORKSPACE_CONFIG=:4096:8 CXX=/usr/bin/g++ DISABLE_AUTOUPDATER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
V=/work/hdd/bcjw/wzhan/evograd_verification
export HF_HOME=$V/hf_home EVOGRAD_BENCHMARK_DATA_ROOT=$V/data EVOGRAD_LLAMA_HF_CACHE=/work/hdd/bcjw/kramesh/hub
case $MODEL in
  qwen3_0_6b|qwen3_5_4b) export EVOGRAD_HF_CACHE=/work/hdd/bcjw/wzhan/hf_cache ;;
  *)                     export EVOGRAD_HF_CACHE=$V/hf_cache ;;
esac
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/$USER/triton TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/$USER/inductor
set +x; set -a; source ~/.adbench_keys; set +a                                 # your API keys
CORPUS=$(python -c "import json; print(json.load(open('/work/hdd/bcjw/wzhan/ADBench/cases.json'))['$MODEL']['corpus'])")
cd "$EVOGRAD"

# one search per case
python candidates/bench.py run --model $MODEL --agent $AGENT --case-id $CASE_ID \
  --budget 10 --time-limit 6000 -- <flags from the table above>
```

- **Smoke first.** It checks the installation and the plumbing on two of the shared task packs. Your smoke root's `tasks` links to them; runs and evaluations stay in the smoke root:

  ```bash
  SMOKE=/work/hdd/bcjw/wzhan/ADBench/smoke/$AGENT; CASES="rmsnorm/rmsnorm qwen3_swiglu_mlp/swiglu_mlp"
  python candidates/bench.py run --root $SMOKE --model qwen3_0_6b --agent $AGENT --case-id $CASES \
    --budget 10 --time-limit 6000 -- <flags from the table above>
  python candidates/bench.py evaluate --root $SMOKE --model qwen3_0_6b --agent $AGENT --case-id $CASES \
    --corpus "$CORPUS" --deadline-min 100     # exit 3: rerun unchanged
  ```

  Check five things: the model actually used, the request count, that `submission.json` exists, that `evaluate` reaches exit 0, and each case's wall time. Start the trial searches only after all five pass. Smoke candidates never enter the trial; smoke costs are reported separately.
- **When to start a case.** Start a case only if the allocation has at least twice the agent's longest smoke case left, and at least 20 minutes.
- **Interrupted searches.** If a search is cut off by the allocation or a node failure, move its run directory and `.log` to `$ADBENCH_ROOT/$MODEL/aborted/` and run the case once more. The rerun replaces the interrupted run whatever its result, and both runs' costs are reported. Never use `--redo`.
- **Failures.** A search that ends without a submission keeps its failure record and is not rerun. Final verdicts are never fed back to an agent.
- **Task packs.** Never run `bench.py tasks`, in the smoke root or the trial root: the packs are prebuilt, read-only, and shared by every agent. Start the smoke only after wzhan says they are ready.

## Outputs

Everything goes under `/work/hdd/bcjw/wzhan/ADBench/<agent>/<model>/`, per case in `runs/<agent>/<case>/`: `run.json`, `submission.py`, `submission.json`, `api_calls.jsonl`, `evaluations.jsonl`, `llm.jsonl` and the agent's logs.

Do not change anything there after a search finishes. Note reruns, and any change of limits or memory, in `/work/hdd/bcjw/wzhan/ADBench/<agent>/NOTES.md`.

## Unified evaluation (wzhan)

The unified evaluation starts once the safety margins, the floors and every model's Gate 3 settings are final. wzhan then:
1. reruns `fidelity` on every corpus checked only at capture time;
2. checks each `submission.py` against the `program_sha256` in its `submission.json`;
3. evaluates seed, evograd and the six agents under `/work/hdd/bcjw/wzhan/ADBench/unified/`, with one shared state;
4. summarizes each model.

The tables report, per model:
- **L1/L2:** gate statuses and T2, with coverage over every observed configuration and the pass rate over the selected cases;
- **L2 single-site T3**;
- **L3**;
- **cost:** requests or directions, evaluator calls, tokens and wall time.

## Known limitations

Codex, Claude Code and GEAK have a shell as their owner's Unix user, so they could read the corpora or other agents' outputs; wzhan checks their transcripts for such access before the unified evaluation. Each agent searches each case once, so variance across searches is not measured.

## Phase 0 (wzhan, before the trial searches)

1. **Fixes, with tests.** Each one is listed with what it fixes or adds:
   - Task-pack statements: give the model's real scalar values (eps and others).
   - The qkv_norm_rope backward formula.
   - `parameter_args` for gelu and quick_gelu.
   - The unified frontend: accept convolutions with padding 0.
   - Search feedback: Gate 1 on calibration payloads, with the compile envelope stored in the task pack, for every agent and for evograd's seed acceptance. The declared-tolerance rule is removed for model cases.
   - The request proxy: an Anthropic endpoint. Claude Code's background requests are turned off.
   - GEAK: request counting and `--effort`.
   - The requirements file.
2. **Corpora.** Remove group write permission.
3. **Weights.** Copy the Qwen3-0.6B weights into `/work/hdd/bcjw/wzhan/hf_cache`.
4. **Task packs.** Build all 58 under `ADBench/tasks/`, link them into every agent root and smoke root, and write `cases.json`.
