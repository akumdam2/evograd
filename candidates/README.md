# Other agents on the evograd benchmark

This directory runs other kernel agents on evograd's operators so they can be
compared with evograd. The design (principles, agents, budgets, code layout,
how `TASK.md` and the other pieces are built) is in
[`src/evograd/adbench_tools/README.md`](../src/evograd/adbench_tools/README.md).
Each agent is judged on the same terms:

| Agent | What runs | Directory |
|---|---|---|
| **seed** (reference point) | evograd's frontend alone, no LLM | `seed/` |
| **evograd** (ours) | unified frontend (PyTorch forward → Triton seed), then OpenEvolve; `--frontend unified\|a\|b\|d` selects the seed frontend (default `unified`) | `evograd/` |
| **OpenEvolve, raw** | the same OpenEvolve backend, without the frontend | `openevolve/` |
| **GEPA** | `gepa.optimize_anything`, reflective Pareto search | `gepa/` |
| **KernelAgent** (Meta) | `TritonKernelAgent` generation, then `OptimizationManager` | `kernelagent/` |
| **GEAK v4** (AMD) | Claude Code `kernel_workflow`, author mode, on NVIDIA | `geak/` |
| **Claude Code** (Anthropic) | `claude -p`, headless, in a workspace with a metered `evaluate.py` | `claude_code/` |
| **Codex** (OpenAI) | `codex exec`, headless, in the same workspace | `codex/` |

Only the agent-specific adapters live here. Everything shared lives in
`src/evograd/adbench_tools`, which reads the benchmark and evaluator and changes
neither of them:
- the task pack;
- the problem statement;
- packaging an agent's code;
- the evaluator every agent goes through;
- the ledger and the final score.

## The benchmark run

The [Qwen3-0.6B evaluation plan](EVALUATION_PLAN.md)
fixes the first comparison's six cases, 10-request budget, gates, timing and reporting.

The comparison that counts is `evograd evaluate` (`evograd-evaluation-v3`:
real-payload gates, Tier 2, single-site Tier 3 and the Level-3 combination), run
on every agent's submissions by `bench.py`; `score`/`compare` below are
single-configuration diagnostics. See
[`adbench_tools/README.md` §7](../src/evograd/adbench_tools/README.md#7-the-benchmark-run).

```bash
export ADBENCH_ROOT=...     # results under $ADBENCH_ROOT/<model>
export ADBENCH_MODEL=gpt-5.5
python candidates/bench.py tasks     --model qwen3_0_6b --case-id qwen3_swiglu_mlp/swiglu_mlp rmsnorm/rmsnorm \
                                     --corpus CORPUS/manifest.json   # search feedback: Gate 1 on its payloads
python candidates/bench.py run       --model qwen3_0_6b --agent seed            # frontend only, no LLM
python candidates/bench.py run       --model qwen3_0_6b --agent gepa --budget 10
python candidates/bench.py evaluate  --model qwen3_0_6b --agent gepa --corpus CORPUS/manifest.json
python candidates/bench.py summarize --model qwen3_0_6b --agent seed gepa
```

| Level | Verification | Formal performance report |
|---|---|---|
| L1 | Gate 1 + operator Gate 2 | T2 |
| L2 | Gate 1 + operator Gate 2 + single-site model Gate 2 + Gate 3 | T2 + single-site T3 |
| L3 combination | Combined model Gate 2 + Gate 3 | Combined T3 |

T2 runs the artifact's real deployment forward and `torch.autograd.grad`, with
every baseline using the same autograd scope; it has no optimizer or `.grad`
accumulation. Its fair timer retains warmup 10, repetitions 50 and 3 blocks.
T1 remains search feedback or a separate pair diagnostic. Each agent's L3
combination contains its accepted L2 candidates whose successful single-site
T3 speedup over eager has a 95% interval with lower bound greater than 1.
The summary retains these individual T3 results, gates and selection decisions;
it does not average them into an end-to-end speedup. L3 is measured separately
against eager and whole-model compile; its reported interval is against eager.

Use a fresh run root for the new API-request budget. `bench.py run` refuses to
reuse a metered agent's completed run if its budget unit or limit differs, and
does not silently delete an unfinished run or reset its API ledger. `--redo`
explicitly replaces that run as a new trial. Existing v1 and v2 evaluations must be
rerun with `evaluate`; renaming old `tier1` results to `tier2` is not a migration.

## Fairness protocol

1. **Same input.** A *task pack* (`python -m evograd.adbench_tools task`) holds
   what evograd's frontend is given, and nothing more:
   - the declared PyTorch forward reference: its full source, extracted from
     evograd and checked bit-identical to the declared forward for all 63
     operators;
   - argument activity, shapes and dtypes;
   - one fixed configuration.

   `declaration.py` is written by the same function `evograd seed unified`
   uses, so every agent faces the same correctness cases, tolerances and timed
   shape. By default the configuration is the operator's first timed case
   (`--case correctness` selects the frontend's own default instead).
2. **Same instructions.** `TASK.md` begins with evograd's OpenEvolve system
   message for that operator, verbatim (with the default frontend; with A, B or
   D on a case that grants primitives, evograd's message adds a paragraph on the
   grant); `task.json` records its SHA-256. The rest of the statement replaces
   what evograd's seed carries (a program that computes the operator): the
   reference source, the configuration, the deliverable, and the static rules
   the evaluator enforces. Those rules are read from the enforcement code
   itself. Each agent receives this text in its own prompt slot:

   | Agent | Where the statement goes |
   |---|---|
   | GEPA | `background` |
   | KernelAgent | `problem_description` |
   | GEAK | `TASK.md` / `meta.json` in its op task dir |
   | Claude Code, Codex | the CLI prompt: one paragraph on the workspace, then the statement (also `TASK.md` in the workspace) |
   | raw OpenEvolve | system message |
   | evograd | its own system message (section 1) plus its seed |

   Each agent's own internal prompts are left unchanged.
3. **Same deliverable.** Every agent writes one module defining
   `<op>_forward_with_saved` and `<op>_backward_from_saved`, in Triton and/or
   PyTorch tensor code. The harness, not the agent, then turns it into an
   evograd artifact:
   - the *whole* module goes inside one `EVOLVE-BLOCK`, so every line the agent
     wrote is under the static rules;
   - the declaration-generated autograd/deployment layer is appended after the
     block, exactly as evograd's own pipelines emit it.

   Library access is matched to the frontend's. The unified frontend keeps
   GEMMs and convolutions in vendor library calls. So `--library auto` (the
   default) grants the trusted `vendor_gemm` primitive (cuBLAS) whenever the
   reference contains a contraction.
4. **Same judge.** In-loop scores and feedback come from
   `evograd.evolve.evaluator_entry`, the evaluator `evograd evolve` uses, with
   the environment it sets.
   - GEPA calls it in-process.
   - KernelAgent's test file, GEAK's `unittest.py` and the coding agents'
     `evaluate.py` are generated by this harness; they call
     `python -m evograd.adbench_tools evaluate` and print its feedback.
   - OpenEvolve and evograd use the evaluator directly.

   No agent's own correctness check or timer decides anything.
5. **Same final measurement.** `bench.py` rebuilds each submission from its
   block body (`fixed_code_intact` records outside edits), then runs the v3
   protocol above. `adbench_tools score` is a separate single-configuration
   diagnostic: seed acceptance and the search evaluator at 25 warmup / 100
   repetitions, with a shared cached baseline.
6. **Same model and model-request cap for the OpenAI agents.**
   - `--model` and `--api-base` are one OpenAI-compatible endpoint for every
     agent except GEAK and Claude Code, which run on Anthropic's API
     (`--api-base` defaults to `$ADBENCH_ANTHROPIC_API_BASE` or
     `https://api.anthropic.com` for them).
   - `--budget N` caps admitted model API requests per agent and case,
     Claude Code's included. The default is `ADBENCH_BUDGET`, or 10. Initial
     generation, reflection, retries, failed upstream requests and context
     compaction all consume it.
   - The shared HTTP proxy records requests in `api_calls.jsonl` before
     forwarding them and refuses request N+1. Native loop limits below are
     additional stopping conditions, not alternative budget units.
   - `evaluations.jsonl` independently records evaluator calls; `llm.jsonl`
     holds agent-reported usage events, which need not correspond to requests.

| Agent | Model-request cap | Additional native stopping control |
|---|---|---|
| evograd | N | OpenEvolve `iterations = N` minus the seed's requests; only `--frontend a` uses the LLM for its seed |
| raw OpenEvolve | N | `iterations = N`; starts from the skeleton |
| GEPA | N | `max_evals = N + 1`; initial generation also consumes the API cap |
| KernelAgent | N across all workers and stages | generation: 4 workers × ⌈N/8⌉ rounds; optimization: max(1, ⌊N/8⌋) beam rounds |
| Codex | N | session time limit; optional independent `--eval-budget`; no evaluator-call cap by default |
| Claude Code | N | session time limit; optional independent `--eval-budget`; no evaluator-call cap by default |
| GEAK | counted, not capped | `budget = max(2, N // 5)` optimization directions (`budget_unit: optimization_directions`) |

Claude Code and GEAK are separate data points: both run on a Claude model,
and GEAK's budget is in optimization directions. The proxy in front of
Anthropic's API caps Claude Code's requests at N and only counts GEAK's, so
both have a `llm_calls_recorded` count. Every request of the claude CLI counts
(subagents and compaction too); `/v1/messages/count_tokens` is refused and
Claude Code estimates tokens locally instead. Runs from before this have
`llm_calls_recorded: null` (`unmetered`); token or session usage events do not
establish a request count.

## What cannot be equalised (and is recorded instead)

- **GEAK targets AMD.** It runs here on NVIDIA through a small overlay
  (`geak/setup.sh`) and nothing else in GEAK is modified:
  - an NVIDIA GPU-identity probe replaces the rocminfo one;
  - Nsight Compute plugs into GEAK's generic `metrix` profiler hook;
  - its steer says the GPU is NVIDIA, because its knowledge base is
    AMD-specific.

  GEAK also runs on Claude Code's model (`--claude-model`), not on `--model`.
  Report it as a separate model and budget point.
- **Claude Code and Codex have a shell on the GPU node.** Their own
  sandboxes block the GPU, so they run with permissions bypassed. Nothing
  they are given names the repository, the run or the task pack: the
  workspace is a fresh directory outside the repository, `evaluate.py`
  reaches the evaluator through a Unix socket served by the driver (which
  redacts those paths from its feedback), and their environment is scrubbed
  of them. They run as the same user, though, so a filesystem or process
  search could still find the repository (the first Claude Code trial read
  evograd's scoring source through the path `evaluate.py` then embedded).
  `run.json` records `oracle_intact` (whether `evaluate.py` was edited) and
  `deliverable` (which rule chose the measured program).
  Claude Code runs on `--claude-model`, like GEAK. Codex uses an explicit
  `adbench` HTTP Responses provider pointed at the shared budget proxy, with
  WebSockets and stream reconnection retries disabled, and a placeholder key
  (`adbench-local`). Other request retries still consume the budget. The proxy holds
  the upstream credential and forwards to `--api-base`; that endpoint must
  support `/responses` and, when used, `/responses/compact`.
- **KernelAgent times one callable.** So its `problem.py` presents the task as
  one training step (eager autograd of the reference), and the module must also
  define a fixed `kernel_function` wrapper around the pair. Our test checks the
  wrapper is the pair. KernelAgent's own bans (e.g. no `torch.sigmoid`) are
  stricter than evograd's and stay on; they are part of the agent.
- **Raw OpenEvolve's first correct program is the LLM's.** Evograd's seed is
  correct before evolution starts. That difference *is* the frontend's
  contribution, and it is what this comparison measures.

## Install (on the GPU node)

```bash
pip install -e ".[gpu,llm]"                      # evograd (includes openevolve)
export OPENAI_API_KEY=...                        # OpenAI-model agents
export ADBENCH_MODEL=gpt-5.5                      # this comparison; overrides the unchanged driver default
export ADBENCH_BUDGET=10
export ADBENCH_API_BASE=https://api.openai.com/v1

# GEPA, at the commit the driver was written against
pip install -e references/gepa                   # 8f287de942f41a6d1e0e7e5b2d576247c4bedee9
# KernelAgent
pip install -e references/KernelAgent            # e0647170da36ef9b059ac0bd3d60103aa4ed378b
# GEAK: Claude Code >= 2.1.177 on PATH, ANTHROPIC_API_KEY (held by the request proxy;
# ADBENCH_ANTHROPIC_API_BASE for a gateway), ncu on PATH
bash candidates/geak/setup.sh                    # NVIDIA working copy -> candidates/geak/.work/GEAK
# Coding agents: native CLIs on PATH (not pip packages; see pyproject.toml)
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.310  # Claude Code -> ~/.local/bin/claude;
                                                 # ANTHROPIC_API_KEY (a `claude` login is not used)
npm install -g @openai/codex@0.149.1             # Codex; uses OPENAI_API_KEY, --model, --api-base
```

## Commands

Build the task pack once per operator configuration. Every agent then runs on
that same directory.

```bash
TASK=results/adbench_tools/llama3_swiglu_mlp/task
python -m evograd.adbench_tools task --op llama3_swiglu_mlp --dtype bfloat16 --output-dir $TASK
python -m evograd.adbench_tools statement $TASK | less      # what every agent is told
```

Run each agent. Every command ends with the diagnostic measurement in
`<run-dir>/final.json`:

```bash
R=results/adbench_tools/llama3_swiglu_mlp/runs
python candidates/evograd/run.py     --task $TASK --run-dir $R/evograd     --budget 10
python candidates/openevolve/run.py  --task $TASK --run-dir $R/openevolve  --budget 10
python candidates/gepa/run.py        --task $TASK --run-dir $R/gepa        --budget 10
python candidates/kernelagent/run.py --task $TASK --run-dir $R/kernelagent --budget 10
python candidates/geak/run.py        --task $TASK --run-dir $R/geak        --budget 10 \
    --claude-model claude-opus-5-5
python candidates/codex/run.py       --task $TASK --run-dir $R/codex       --budget 10
python candidates/claude_code/run.py --task $TASK --run-dir $R/claude_code --budget 10 \
    --claude-model claude-opus-5-5
```

Compare them:

```bash
python -m evograd.adbench_tools compare $R/*/final.json
```

Or run everything for one operator, one agent after another on the same GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash candidates/run_all.sh llama3_swiglu_mlp bfloat16 10
AGENTS="evograd gepa" bash candidates/run_all.sh llama3_attention        # a subset
```

To check and run the five OpenAI-model agents (no GEAK or Claude Code) in one go, use
`run_openai.sh`. It runs unit tests, builds the task pack, does a no-LLM check
of the evaluator, a tiny-budget smoke run of each agent, then the full run:

```bash
SETUP=1 bash candidates/run_openai.sh llama3_swiglu_mlp bfloat16 10   # first time: also installs GEPA/KernelAgent
STAGES="sanity smoke" bash candidates/run_openai.sh                    # just the checks
```

Score or re-score any program by hand. It may be a bare agent module or a
packaged artifact:

```bash
python -m evograd.adbench_tools evaluate --task $TASK some_module.py          # in-loop evaluator
python -m evograd.adbench_tools evaluate --task $TASK --correctness-only some_module.py
python -m evograd.adbench_tools score --task $TASK --run-dir $R/manual some_module.py
```

`final.json` → `follow_up` contains optional standalone tier diagnostics.
Their declared-case checks and timers are separate from the formal v3 protocol
run by `bench.py evaluate`:

```bash
evograd tier2-bench --op llama3_swiglu_mlp --candidate $R/gepa/final_program.py \
    --baseline none --dtype bfloat16
evograd tier3-bench --model llama_3_2_1b --patch-set gepa:swiglu_mlp=$R/gepa/final_program.py ...
```

## Run directory

```
<run-dir>/
  run.json             agent, task, statement hash, config (budget unit/limit, model), status, wall time
  evaluations.jsonl    every evaluation the agent requested (kind, verdict, score, program)
  api_calls.jsonl      admitted model API requests, independently metered for OpenAI agents
  llm.jsonl            LLM usage, where the agent exposes it (GEPA metadata, Claude Code cost, Codex tokens)
  programs/            every packaged program evaluated through adbench-tools
  final_program.py     the re-packaged best program that was measured
  final.json           acceptance, correctness, combined_score, speedups, saved memory, ledger summary
  final/               acceptance and evaluation reports
  prompt.md            the coding agents' prompt (Claude Code, Codex)
  workspace/           the coding agents' workspace as they left it
  ...                  the agent's own artifacts (evolve/, gepa_state/, generation/, exp/,
                       claude_stream.jsonl, codex_events.jsonl, ...)
```
