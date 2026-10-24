# adbench-tools: running kernel agents on adbench

adbench-tools is a helper for the benchmark. It runs other LLM kernel agents
on its operators and compares them with evograd's own agent, on the same terms.
It only uses the benchmark and evaluator, and changes neither. This document covers:

- the principles of the comparison;
- the agents and what each one needs;
- the current budgets;
- how the code is split;
- how the key pieces are built: the task pack, `TASK.md`, packaging, the
  evaluator and the final score.

For commands only, see `candidates/README.md`, or run `candidates/run_openai.sh`.

---

## 1. Principles

### 1.1 What is being compared

Every agent faces the same problem. It gets an operator's **PyTorch forward
function** at one fixed shape and dtype, and must deliver a **forward and
backward** for it, in Triton and/or PyTorch tensor code, that is correct
against `torch.autograd` and faster than PyTorch eager.

| Agent | What runs | Status |
|---|---|---|
| **evograd** (ours) | Unified frontend (PyTorch forward → correct Triton seed), then the OpenEvolve backend | active |
| **OpenEvolve, raw** | The same OpenEvolve backend and configuration, without the frontend, starting from an empty skeleton | active |
| **GEPA** | `gepa.optimize_anything`: reflective, Pareto-based text evolution | active |
| **KernelAgent** (Meta) | `TritonKernelAgent` parallel generation, then `OptimizationManager` beam search | active |
| **GEAK v4** (AMD) | Claude Code `kernel_workflow`, author mode, adapted to NVIDIA | implemented, parked |
| **Claude Code** (Anthropic) | The `claude` CLI, headless, in a workspace with a metered evaluator | implemented, separate model |
| **Codex** (OpenAI) | The `codex` CLI, headless, in the same workspace | active |

The current round runs only agents on OpenAI models: evograd, raw OpenEvolve,
GEPA, KernelAgent and Codex. GEAK and Claude Code run on Claude Code's model,
so they can't share the same model. Their drivers work the same way, but they
are left out of `run_openai.sh`.

Claude Code and Codex are the general-purpose coding agents: no kernel-specific
loop, only a shell, a file editor and the evaluator. They show what an
off-the-shelf agent does with the same statement and budget.

Raw OpenEvolve is the key ablation. It is evograd minus the frontend: same
backend, same configuration, same evaluator, same budget. The gap between the
two is what the frontend contributes.

### 1.2 Fairness rules

1. **Same input.** Each agent gets what evograd's frontend gets and nothing
   more:
   - the declared PyTorch forward reference;
   - which arguments need gradients;
   - the shapes and dtypes;
   - one fixed configuration.

   Nobody gets a backward, an autograd graph or test tensors. evograd's own
   head start (a correct seed) is produced by its frontend from that same
   input.
2. **Same instructions.** Every agent's problem text starts with evograd's own
   OpenEvolve system message for the operator, word for word (its hash is
   recorded). The text goes into whatever prompt slot the agent provides. The
   agents' internal prompts are not changed.
3. **Same deliverable.** Every agent produces one Python module that defines
   `<op>_forward_with_saved` and `<op>_backward_from_saved`. The harness, not
   the agent, turns it into an evograd artifact, so every line the agent wrote
   is checked by the same static rules.
4. **Same judge.** Every score and pass/fail an agent sees comes from
   `evograd.evolve.evaluator_entry`, the evaluator `evograd evolve` uses, with
   the environment it sets. KernelAgent's test file and GEAK's unit test are
   written by us and call that evaluator. No agent's own checker or timer
   decides anything.
5. **Same final measurement.** The agent's best program is rebuilt from the
   code inside its block, so edits outside it are discarded. It must then pass
   the seed-acceptance check evograd's own seed passes, and is timed with more
   repetitions against a baseline measured once and shared by all agents.
6. **Same model and budget, as far as each agent allows.**
   - One `--model` / `--api-base` (an OpenAI-compatible endpoint) is used for
     every LLM call.
   - One `--budget N` is translated into each agent's own controls (§3).
   - Every evaluation actually requested is logged, so differences in budget
     use are visible rather than hidden.

### 1.3 What cannot be made equal (and is recorded instead)

- **evograd starts from a correct program; the others start from nothing.**
  This is not a fairness bug. It is the frontend's contribution, and what the
  comparison measures.
- **KernelAgent can only time one function.** Its `problem.py` therefore
  presents the task as one training step: the reference's forward plus
  backward under PyTorch autograd. The module must also contain a fixed
  `kernel_function` wrapper that calls the pair; our test checks that the
  wrapper is just the pair. KernelAgent's own bans (e.g. no `torch.sigmoid`)
  are stricter than ours and are left on, as part of that agent.
- **GEAK** targets AMD GPUs and runs on Claude Code's model (see §2.5).
- **Claude Code and Codex have a shell on the GPU node** (see §2.6). Their
  own sandboxes would block the GPU, so they run with permissions bypassed.
  Only the prompt keeps them inside their workspace, which is a fresh
  directory outside the repository. Whether they edited `evaluate.py` is
  recorded. Claude Code runs on a Claude model.
- **Library access.** evograd's frontend keeps matrix multiplications and
  convolutions in cuBLAS/cuDNN calls. So with `--library auto` (the default),
  every agent may call a trusted `vendor_gemm` (cuBLAS) primitive whenever the
  reference contains a matrix multiplication. Every other contraction must be
  written in Triton.

---

## 2. The agents and what each needs

All agents receive the problem as text: an LLM prompt, in Markdown, with code
in fenced blocks. They differ in the loop they run around the model and in how
they take code out of its reply. We supply the problem text and the judge, and
never change an agent's internals.

### 2.1 evograd (ours): `candidates/evograd/run.py`

| | |
|---|---|
| Needs | evograd only (`pip install -e ".[gpu,llm]"`) |
| Input | Nothing from the task pack. It gets its usual unified seed, built at the task's exact shape and dtype, and its usual system message |
| Loop | `synthesize_unified_seed`, then `run_evolve`: exactly the functions `evograd seed unified` and `evograd evolve` call. Nothing in evograd was changed |
| Feedback | `evaluator_entry`, with the unified contract enforced (as always for a unified seed) |
| Code from reply | OpenEvolve SEARCH/REPLACE diffs (`diff_based_evolution: true`) |
| Output | `evolve/evolved_best_program.py` |

### 2.2 Raw OpenEvolve: `candidates/openevolve/run.py`

| | |
|---|---|
| Needs | Nothing beyond evograd. It uses the `openevolve` package evograd already depends on |
| Input | Starts from `skeleton.py` (signatures that raise `NotImplementedError`), packaged as an artifact. The system message is `TASK.md` (`--prompt statement`, default). `--prompt evograd` keeps evograd's system message alone, so the rules arrive only as evaluator feedback (strict ablation) |
| Loop | `run_evolve` with the same `render_config` as evograd |
| Feedback | `evaluator_entry` |
| Code from reply | SEARCH/REPLACE diffs |
| Output | The best program from OpenEvolve |

### 2.3 GEPA: `candidates/gepa/run.py`

| | |
|---|---|
| Needs | `gepa` at commit `8f287de9…` and `litellm`. Installed from git; no checkout needed |
| Input | `background` = `TASK.md`, plus a one-sentence `objective`. No starting program by default: GEPA's seedless mode, where its model writes the first module. `--seed-from skeleton` starts from the skeleton instead |
| Loop | `optimize_anything`, single-task mode, its own reflection prompt; the reflection model is `openai/<model>` via litellm |
| Feedback | A Python callback that runs our evaluator. It returns `(score, side_info)`, where score = `combined_score` if correct, else 0, and side_info holds the verdict and the evaluator's feedback text |
| Code from reply | GEPA takes the text between the first and last ```` ``` ```` |
| Output | `best.py`: GEPA's chosen candidate, or the best correct one it evaluated if its choice was never correct |

### 2.4 KernelAgent: `candidates/kernelagent/run.py`

| | |
|---|---|
| Needs | A checkout at commit `e0647170…`, installed with `pip install -e` (not on PyPI; a non-editable install drops `utils.providers`). The driver reads the checkout's `examples/configs/beam_search.yaml` (`--kernelagent-root`, default `references/KernelAgent`) |
| Input | `problem_description` = `TASK.md` plus a short section on its harness, including the fixed `kernel_function` wrapper. `problem.py`: `Model` runs one forward+backward step of the real reference under autograd, and `get_inputs` uses evograd's own input generator. `test_code`: our test (`generate_default_test=False`, so its model never writes the judge) |
| Loop | Generation: parallel workers write `kernel.py`, run the test and fix it from the failure output, until the first one passes. Optimization: Nsight Compute profiling, then a model diagnoses the bottleneck and rewrites, with beam search over the results |
| Feedback | Our test runs the evaluator in correctness-only mode and prints its verdict, then checks that `kernel_function` matches the pair |
| Code from reply | ```` ```python ```` blocks; it prefers the one that defines `kernel_function` |
| Output | `optimized_kernel.py`, or `generated_kernel.py` if optimization is skipped or fails |
| Notes | The model name is registered with KernelAgent's OpenAI provider (`OPENAI_BASE_URL` is set). KernelAgent's GPU table has no GH200, so `--gpu-name` defaults to `NVIDIA H100 SXM5 80GB` |

We use `TritonKernelAgent` and `OptimizationManager` directly, not KernelAgent's
`Fuser` (which splits a whole model into fusable groups), because our tasks are
already single operators.

### 2.5 GEAK v4 (parked): `candidates/geak/`

| | |
|---|---|
| Needs | A checkout at commit `02095f2c…`, prepared by `candidates/geak/setup.sh` into `candidates/geak/.work/GEAK`; the `claude` CLI ≥ 2.1.177; `ANTHROPIC_API_KEY`; `ncu` |
| Input | An op task folder made by `prepare.py`: `TASK.md`, `meta.json`, the reference step as its baseline, and an `unittest.py` it may not edit. The only steering is one sentence: NVIDIA GPU, write Triton, deliverable path |
| NVIDIA overlay | An NVIDIA GPU-identity probe replaces the `rocminfo` one. Nsight Compute is connected through GEAK's own generic `metrix` profiler hook. Nothing else in GEAK is changed |
| Feedback | `unittest.py` runs our evaluator and prints PASS/FAIL, the latencies and `combined_score` |
| Code from reply | None to extract: Claude Code agents write `kernel_src/candidate.py` with their file tools |
| Why parked | It can't use the same OpenAI model as the other agents |

### 2.6 Claude Code and Codex: `candidates/claude_code/`, `candidates/codex/`

Both run through `candidates/_coding_agent.py`, which builds the same workspace
and prompt for each, launches the CLI and picks the program to measure.

| | |
|---|---|
| Needs | Claude Code: the `claude` CLI (`curl -fsSL https://claude.ai/install.sh \| bash -s 2.1.310`) and `ANTHROPIC_API_KEY` or a `claude` login. Codex: the `codex` CLI (`npm install -g @openai/codex@0.149.1`) and `OPENAI_API_KEY`. Both are native CLIs, not pip packages (see `pyproject.toml`). `run.json` records the version each run used, and Claude Code's self-updater is off during a run |
| Input | A workspace with `TASK.md`, `candidate.py` (the skeleton), `adbench_primitives.py` (if granted) and an `evaluate.py` it may not edit. The prompt (`prompt.md` in the run) is one paragraph on the workspace, the budget and the deliverable, followed by `TASK.md` verbatim. It is the same for both agents |
| Loop | The agent's own: `claude -p` (stream-JSON output) or `codex exec --json`, headless, permissions bypassed, until it stops or `--time-limit` (default 6 h) |
| Isolation | The workspace is a new temporary directory, copied to `<run>/workspace` afterwards (`--workspace` overrides). User settings are not loaded: Claude Code gets `--setting-sources project,local --strict-mcp-config`, Codex gets `--ignore-user-config`. When the session ends, the CLI's whole process group is killed, so nothing it started competes with the final measurement |
| Model | Claude Code: `--claude-model`. Codex: `--model` at `--api-base` (set as `openai_base_url`, Responses API), with `OPENAI_API_KEY` passed as `CODEX_API_KEY`. `--effort` sets either agent's reasoning effort |
| Feedback | `evaluate.py` runs our evaluator (full, or `--correctness`) on `candidate.py` or a given path, one call at a time. It prints the evaluator's feedback, `CORRECT`/`INCORRECT` and the evaluations left |
| Code from reply | None to extract: the agent writes `candidate.py` with its file tools |
| Output | `candidate.py` if `evaluate.py` passed exactly that module (the agent's choice). Otherwise, the best correct module it evaluated (full evaluations by `combined_score` first, then the latest). Otherwise, `candidate.py` as it is, unless it is still the skeleton. `run.json` records which rule applied (`deliverable`) |

### 2.7 Shared requirements

- An NVIDIA GPU with CUDA, Triton and `evograd[gpu,llm]` installed.
- `OPENAI_API_KEY`. Optionally `ADBENCH_MODEL` (default `gpt-5.6-sol`) and
  `ADBENCH_API_BASE` (default `https://api.openai.com/v1`).
- `references/` is gitignored. The KernelAgent (and GEAK) checkouts must exist
  on the machine; `SETUP=1 bash candidates/run_openai.sh` creates them.

---

## 3. Budgets

The budget `--budget N` (default **30**) is *the number of candidate programs
the agent may have evaluated*. Each agent counts work differently, so N is
translated as follows:

| Agent | `--budget N` becomes | At N = 30 |
|---|---|---|
| evograd | OpenEvolve `iterations = N` (the seed costs no LLM call) | 30 iterations |
| raw OpenEvolve | `iterations = N` | 30 iterations |
| GEPA | `max_evals = N + 1` (one evaluation goes to its first candidate) | 31 evaluations |
| KernelAgent | Generation: 4 workers × ⌈N/2/4⌉ rounds (stops at the first pass). Optimization: max(1, ⌊N/2/4⌋) beam-search rounds | 4 × 4 rounds, then 3 rounds |
| GEAK | `budget = max(2, N // 5)` optimization directions | 6 directions |
| Claude Code, Codex | At most N calls of `evaluate.py`, full or correctness-only. Call N+1 is refused | 30 evaluations |

Every evaluation an agent actually requests is logged, and `compare` reports
the count.

Other limits:

| Setting | Value |
|---|---|
| Final measurement | 25 warmup / 100 repetitions (in-loop runs use the evaluator's defaults) |
| Per-evaluation timeout | 900 s (GEPA, the CLI, the coding agents' `evaluate.py`), 1200 s (KernelAgent test) |
| Per-agent wall time in `run_all.sh` | `TIME_LIMIT`, default 21600 s (6 h); the coding agents also stop at `--time-limit` (default 6 h) |
| Smoke run (`run_openai.sh`) | budget 2 per agent; KernelAgent 8 with `--skip-optimization`; 1 h each |
| Task configuration | The operator's first timed benchmark case (`--case benchmark`), or the first correctness case if none is declared |

---

## 4. Code layout

The rule: anything that doesn't depend on a particular agent lives in
`src/evograd/adbench_tools/`, which *uses* the benchmark and evaluator and changes
neither. Anything specific to one agent lives in `candidates/<agent>/`.

### 4.1 `src/evograd/adbench_tools/`: shared harness

| File | Role |
|---|---|
| `__init__.py` | The fairness rules (module docstring); schema names `adbench-tools-task/1` and `-result/1` |
| `__main__.py` | Command line: `task`, `statement`, `package`, `evaluate`, `score`, `compare` |
| `task.py` | Builds and loads the **task pack**: picks the configuration, writes the declaration, extracts the reference, describes the forward/backward signatures, writes all files (§5.1) |
| `statement.py` | Renders **`TASK.md`** (§5.2) |
| `artifact.py` | **Packaging** (§5.3): extracts code from a reply, checks the module, wraps it into an evograd artifact, unwraps it; renders the skeleton and `adbench_primitives.py` |
| `evaluate.py` | **The evaluator every agent goes through** (§5.4): full or correctness-only, in a child process that can be killed; Python API and command line |
| `_check.py` | The correctness-only child: static rules, import, correctness and a run at the timed shape, no timing |
| `ledger.py` | **Run log** (§5.5): `run.json`, `evaluations.jsonl`, `llm.jsonl` |
| `score.py` | **Final measurement** and `compare` (§5.6) |
| `_example_TASK.md` | A generated `TASK.md` for `llama3_swiglu_mlp`, bf16, B=2 T=2048 H=2048 I=8192, for reading |

### 4.2 `candidates/`: one folder per agent

| Path | Role |
|---|---|
| `_common.py` | Flags shared by all drivers (`--task --run-dir --budget --model --api-base --seed --no-score`), ledger setup, final scoring, importing OpenEvolve's program archive into the ledger |
| `evograd/run.py` | Our agent: unified seed, then `run_evolve` |
| `openevolve/run.py` | Raw OpenEvolve: skeleton seed, `TASK.md` as the system message |
| `gepa/run.py` | GEPA driver |
| `kernelagent/run.py` | KernelAgent driver: writes `problem.py`, the test file and the problem description, then runs generation and optimization |
| `geak/setup.sh`, `prepare.py`, `run.py`, `nvidia/` | GEAK working copy, its op task folder, the Claude Code launch, and the NVIDIA overlay (GPU probe, `metrix` → `ncu` shim) |
| `_coding_agent.py` | What Claude Code and Codex share: the workspace, the metered `evaluate.py`, the prompt, the launch, and the choice of deliverable |
| `claude_code/run.py`, `codex/run.py` | The two CLI launches and their usage records |
| `run_all.sh` | Builds the task pack, runs the chosen agents one after another on one GPU, then `compare` |
| `run_openai.sh` | The OpenAI-model agents (`AGENTS`, default all five), end to end: setup, unit tests, task pack, sanity check without an LLM, smoke run, full run |
| `README.md` | Commands |

Tests: `tests/adbench_tools/test_adbench_tools.py` (25 tests, CPU-only).

### 4.3 Data flow

```
evograd declaration ──► task pack (task.py)
                         ├─ TASK.md (statement.py) ─────► agent prompt slot
                         ├─ skeleton.py / problem.py ───► agent starting point (where the agent has one)
                         └─ declaration.py ─────────────► evaluator configuration
agent loop ── candidate module ──► package (artifact.py) ──► evaluator_entry (evaluate.py)
                                                         └─► feedback text ──► agent
                                                         └─► evaluations.jsonl (ledger.py)
best module ──► score.py: rebuild from block → seed acceptance → 25/100 timing → final.json
final.json × agents ──► compare ──► Markdown table
```

---

## 5. Key components

### 5.1 The task pack

`python -m evograd.adbench_tools task --op OP --dtype DTYPE --output-dir DIR`
builds one directory, which every agent then runs on. It refuses to overwrite
an existing pack, so all agents in a comparison see the same bytes.

| File | Content | Built from |
|---|---|---|
| `declaration.py` | The operator's declaration narrowed to the one configuration: correctness cases, tolerances, timed shape | The unified frontend's own `select_workload` + `_write_declaration`, byte-identical to what `evograd seed unified` writes |
| `reference.py` | The forward reference as a self-contained file, ending in `reference_forward = <fn>` | `reference_closure`: starts from the declared forward, copies in every evograd helper and constant it uses, and turns library names into imports. Checked bit-identical to the declared forward for all 63 operators |
| `TASK.md` | The problem statement (§5.2) | `statement.py` |
| `skeleton.py` | The two signatures raising `NotImplementedError`, with the reference as comments | `artifact.render_skeleton` |
| `adbench_primitives.py` | A local `vendor_gemm` so agents can run their module on their own (only if granted) | `artifact.render_primitives_module` |
| `task.json` | Task id, op, workload (dims, dtype), how the case was chosen, forward/backward signatures (argument kinds, shapes, gradient order), tolerances, granted primitives, scoring policy, baseline, evograd revision, `statement_sha256`, `system_message_sha256` | `task.py` |

The task id looks like `llama3_swiglu_mlp.bf16.B2_H2048_I8192_T2048`.
`compare` refuses to mix results from different task ids or statement hashes.

### 5.2 `TASK.md`: the problem statement

The statement has four sections. Each is built from evograd itself, so it
can't drift from what evograd uses or enforces.

1. **Instructions (identical to the evograd agent's).** evograd's OpenEvolve
   system message for this operator, from `evograd.evolve.run.render_config`
   (the function `evograd evolve` uses), copied word for word. It contains the
   operator's meaning, the gradient formulas, which arguments need gradients
   and the constraints, as declared.
2. **What you are given.** This replaces what evograd's seed carries. It
   starts with "There is no working starting program…", so references to "the
   EVOLVE-BLOCK" read as "the module you write". Then:
   - the reference source in a ```` ```python ```` block;
   - the fixed configuration;
   - each argument and output with its shape, dtype and gradient role;
   - the tolerances for each output and gradient.
3. **Deliverable and rules.**
   - The exact signatures to define.
   - The `saved_tensors` contract and the gradient order.
   - Don't write an `autograd.Function` or `nn.Module`; the reserved names to
     avoid.
   - The banned constructs, read from the checker's own tables
     (`pipelines.shared.primitives`).
   - The granted library primitives and how to import them locally.
4. **How it is scored.** The same evaluator for everyone; what counts as
   incorrect; what is timed (forward, backward from saved state, full step,
   saved memory); the baseline and scoring policy.

The text is the same for every agent. Each agent receives it in its own prompt
slot:

| Agent | Slot |
|---|---|
| GEPA | `background` |
| KernelAgent | `problem_description` (+ harness section) |
| raw OpenEvolve | system message |
| GEAK | `TASK.md` file in its op task folder (+ one steering sentence) |
| Claude Code, Codex | their CLI prompt, after one paragraph on the workspace (+ `TASK.md` in the workspace) |
| evograd | doesn't read it; gets section 1 natively, plus its seed |

`python -m evograd.adbench_tools statement TASK_DIR` prints it, and
`_example_TASK.md` here is a real one.

### 5.3 Packaging (`artifact.py`)

Agents produce a bare module; evograd evaluates an artifact. `package`
connects the two:

```
"""adbench-tools artifact … provenance …"""
from __future__ import …            # moved up from the agent's module
import torch, triton, triton.language as tl
# EVOLVE-BLOCK-START
<the agent's whole module>          # every line is under the static rules
# EVOLVE-BLOCK-END
<deployment layer rendered from the declaration>   # autograd.Function, module, DEPLOYMENT_ENTRY, ARTIFACT_CONTRACT
```

Before wrapping, `check_module`:
- requires the module to parse and to define both pair functions at the top
  level;
- rejects block markers inside it;
- rejects defining reserved names;
- removes only `from adbench_primitives import <granted names>` (any other
  use of that module is rejected).

`extract_code` strips a leftover Markdown fence first. The deployment layer is
the same one evograd's own pipelines produce (`render_deployment_layer`).

### 5.4 The evaluator (`evaluate.py`)

| Depth | Runs | Used by |
|---|---|---|
| **full** | `evograd.evolve.evaluator_entry` in a child process, with the environment `run_evolve` sets: scoped declaration, forward override, scoring policy, performance baseline, baseline timing cache, granted primitives. It runs static rules → import → correctness of every output and gradient → a run at the timed shape → timing → `combined_score` | evograd, raw OpenEvolve, GEPA, GEAK, Claude Code, Codex, final score |
| **correctness** | The same up to the timed-shape run, without timing (`_check.py`) | KernelAgent's test, GEAK's and the coding agents' `--correctness` mode |

- `EvalRecord.feedback()` renders the verdict in the format OpenEvolve shows
  its model. Every agent gets this text.
- `agent_score()` is `combined_score` if correct, else 0, for optimizers that
  need a non-negative score.
- The command line (`python -m evograd.adbench_tools evaluate --task T
  [--run-dir R] [--correctness-only] cand.py`) prints a JSON report and exits
  0 only if the candidate is correct. This is how non-Python or out-of-process
  harnesses use it.
- The baseline timing is cached per task (`TASK/.cache/`), keyed by op,
  baseline, GPU, warmup/repetitions, dims and dtype. So every agent's speedup
  is divided by the same baseline measurement.

### 5.5 The run log (`ledger.py`)

Each run directory has:

| File | Content |
|---|---|
| `run.json` | Agent, task id, statement hash, config (budget translation, model, commits), status, wall time, best program |
| `evaluations.jsonl` | One line per evaluation requested: kind, origin, correct, score, program path, seconds, failure type |
| `llm.jsonl` | LLM usage where the agent exposes it (GEPA metadata; Claude Code's session cost and usage for GEAK and Claude Code; Codex's per-turn tokens) |
| `programs/` | Every packaged program evaluated through adbench-tools |

Lines are appended with single `O_APPEND` writes, so parallel workers
(KernelAgent) don't interleave.

### 5.6 The final measurement and comparison (`score.py`)

`python -m evograd.adbench_tools score --task T --run-dir R PROGRAM` (every
driver calls it at the end):

1. **Rebuild** the program from the code inside its block, with fresh wrapper
   code. `fixed_code_intact` records whether the agent had changed anything
   outside the block.
2. **Seed acceptance**: `verify_candidate(seed_acceptance=True)`, the check
   evograd's frontend seed must pass. It checks the deployment entry, every
   output and gradient against the oracle, and the timed shape.
3. **Timing** with the full evaluator at 25 warmup / 100 repetitions.
4. Write `final_program.py` and `final.json`:
   - accepted, correct, `combined_score`;
   - speedups, forward/backward/full-step times, saved and input bytes;
   - primitives used;
   - the run summary;
   - `follow_up` commands for Tier 2 (`evograd tier2-bench`, the operator's
     declared benchmark cases) and Tier 3 (`--patch-set agent:<site>=…`).

`python -m evograd.adbench_tools compare R/*/final.json` writes one Markdown
table, sorted by `combined_score`, with columns:
- accepted, correct, `combined_score`;
- backward and full-step speedup;
- saved/input bytes;
- evaluations, LLM calls, wall time.

---

## 6. Running

```bash
export OPENAI_API_KEY=...
SETUP=1 bash candidates/run_openai.sh llama3_swiglu_mlp bfloat16 30     # first time
STAGES="unit task sanity smoke" bash candidates/run_openai.sh           # checks only
```

Results go to `results/adbench_tools/<op>/`:

| Path | Content |
|---|---|
| `task/` | The pack, including `TASK.md` |
| `sanity_<stamp>/` | Evaluator check without an LLM |
| `smoke_<stamp>/<agent>/` | Smoke runs |
| `runs/<agent>_<stamp>/` | Full runs |
| `comparison_<stamp>.md` | The table |

---

## 7. Open items

- No real GPU + LLM run has completed yet for GEPA, raw OpenEvolve,
  KernelAgent, Codex or Claude Code. The first smoke run is the end-to-end
  test.
- GEAK is parked until there is a way to run it on the same model (e.g. an
  OpenAI-compatible gateway for Claude Code), or until it is reported as a
  separate data point.
- Each comparison uses one task configuration per operator. Tier 2/3 results
  on the final programs are follow-ups, not part of `compare`.
