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
so they are separate data points and are left out of `run_openai.sh`. They also
retain `legacy_native` budgets rather than the OpenAI agents' API-request cap.

Claude Code and Codex are the general-purpose coding agents: no kernel-specific
loop, only a shell, a file editor and the evaluator. They show what an
off-the-shelf agent does with the same task statement. Codex shares the metered
OpenAI model budget; Claude Code's model and budget are reported separately.

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
   code inside its block, so edits outside it are discarded. Every submission
   goes through `evograd-evaluation-v3` (§7): the applicable real-payload gates,
   formal T2 and model-scope T3. `score`/`compare` are separate diagnostics (§5.6).
6. **Same model and API-request cap for the OpenAI agents.**
   - One `--model` / `--api-base` (an OpenAI-compatible endpoint) is used for
     every metered model request; GEAK and Claude Code are separate points.
   - `--budget N` caps model requests per agent/case at a shared HTTP proxy (§3).
   - API requests, evaluator calls and agent-reported usage events have separate
     ledgers; native iteration limits are additional stopping conditions.

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
  Nothing they are given names the repository, but they run as the same
  user, so a filesystem or process search could still find it. Whether they
  edited `evaluate.py` is recorded. Claude Code runs on a Claude model.
- **Library access.** evograd's frontend preserves what Inductor hands to a
  vendor library: matrix multiplications and convolutions (cuBLAS/cuDNN) and
  fused attention (cuDNN SDPA). So with `--library auto` (the default), every
  agent may call a trusted `vendor_gemm` primitive whenever the reference
  contains a matrix multiplication, and the `vendor_sdpa_forward` /
  `vendor_sdpa_backward` pair whenever it calls
  `scaled_dot_product_attention`. Every other contraction must be written in
  Triton. A case whose reference is one preserved call (a bare projection, a
  bare attention) leaves nothing to evolve: its candidate equals the baseline.

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
| Input | A workspace with `TASK.md`, `candidate.py` (the skeleton), `adbench_primitives.py` (if granted) and an `evaluate.py` it may not edit. The prompt (`prompt.md` in the run) describes the workspace, the applicable budget and the deliverable, followed by the same `TASK.md` verbatim |
| Loop | The agent's own: `claude -p` (stream-JSON output) or `codex exec --json`, headless, permissions bypassed, until it stops or `--time-limit` (default 6 h) |
| Isolation | The workspace is a new temporary directory, copied to `<run>/workspace` afterwards (`--workspace` overrides). Nothing the agent is given names the repository, the run or the task pack: `evaluate.py` holds only a socket address (Linux abstract namespace, this user only); the driver's evaluation server owns evaluator limits and records and replaces those paths in its feedback with `<evograd>`, `<run>` and `<task>`; and environment variables naming them are dropped (`PWD`, `SLURM_SUBMIT_DIR`) or pruned (`PYTHONPATH`). The agent runs as the same user, so a filesystem or process search could still find the repository; only a container would close that. User settings are not loaded: Claude Code gets `--setting-sources project,local --strict-mcp-config`, Codex gets `--ignore-user-config`. When the session ends, the CLI's whole process group is killed, so nothing it started competes with the final measurement |
| Model | Claude Code: `--claude-model`. Codex: `--model` through a custom `adbench` HTTP Responses provider pointing to the shared budget proxy, with WebSockets disabled and `stream_max_retries=0`. Other request retries still consume budget. The CLI receives placeholder credentials (`adbench-local`); the proxy holds the upstream key and forwards to `--api-base`. `--effort` sets either agent's reasoning effort |
| Feedback | `evaluate.py` sends `candidate.py` or a given path to the driver's evaluation server, which runs our evaluator (full, or `--correctness`), one call at a time. It prints the evaluator's feedback, `CORRECT`/`INCORRECT` and any remaining evaluator budget. Codex has no evaluator-call cap unless `--eval-budget` is set |
| Code from reply | None to extract: the agent writes `candidate.py` with its file tools |
| Output | `candidate.py` if `evaluate.py` passed exactly that module (the agent's choice). Otherwise, the best correct module it evaluated (full evaluations by `combined_score` first, then the latest). Otherwise, `candidate.py` as it is, unless it is still the skeleton. `run.json` records which rule applied (`deliverable`) |

### 2.7 Shared requirements

- An NVIDIA GPU with CUDA, Triton and `evograd[gpu,llm]` installed.
- `OPENAI_API_KEY`. Optionally `ADBENCH_MODEL` (default `gpt-5.6-sol`) and
  `ADBENCH_API_BASE` (default `https://api.openai.com/v1`). Set
  `ADBENCH_MODEL=gpt-5.5` for this comparison; the driver default is unchanged.
  Codex's upstream endpoint must serve `/responses` and, when used,
  `/responses/compact`.
- `references/` is gitignored. The KernelAgent (and GEAK) checkouts must exist
  on the machine; `SETUP=1 bash candidates/run_openai.sh` creates them.

---

## 3. Budgets

For evograd, raw OpenEvolve, GEPA, KernelAgent and Codex, `--budget N` is a hard
cap on admitted **model API requests per agent and benchmark case**. The default
is `ADBENCH_BUDGET`, or **10**. `_common.start` routes those agents through
`LLMBudgetProxy`, which reserves a request in `api_calls.jsonl` before forwarding
it. Initial generation, reflection, retries, failed upstream calls and context
compaction all count. Parallel workers share one cap. Request N+1 is refused;
the proxy does not retry upstream or follow redirects.

The proxy meters `/chat/completions`, `/completions`, `/responses` and
`/responses/compact`. Model metadata reads do not consume the cap. Agents and
their subprocesses receive the metered endpoint and placeholder credentials;
the upstream key remains in the proxy. This routing is not a filesystem or
network sandbox for an agent running as the same user.

Native stopping limits remain in place:

| Agent | Request cap | Additional stopping control |
|---|---|---|
| evograd | N | OpenEvolve `iterations = N`; frontend seed uses no LLM |
| raw OpenEvolve | N | OpenEvolve `iterations = N` |
| GEPA | N | `max_evals = N + 1`; first candidate generation also consumes the API cap |
| KernelAgent | N across generation and optimization | 4 generation workers × ⌈N/8⌉ rounds, stopping on first pass; max(1, ⌊N/8⌋) optimization rounds |
| Codex | N | Session time limit; optional independent `--eval-budget`; no evaluator-call cap by default |
| GEAK | Unmetered (`legacy_native`) | `max(2, N // 5)` optimization directions unless `--geak-budget` overrides |
| Claude Code | Unmetered (`legacy_native`) | N calls of `evaluate.py` by default, or `--eval-budget` if set |

These native limits can stop a run before N model requests. They do not grant
extra requests. GEAK and Claude Code do not have an equal API-request budget
and must be reported separately.

`evaluations.jsonl` counts evaluator calls independently; `llm.jsonl` stores
token/session usage events. `llm_calls_recorded` comes from `api_calls.jsonl`
for metered agents, is 0 for the frontend-only seed, and is `null`/`unmetered`
when request count is unknown. Usage-event counts cannot replace API counts.

Other limits:

| Setting | Value |
|---|---|
| Diagnostic `score` measurement | 25 warmup / 100 repetitions (in-loop runs use the evaluator's defaults) |
| Formal v3 T2 measurement | every recorded call of the case in the acceptance order-303 step; per call warmup 10 / repetitions 50 / 3 fair-timing blocks (§7) |
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
| `ledger.py` | **Run log** (§5.5): `run.json`, `evaluations.jsonl`, `llm.jsonl`, `api_calls.jsonl` |
| `llm_budget.py` | Shared HTTP proxy that reserves and enforces model API requests independently of evaluator calls |
| `score.py` | **Final measurement** and `compare` (§5.6) |
| `_example_TASK.md` | A generated `TASK.md` for `llama3_swiglu_mlp`, bf16, B=2 T=2048 H=2048 I=8192, for reading |

`task --model M --case-id TASK/COMPONENT` builds the pack of one of a verified
model's *observed cases* (§7) instead of an operator's declared configuration.

### 4.2 `candidates/`: one folder per agent

| Path | Role |
|---|---|
| `_common.py` | Shared flags, request-budget proxy and ledger setup, diagnostic scoring, importing OpenEvolve's program archive into the ledger |
| `bench.py` | **The benchmark run** (§7): task packs per observed case, the agent on each, `evograd evaluate`, the per-level summary |
| `seed/run.py` | evograd's frontend alone (the unified seed, no LLM): the no-LLM pipeline check and the "frontend only" reference point |
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

Tests: `PYTHONPATH=src python -m pytest tests/adbench_tools -q` (CPU-only).

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
| `run.json` | Agent, task id, statement hash, config (budget limit and unit, model, commits), status, wall time, best program, final request-budget usage |
| `evaluations.jsonl` | One line per evaluation requested: kind, origin, correct, score, program path, seconds, failure type |
| `api_calls.jsonl` | One line per admitted model API request for metered agents; request metadata only, no credentials or prompt/response content |
| `llm.jsonl` | LLM usage where the agent exposes it (GEPA metadata; Claude Code's session cost and usage for GEAK and Claude Code; Codex's per-turn tokens) |
| `programs/` | Every packaged program evaluated through adbench-tools |

Evaluation and usage lines are appended with single `O_APPEND` writes, so
parallel workers do not interleave. API reservations use a lock and durable
append before forwarding, so workers share the same limit.

### 5.6 The final measurement and comparison (`score.py`)

**Diagnostic.** `score` and `compare` measure one configuration with seed
acceptance and Tier 1 only; the benchmark's judgement is `evograd evaluate`
(§7), which `candidates/bench.py` runs and the drivers skip with `--no-score`.

`python -m evograd.adbench_tools score --task T --run-dir R PROGRAM` (every
driver calls it at the end unless told `--no-score`):

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
export ADBENCH_MODEL=gpt-5.5
export ADBENCH_BUDGET=10
SETUP=1 bash candidates/run_openai.sh llama3_swiglu_mlp bfloat16 10     # first time
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

## 7. The benchmark run

The comparison the paper reports is not `score`/`compare` but the benchmark's
evaluation protocol, `evograd evaluate` (`src/evograd/evaluation/protocol.py`),
run by `candidates/bench.py` on every agent in the same way:

```bash
export ADBENCH_ROOT=...            # default results/adbench; results go to $ADBENCH_ROOT/<model>
export ADBENCH_MODEL=gpt-5.5
python candidates/bench.py tasks     --model qwen3_0_6b --case-id qwen3_swiglu_mlp/swiglu_mlp rmsnorm/rmsnorm ...
python candidates/bench.py run       --model qwen3_0_6b --agent gepa --budget 10
python candidates/bench.py evaluate  --model qwen3_0_6b --agent gepa --corpus CORPUS/manifest.json --deadline-min 50
python candidates/bench.py summarize --model qwen3_0_6b --agent seed evograd openevolve gepa kernelagent
```

1. **Tasks are a model's observed cases.** A case is one configuration a
   verified model trains at (`evograd.verification.integrations.base.observed_cases`),
   and its task pack carries the model's own reference computation
   (`Model.reference_op`) at that configuration: what the agent optimizes is
   what the gates judge. The agent still sees kernel-scope information only
   (the forward's source, shapes, dtypes, tolerances); nothing about the model.
2. **The agent runs its own loop on each pack**, as in §2, with `--no-score`.
   Its best program is rebuilt from its evolve block into `submission.py`
   (`fixed_code_intact` is recorded).
3. **Every submission goes through `evograd-evaluation-v3`**: the static rules
   (`check_source`; a violation is neither verified nor timed), then the gates
   on the model's real training payloads -- Level 1: Gate 1 and operator
   Gate 2; Level 2: also model Gate 2 and Gate 3 alone at its site. Every
   executable L1/L2 candidate gets formal T2; each executable L2 candidate also
   gets single-site T3 against eager. Each agent's positive L2 candidates are
   combined: positive means accepted and a successful single-site T3 speedup
   whose 95% interval lower bound is greater than 1. That combination gets
   model Gate 2 and Gate 3 (L3), then T3 against eager and whole-model compile.
   Rejected candidates' timings remain diagnostic and do not enter aggregates.
4. **`summarize`** reports, per agent and level, the funnel (submitted,
   violation, executable, Gate 1, accepted), correctness coverage
   (accepted/observed) and T2 geometric-mean speedups over accepted configurations
   against eager and compile. Each baseline separately records how many
   configurations were timed. L2 single-site T3 is retained per model/case,
   including all gate statuses, runtime, speedup and interval, `positive` and
   `selected_for_combination`; these speeds are not averaged into a model result.
   L3 reports the measured combination and the cost ledgers distinguish API
   requests, evaluator calls and wall time. The L3 interval is against eager;
   its compile-relative speedup is a separate point estimate.

Formal T2 measures the artifact's actual deployment forward followed by
`torch.autograd.grad`, without an optimizer or `.grad` accumulation. Eager,
compile and declared library baselines all use that autograd scope and the
existing fair timer (warmup 10, repetitions 50, 3 blocks) on the same inputs:
every recorded call of the case in the acceptance order-303 training step of
the verification corpus, each provider on its own layout-exact copy, with the
speedup the ratio of per-call medians summed over the step. Real-payload gates
determine correctness separately; the standalone `tier2-bench` declared-input
gate and timer are not substituted into this protocol. T1 is search feedback
or an independent pair diagnostic, not the formal eager/compile result.

Gate 3 controls and Gate 2 calibrations live in `$ADBENCH_ROOT/<model>/state`,
shared by every agent; `evaluate` is resumable (`--deadline-min`, exit 3).
The request cap applies to each independent agent/case run, not to evaluation.
`bench.py run` refuses to reuse a metered agent's completed run with an old
budget unit or different limit, and refuses to silently delete an unfinished
run and reset its API ledger. Use a fresh `--root`, or `--redo` to replace the
run as a new trial.
Direct drivers likewise require a fresh `--run-dir`. Old v1 and v2 evaluations
must be rerun with `evaluate`; `summarize` rejects them. Renaming `tier1` fields
to `tier2` does not make them v3 measurements, and v2 timed generated inputs.

## 8. Open items

- No real GPU + LLM run has completed yet for GEPA, raw OpenEvolve or
  KernelAgent. The first smoke run is the end-to-end test. Codex and Claude
  Code each completed one trial (budget 5, `llama3_swiglu_mlp` bf16).
- In its trial, Claude Code followed the repository path `evaluate.py` then
  embedded and read evograd's scoring and timing source. The path is now
  hidden, but an agent running as the same user can still search for the
  repository. Running the coding agents in a container would close that.
- GEAK is parked until there is a way to run it on the same model (e.g. an
  OpenAI-compatible gateway for Claude Code), or until it is reported as a
  separate data point.
- The diagnostic `compare` uses one task configuration per operator. Its
  standalone Tier 2/3 follow-ups remain separate; the formal benchmark runs
  observed cases and includes T2/T3 through `bench.py evaluate` (§7).
