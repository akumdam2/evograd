# Handoff: benchmark v1

Written for whoever picks this up on another machine. The work below is
code-complete and CPU-verified; every remaining item needs a GPU, which is why
it moved.

Branch: `benchmark-v1`. Base: `origin/main` plus two pre-existing commits that
were never pushed (`e442610` fair-bench protocol, `4c29270` codegen test dtype).

## Why this branch exists

The NCSA Delta allocation (`bcjw-delta-gpu`) is exhausted — balance -969 hours
against 51,880 deposited, and jobs sit in `PENDING` with
`Reason=QOSGrpBillingMinutes`. Nothing here is blocked on code.

## Get running

```bash
git fetch origin && git checkout benchmark-v1

# The environment needs: torch, triton, and liger-kernel for the baselines.
# On Delta this was the `openev` conda env (liger-kernel 0.8.0, openevolve 0.3.2).
pip install -e ".[gpu,baselines]"          # or recreate the env

cd <repo> && PYTHONPATH=src python -m unittest discover tests
```

Expect **170 tests**. Two known environment-dependent failures, neither caused
by this work:

- 3 errors from `test_program_archive` / `test_codegen_and_prompts` if
  `openevolve` is not installed.
- 1 error from `test_geak_nvidia_profile` if `python-dotenv` is not installed.

The package is not `pip install -e`'d in the original setup; everything runs via
`PYTHONPATH=src`.

## What this branch changed

Read `docs/BENCHMARK.md` first — it is the specification, written to be
published. This file only covers what a successor needs that the spec does not
say.

25 operators across three levels (18 primitive / 5 fused / 2 block), 191 timed
configurations. The four substantive changes:

1. **Shapes are derived, not typed.** `src/evograd/opdecl/models.py` holds frozen
   Llama-3-8B and AlphaFold3 configurations; declarations call
   `model_workloads(config, component, free_sweep, dtypes)` and every workload
   carries a `Provenance`. `tests/test_provenance.py` re-derives all 172
   `hf_config` workloads and fails on drift.
2. **Tasks carry a level and a family.** Both are validated. Provenance is
   required for any operator that declares a level.
3. **`reference_dtype`.** Level-3 tasks compute the correctness reference in
   float32 against a bfloat16 candidate.
4. **`evograd suite`** produces the cross-operator report — the thing that makes
   this a benchmark rather than a pile of kernels.

## Things that will bite you

These are all decisions with a reason. Changing them without reading the reason
will produce numbers that look fine and are wrong.

**Never let `reference_dtype` reach the timing path.** `bench/harness.py` times
the eager PyTorch baseline by calling `oracle()`. If that promotes to float32
while the candidate runs bfloat16, the "baseline" is a slower computation and
every speedup is inflated. The call site passes `use_reference_dtype=False`
explicitly; keep it.

**The suite reads full-step speedup only.** `speedup_vs_baseline_backward`
compares asymmetric things — the eager baseline's backward timing runs the
oracle, which computes the forward *and* backward, while the candidate's
backward is timed from pre-saved state. For a level-3 block that inflates the
ratio by roughly half. `bench/suite.py` names the key once, in
`FULL_STEP_SPEEDUP_KEY`, and `tests/test_suite_report.py` plants a decoy
backward value ten times larger so a regression fails loudly.

**Speedups pool per family before pooling across families.** Fourteen of the
twenty-five operators are norms, activations and losses. A flat mean would let
declaration count decide the headline. There is a test for this too.

**`Inactive` tensors default to zeros.** `opdecl/inputs.py` fills them with
`torch.zeros`. For RoPE's `cos`/`sin` that turns the operator into the zero map
— nothing crashes, correctness still passes because the oracle sees the same
zeros, and the benchmark silently measures a different function. Any operator
with a semantically meaningful `Inactive` tensor needs a `make_inputs` hook.
`rope`, `llama3_decoder_layer`, `af3_single_repr_block` and `evoattention` all
have one.

**A declared dim that appears in no tensor shape cannot be recovered.**
`dispatch._route` rebuilds the dim dict from tensor shapes at deploy time. That
is why the Llama block declares `q_out`/`kv_out` rather than
`n_heads`/`n_kv_heads`, and why the AF3 block has no `D` — the references divide
to recover them.

**Shape strings take no arithmetic.** `_validate_shape` accepts declared dims or
integer literals only. A fused QKV weight has to enter as its own symbolic dim.

**Liger's RoPE rotates q and k jointly.** The adapter passes a one-head slice as
`k` because the declaration has one output. Read
`src/evograd/ops/rope/liger.py` before changing it — the overhead is bounded at
about `1/n_heads` and the comment explains why the obvious alternative (passing
the same tensor twice) would make Liger look 2× slower than it is.

**FLCE computes its gradients in the forward.** Its backward only rescales them,
and in real training the upstream gradient is exactly 1.0, so Liger skips the
rescale entirely. A backward-only timing of this operator is meaningless. The
declaration sets `dloss = 1.0` deliberately.

## What is verified, and what is not

Verified on GPU (an A100-40GB, before the allocation ran out):

- The full test suite, on `openev` with liger 0.8.0 and torch 2.11.
- `rmsnorm`'s new Liger adapter, against the autograd oracle across all seven
  correctness workloads (fp32/fp16/bf16).

Verified on CPU only:

- 170 tests.
- Both level-3 blocks: complete forward and backward, gradient counts and dtypes
  correct, no non-finite values.
- `rope` end-to-end through `evograd verify` with a hand-written candidate, all
  9 correctness workloads.
- The Llama block's bfloat16 tolerance against a measured noise floor: the
  tightest margin over the bf16-vs-fp32 discrepancy is 2.1×. Worth re-measuring
  at real dimensions on a GPU; the CPU measurement used the scaled-down
  correctness shapes.
- Independent mathematical checks rather than self-consistency: RoPE preserves
  each head vector's norm (1.1e-7 relative), and FLCE's `dweight` sums to zero
  along the vocabulary axis (8e-8).

**Not verified — do these first on the new machine:**

1. The `rope` and `fused_linear_cross_entropy` Liger adapters have only been
   signature-checked against liger 0.8.0. They have never executed. The
   signatures were read from source, and one earlier adapter was written from
   memory and got a parameter wrong, so treat these as unproven:

   ```bash
   PYTHONPATH=src python -c "
   from evograd.opdecl.baselines import verify_performance_baseline
   from evograd.ops import get_op
   for name in ('rope', 'fused_linear_cross_entropy'):
       verify_performance_baseline(get_op(name), 'liger'); print(name, 'ok')"
   ```

2. Both level-3 blocks have never run on a GPU at their benchmark dimensions.
   Watch memory: the reference builds an autograd graph for the whole layer at
   float32. The timed grid stops at 4096 tokens for that reason and 8192 lives
   in untimed `coverage`. If the machine has more than 40 GB, the timed ceiling
   can rise.

3. No performance number in this repository has been reproduced since the
   migration off the OpenEvolve fork. `docs/MIGRATION_AUDIT.md` says so. Run
   `scripts/gpu_smoke.py --oracle-only` and `scripts/gpu_parity.py` before
   trusting any speedup.

## Known gaps, deliberately left open

**Level-3 baselines are not wired.** Both blocks currently get only
`pytorch_autograd` and the built-in `torch_compile`. They should also have:

- `llama3_decoder_layer`: a Liger-patched layer built from `LigerRMSNorm`
  (`casting_mode="llama"`, `offset=0.0`), `liger_rotary_pos_emb` and
  `LigerSwiGLUMLP` — i.e. what `apply_liger_kernel_to_llama` installs.
- `af3_single_repr_block`: MegaFold's kernels.

Follow `ops/_common.py::make_pair_baseline`. The pattern that keeps it fair is
to build the autograd graph in the untimed forward and time only
`torch.autograd.grad` — `bench/fair.py::pytorch_autograd_provider` does exactly
that.

**No DeepSpeed baseline for `evoattention`.** MegaFold compares against
`DS4Sci_EvoformerAttention`; deepspeed was not installed, and shipping a
baseline that has never executed is worse than declaring none.

**`conv2d` has no model provenance.** Its contract is stride 1 / padding 0,
while every real convolutional network pads. Tying it to a real backbone
(RTDetrV2's ResNet-101 is the closest match among FastKernels' level-3 models)
requires extending the declaration with stride and padding, which touches
Pipeline B's handwritten Triton lowerings for forward, dX, dWeight and dBias.
There is a second obstacle: a ResNet bottleneck block contains BatchNorm, whose
running statistics mutate during training — the fair-bench protocol rejects
input mutation, so such a block would have to declare training-mode statistics
only and say that the running-stat update is out of scope.

## Running on a new cluster

The old one used Slurm:

```bash
srun -A <account> --time=1:00:00 --nodes=1 --ntasks-per-node=1 \
     --partition=<gpu-partition> --gpus=1 --cpus-per-task=8 --mem=32g --pty /bin/bash
conda activate openev
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:$LD_LIBRARY_PATH
```

Long jobs and node allocations were kept in a tmux session named `code`,
window 0, so the work is visible on attach. `OPENAI_API_KEY` is only needed for
the LLM-backed seed pipelines (A and C) and evolution — none of the benchmark
work above requires it.
