# Evolved `fused_add_rms_norm`

`triton_a2fcdc13.py` is one selected Level-2 candidate, preserved byte-for-byte
as it was evaluated. It is checked in as evidence for review; nothing in
`src/` imports it, and it is not part of a benchmark's measured path.

    sha256  a2fcdc137e51e9abff90eb2fdd474024b51457aaa63a1298ad55233c46d67bfd

Verify with `sha256sum triton_a2fcdc13.py`. Do not reformat the generated file:
the hash is how the evaluation record identifies the exact program.

## What it contains

The artifact implements Qwen3-0.6B's fused residual add plus RMSNorm contract.
It includes evolved Triton forward and backward kernels, the two-output
`(out, summed)` interface, an `evograd-artifact/1` manifest, a static
`torch.autograd.Function`, and the deployment callable used by both Tier 2 and
Tier 3.

| | |
|---|---|
| Pipeline | A (AtenIR-grounded synthesis), followed by 10 evolution iterations |
| Operator | `fused_add_rms_norm` |
| Outputs | `out`, `summed` |
| Gradients | `dx`, `dr`, `dweight`; independent `dout` and `dsummed` |
| Deployment | `fused_add_rms_norm_deployment` |
| Artifact contract | `evograd-artifact/1` |

## Evaluation status

The program passed the declared Tier-1 and Tier-2 output and gradient checks,
including both output paths. It was then patched into Qwen3-0.6B at the
`residual_rmsnorm` site. It passed site preflight, repeated-call purity, live
boundary validation, and finiteness, but the provisional one-step Tier-3
numerical policy rejected its model-level drift.

Under the patch-set-matched shadow policy, its logits relative L2 was about
`9.27e-3`, versus a `6.86e-3` threshold (about 1.35x). Liger passed the same
shadow policy. This is useful diagnostic evidence, but it is **not** proof that
the candidate changes long-horizon training convergence: no 1,000-step training
comparison has been run, and the Tier-3 acceptance policy remains under review.

## Provenance

Selected program from:

    /u/wzhan/.cache/evograd-mF/pilot2/20260903-185752/phase3/best_program.py

The source run used an NVIDIA GH200 120GB system. Performance and numerical
results must be remeasured before making claims on another environment.
