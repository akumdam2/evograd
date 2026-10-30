# Evolved programs: ADBench search trial

The programs each agent submitted in the ADBench search trial, one per case, byte for byte as the run
packaged them (the agent's module in the EVOLVE-BLOCK, evograd's generated deployment layer after it).
"Submitted" means the search's own correctness feedback (Gate 1 on calibration payloads) accepted it;
the final gates (`bench.py evaluate`: Gate 2, Gate 3, speedups) are reported separately.

| agent | cases | programs | site | GPU the search timed on |
|---|---|---|---|---|
| [evograd](evograd/) | 58 | 58 | iu_a100 | NVIDIA A100-SXM4-40GB (Indiana University Jetstream2) |
| [kernelagent](kernelagent/) | 58 | 50 | iu_a100 | NVIDIA A100-SXM4-40GB (Indiana University Jetstream2) |

Each agent folder has a README (code, model, command) and a `manifest.csv` (one row per case).
`patches/<site>/` holds the local patches that site's runs used. Programs searched on different GPUs
were tuned against different timing; compare them only after evaluating them on the same GPU.

To evaluate one agent's programs for one model (on a tree with that site's patches):

    python evolved_programs/eval_command.py AGENT MODEL --adbench /work/hdd/bcjw/wzhan/ADBench --out OUT_ROOT

prints the formal `evograd.evaluation.protocol` command for them.
