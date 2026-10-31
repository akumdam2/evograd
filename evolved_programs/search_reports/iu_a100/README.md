# Search reports (iu_a100)

The records of the two searches on the IU A100 that produced [`../../evograd/`](../../evograd/) and
[`../../kernelagent/`](../../kernelagent/), copied unchanged from the instance on 2026-10-30.

| file | what |
|---|---|
| `evograd_summary.md` | evograd search: per-model counts (submissions, seed frontend, fallbacks to A, failures), requests and wall time; why each fallback happened; the seed fixes; time spent waiting on the API and the GPU lock |
| `kernelagent_summary.md` | KernelAgent search: per-model counts, the 8 cases without a submission and why, the harness problems found and fixed during its runs, cost and time |
| `evograd_NOTES.md`, `kernelagent_NOTES.md` | each search's running log: code, patches and the reason for each, command, LLM and GPU setup, and every fallback, failure, superseded run and rerun in time order |

Reading them here:

- `$T` is `/work/hdd/bcjw/wzhan/adbench_trial` on the IU instance. `$T/patches/<name>.patch` and
  `patches/<name>.patch` are in [`../../patches/iu_a100/`](../../patches/iu_a100/).
- `<model>/superseded/`, `<model>/aborted/`, `runs/...`, `$T/state/gpu.lock.log` and `$T/tools/...`
  are the searches' full outputs and scripts on the instance's data volume; they are not in this
  repository.
- `kernelagent_summary.md` says the final gates had not been run; that was true when it was written.
  The six models IU can hold have since been evaluated:
  [`../../../evaluation_results/iu_a100/`](../../../evaluation_results/iu_a100/).
