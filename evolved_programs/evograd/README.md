# evograd: ADBench search programs (iu_a100)

58 programs for 58 cases; `manifest.csv` has every case, with the run's status, the
program's sha256 and, for a case without a program, the reason found in its run.

- GPU: NVIDIA A100-SXM4-40GB (Indiana University Jetstream2)
- Code: evograd 3930220, task packs 8cff3b9, plus the patches in patches/iu_a100
- Model and budget: gpt-5.5, 10 model requests per case (metered by a proxy; retries count), API default reasoning effort; three searches at a time, GPU work serialized by a file lock.
- Command: `bench.py run --budget 10 --time-limit 6000 -- --frontend unified --fallback-a` (seed from the unified frontend, frontend A when unified gives no accepted seed).
