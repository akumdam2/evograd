# kernelagent: ADBench search programs (iu_a100)

50 programs for 58 cases; `manifest.csv` has every case, with the run's status, the
program's sha256 and, for a case without a program, the reason found in its run.

- GPU: NVIDIA A100-SXM4-40GB (Indiana University Jetstream2)
- Code: evograd 3930220, task packs 8cff3b9, plus the patches in patches/iu_a100
- Model and budget: gpt-5.5, 10 model requests per case (metered by a proxy; retries count), API default reasoning effort; three searches at a time, GPU work serialized by a file lock.
- Command: `bench.py run --budget 10 --time-limit 6000 -- --gpu-name "NVIDIA A100 SXM4 40GB" --test-timeout 3600` (1200 for the runs of the first kept session); KernelAgent e064717, its code unchanged; its requests' own reasoning effort dropped and output caps raised to at least 32000 tokens (`request_settings`).
