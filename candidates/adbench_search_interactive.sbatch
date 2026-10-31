#!/usr/bin/env bash
# Same serial search worker as the batch queue, with an interactive allocation.
#SBATCH --job-name=adbench-search
#SBATCH --account=betw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=adbench-search-%j.out

set -euo pipefail
# Running the batch script with bash reuses its worker, not its SBATCH directives.
exec bash "${SLURM_SUBMIT_DIR:?Submit from the evograd checkout}/candidates/adbench_search.sbatch" "$@"
