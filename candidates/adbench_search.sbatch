#!/usr/bin/env bash
# One agent, remaining cases serially; optional arguments: --model MODEL.
#SBATCH --job-name=adbench-search
#SBATCH --account=betw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=adbench-search-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the evograd checkout}"
source candidates/adbench_cluster_env.sh
"$ADBENCH_ENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"; print(torch.cuda.get_device_name())'
"$ADBENCH_ENV/bin/python" candidates/adbench_search.py run --agent "$AGENT" "$@"
