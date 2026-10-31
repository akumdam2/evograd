#!/usr/bin/env bash
# Submit from the repository: sbatch candidates/adbench_evaluate.sbatch MODEL AGENT
#SBATCH --job-name=adbench-evaluate
#SBATCH --account=betw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -eo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: sbatch candidates/adbench_evaluate.sbatch MODEL AGENT [evaluate_model.py options]' >&2
  exit 2
fi
MODEL="$1"
AGENT="$2"
shift 2
cd "${SLURM_SUBMIT_DIR:?Submit from the evograd checkout}"

# Match the environment used in the interactive evaluation; no API keys needed.
module load python/anaconda3/2.12.0
eval "$(conda shell.bash hook)"
export ADBENCH_ENV="${ADBENCH_ENV:-$HOME/venvs/adbench-20261028}"
conda activate "$ADBENCH_ENV"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"; print(torch.cuda.get_device_name(), flush=True)'
if python -u candidates/evaluate_model.py "$MODEL" --agent "$AGENT" \
    --submission-root /work/hdd/bcjw/wzhan/ADBench-redeploy \
    --out-root "${ADBENCH_EVAL_ROOT:-$HOME/adbench-evaluation-redeploy-v1}" \
    --deadline-min 100 --tier3-blocks 10 "$@"; then
  exit 0
else
  result=$?
  if [[ "$result" -eq 3 ]]; then
    echo 'Evaluation checkpointed (exit 3). Resubmit the same sbatch command to resume.'
  fi
  exit "$result"
fi
