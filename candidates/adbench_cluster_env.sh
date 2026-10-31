# Source from a GPU job after setting AGENT. No installation or GPU allocation.
: "${AGENT:?Set AGENT to openevolve, gepa, geak, or codex}"
case "$AGENT" in openevolve|gepa|geak|codex) ;; *) echo "Unknown AGENT: $AGENT" >&2; return 2 ;; esac
export EVOGRAD
EVOGRAD=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$EVOGRAD"
export ADBENCH_ENV="${ADBENCH_ENV:-$HOME/venvs/adbench-20261028}"
test -x "$ADBENCH_ENV/bin/python"
export PATH="$ADBENCH_ENV/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$EVOGRAD/src"
export ADBENCH_ROOT="/work/hdd/bcjw/wzhan/ADBench/$AGENT"
export ADBENCH_MODEL=gpt-5.5 ADBENCH_BUDGET=10
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CXX=/usr/bin/g++ DISABLE_AUTOUPDATER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/work/hdd/bcjw/wzhan/evograd_verification/hf_home
export EVOGRAD_BENCHMARK_DATA_ROOT=/work/hdd/bcjw/wzhan/evograd_verification/data
export EVOGRAD_LLAMA_HF_CACHE=/work/hdd/bcjw/kramesh/hub
export EVOGRAD_HF_CACHE=/work/hdd/bcjw/wzhan/hf_cache
# The formal runner overrides EVOGRAD_HF_CACHE for non-Qwen3 models.
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/$USER/triton"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/$USER/inductor"
set +x
set -a
source "$HOME/.adbench_keys.txt"
set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing from ~/.adbench_keys.txt}"
# Pin after sourcing credentials, which may also contain model settings.
export ADBENCH_MODEL=gpt-5.5
