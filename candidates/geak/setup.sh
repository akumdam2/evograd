#!/bin/bash
# Prepare a GEAK v4 working copy that runs on NVIDIA.
#
#   bash candidates/geak/setup.sh [GEAK_SOURCE] [DEST]
#
# GEAK_SOURCE  a GEAK checkout (default: references/GEAK, pinned below)
# DEST         where the working copy goes (default: candidates/geak/.work/GEAK)
#
# GEAK targets AMD GPUs. Its kernel_workflow is portable except for the GPU
# identity probe the Director must run (scripts/gpu_identity.py reads
# rocminfo). The copy gets the NVIDIA probe from candidates/geak/nvidia/; the
# AMD original is kept beside it as gpu_identity.amd.py. Nothing else in GEAK
# is changed: profiling reaches Nsight Compute through GEAK's own generic
# `metrix` hook (PROFILER_PRIORITY=metrix, shim in nvidia/bin), and GPU
# selection is CUDA_VISIBLE_DEVICES around the whole run (run.py sets both).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="${1:-$REPO/references/GEAK}"
DEST="${2:-$HERE/.work/GEAK}"
PINNED=02095f2c8665a7d8741924546d27a30cc19794b3

if [ -d "$SRC/.git" ]; then
    actual="$(git -C "$SRC" rev-parse HEAD)"
    [ "$actual" = "$PINNED" ] || echo "WARNING: GEAK at $actual, adbench-tools was written against $PINNED" >&2
fi
mkdir -p "$DEST"
rsync -a --delete --exclude .git --exclude 'kernel_workflow/exp/' "$SRC/" "$DEST/"
cp "$DEST/scripts/gpu_identity.py" "$DEST/scripts/gpu_identity.amd.py"
cp "$HERE/nvidia/gpu_identity.py" "$DEST/scripts/gpu_identity.py"
echo "$PINNED" > "$DEST/.adbench_tools_geak_commit"
command -v claude >/dev/null || echo "WARNING: the claude CLI is not on PATH (GEAK runs inside Claude Code >= 2.1.177)" >&2
command -v ncu >/dev/null || echo "WARNING: ncu is not on PATH; GEAK's profile step will fall back to benchmark-only" >&2
echo "$DEST"
