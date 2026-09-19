#!/bin/bash
#SBATCH --job-name=n0vtla_t3_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t3_build_%j.log
#
# Builds Task3's canonical train+val LeRobot v3 datasets from the raw release
# (n0vtla_task3_download_raw.sbatch), on VISION -- unlike Task1, whose raw->canonical
# build happened on lab and only the finished canonical_wetlab_task1_{train,val} was
# ever pulled onto VISION. Same pipeline, same tools (wetlab_smoke_adapter.py via
# build_wetlab_canonical_dataset.py / fit_wetlab_tactile_norm.py), just run here.
#
# TASK_NAME / TASK_DESC are TODO placeholders (guarded above): fill from Task3's robot/manifest.json
# task.name and dataset card. Verify the split schema (translate_smoketestv2_split.py KNOWN_SCHEMAS)
# and whether --skip-success-label-check is still needed (trial.label) before submitting.
#
# Submit with: sbatch --dependency=afterok:<download_jobid> deploy/vision/n0vtla_task3_build_canonical.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
# Neither `source activate ENVPATH` nor `conda activate` (after sourcing conda.sh) actually
# switched the interpreter on the compute nodes this hit (dgx001/dgx002) -- both exited 0 but
# `which python` still resolved to base Miniforge's python3.12, not the env's python3.11
# (confirmed via diagnostics: numpy genuinely installed in the env, "No module named numpy"
# only because the wrong interpreter was running). Root cause not fully pinned down (conda's
# shell-function activation is apparently unreliable in this non-interactive sbatch context on
# some nodes), so sidestep it entirely: prepend the env's bin/ to PATH directly instead of
# relying on any conda activation machinery.
N0VTLA_ENV=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$N0VTLA_ENV/bin:$PATH"

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

echo "=== env diagnostics (numpy import failed on dgx001/dgx002 in previous attempts) ==="
hostname
which python
python -c "import sys; print('executable:', sys.executable); print('path:'); [print(' ', p) for p in sys.path]"
python -c "import numpy; print('numpy OK:', numpy.__file__)"

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
RAW_DIR="$DATA_ROOT/wetlab_raw/task3_whiteboard_erasing_20260918"
SPLIT_MANIFEST="scripts/wetlab_split_task3_v1.json"
NORM_PATH="$DATA_ROOT/wetlab_tactile_norm_task3_v1.json"
TASK_NAME="TODO_FROM_MANIFEST"   # must equal robot/manifest.json task.name; verify on VISION before submitting
TASK_DESC="TODO_FROM_DATASET_CARD"

if [[ "$TASK_NAME" == TODO* || "$TASK_DESC" == TODO* ]]; then echo "ERROR: fill TASK_NAME/TASK_DESC first" >&2; exit 2; fi

echo "=== [1/4] translating training_split.json -> $SPLIT_MANIFEST ==="
if [[ -f "$SPLIT_MANIFEST" ]]; then
  echo "$SPLIT_MANIFEST already exists, skipping translation"
else
  python scripts/translate_smoketestv2_split.py \
    "$RAW_DIR/training_split.json" \
    "$SPLIT_MANIFEST"
  echo "=== committing/pushing the new split manifest (data file, generated here since the" \
       "raw training_split.json this translates only lives on VISION) ==="
  git config user.name "qqyang"
  git config user.email "yqq718@outlook.com"
  git add "$SPLIT_MANIFEST"
  git commit -m "Add Task3 (soft_hard_sorting) train/val episode-UUID split manifest

Translated from task3_whiteboard_erasing_20260918/training_split.json
(schema wetlab_episode_window_split_v1) via translate_smoketestv2_split.py.
Generated on VISION since the raw file this translates only lives there.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
  git push origin main
fi

echo
echo "=== [2/4] fitting shared tactile normalization (train split only) ==="
if [[ -f "$NORM_PATH" ]]; then
  echo "$NORM_PATH already exists, skipping fit"
else
  python scripts/fit_wetlab_tactile_norm.py \
    "$RAW_DIR" \
    "$SPLIT_MANIFEST" \
    "$NORM_PATH" \
    --task-name "$TASK_NAME" \
    --skip-success-label-check
fi

echo
echo "=== [3/4] building canonical_wetlab_task3_train ==="
python scripts/build_wetlab_canonical_dataset.py \
  "$RAW_DIR" \
  "$SPLIT_MANIFEST" \
  "$NORM_PATH" \
  "$DATA_ROOT/canonical_wetlab_task3_train" \
  --which train --workers 16 \
  --task-name "$TASK_NAME" \
  --task-description "$TASK_DESC" \
  --skip-success-label-check

echo
echo "=== [4/4] building canonical_wetlab_task3_val ==="
python scripts/build_wetlab_canonical_dataset.py \
  "$RAW_DIR" \
  "$SPLIT_MANIFEST" \
  "$NORM_PATH" \
  "$DATA_ROOT/canonical_wetlab_task3_val" \
  --which val --workers 16 \
  --task-name "$TASK_NAME" \
  --task-description "$TASK_DESC" \
  --skip-success-label-check

echo
echo "=== TASK3_CANONICAL_BUILD_COMPLETE ==="
