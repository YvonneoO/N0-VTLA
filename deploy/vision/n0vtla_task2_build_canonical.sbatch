#!/bin/bash
#SBATCH --job-name=n0vtla_t2_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t2_build_%j.log
#
# Builds Task2's canonical train+val LeRobot v3 datasets from the raw release
# (n0vtla_task2_download_raw.sbatch), on VISION -- unlike Task1, whose raw->canonical
# build happened on lab and only the finished canonical_wetlab_task1_{train,val} was
# ever pulled onto VISION. Same pipeline, same tools (wetlab_smoke_adapter.py via
# build_wetlab_canonical_dataset.py / fit_wetlab_tactile_norm.py), just run here.
#
# task-name=soft_select_bucket_v1 matches robot/manifest.json's task.name for these
# episodes (verified against a sample episode -- NOT the higher-level
# "task2_soft_hard_active_probe_sort" identifier from training_split.json, which is
# a release-level label, not the per-episode manifest field the build script checks).
# --skip-success-label-check because trial.label is null on every episode here (same
# "later releases" pattern as Task1 -- training_split.json/admission_index.json is the
# authoritative admission list instead, and scripts/wetlab_split_task2_v1.json below
# already contains ONLY admitted uuids after translation).
#
# --task-description is passed explicitly this time (Task1's own canonical_wetlab_task1_train
# tasks.jsonl turned out to hold the cap_to_tray SMOKE-TEST default text, "Pick the cap
# and place it in the white container" -- not a tube-rack description -- because no
# --task-description was passed for that build. Flagging, not silently repeating it.)
#
# Submit with: sbatch --dependency=afterok:<download_jobid> deploy/vision/n0vtla_task2_build_canonical.sbatch
set -euo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
RAW_DIR="$DATA_ROOT/wetlab_raw/task2_soft_hard_sorting_20260918"
SPLIT_MANIFEST="scripts/wetlab_split_task2_v1.json"
NORM_PATH="$DATA_ROOT/wetlab_tactile_norm_task2_v1.json"
TASK_NAME="soft_select_bucket_v1"
TASK_DESC="Actively probe the object and sort it into the soft bucket or the hard bucket based on its tactile firmness."

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
  git commit -m "Add Task2 (soft_hard_sorting) train/val episode-UUID split manifest

Translated from task2_soft_hard_sorting_20260918/training_split.json
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
echo "=== [3/4] building canonical_wetlab_task2_train ==="
python scripts/build_wetlab_canonical_dataset.py \
  "$RAW_DIR" \
  "$SPLIT_MANIFEST" \
  "$NORM_PATH" \
  "$DATA_ROOT/canonical_wetlab_task2_train" \
  --which train --workers 16 \
  --task-name "$TASK_NAME" \
  --task-description "$TASK_DESC" \
  --skip-success-label-check

echo
echo "=== [4/4] building canonical_wetlab_task2_val ==="
python scripts/build_wetlab_canonical_dataset.py \
  "$RAW_DIR" \
  "$SPLIT_MANIFEST" \
  "$NORM_PATH" \
  "$DATA_ROOT/canonical_wetlab_task2_val" \
  --which val --workers 16 \
  --task-name "$TASK_NAME" \
  --task-description "$TASK_DESC" \
  --skip-success-label-check

echo
echo "=== TASK2_CANONICAL_BUILD_COMPLETE ==="
