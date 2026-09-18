#!/bin/bash
#SBATCH --job-name=n0vtla_watch_upload_10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_watch_upload_10k_%j.log
#
# Watcher, not a completion-dependent job: runs concurrently with the two still-training
# jobs (task1_posttrain_ours_vision8gpu, task1_posttrain_baseline_vision8gpu) and uploads
# ONLY the step-10000 checkpoint of each the moment it lands on disk, for an early inference
# rollout -- doesn't wait for the full 20000-step runs to finish. CPU-only, polls the local
# filesystem every 120s (cheap; this is the watcher's own allocated compute node, not the
# shared login node or a tmux pane, so this loop is not the polling pattern the VISION rules
# warn about). 4h budget: at ~54 steps/min both runs need roughly ~2h to reach step 10000
# from where they were when this was written, so this leaves ~2x margin.
#
# train_pytorch.py (see save_checkpoint()) writes to tmp_<step>/ and atomically renames to
# <step>/ only once the save is fully done, so "the numeric dir exists" is a safe completeness
# check -- no risk of uploading a partially-written checkpoint.
#
# qqyang/zihiao_real_test is a PRE-EXISTING dataset repo (see the HF repo registry in
# Ego2Contact_project/rules/vision-workflow.md) -- --repo-type MUST be dataset, never model.
#
# Submit with: sbatch deploy/vision/n0vtla_watch_upload_task1_10k.sbatch
set -uo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

echo "=== HF auth check (must be qqyang, never the cached labmate login) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

HF_DEST_ROOT="n0vtla_wetlab_posttrain/post_train_task1_vision8gpu"
STEP=10000

declare -A CKPT_DIRS=(
  [ours]="checkpoints/vtla_tactile_posttrain/task1_posttrain_ours_vision8gpu/${STEP}"
  [baseline]="checkpoints/vtla_tactile_posttrain/task1_posttrain_baseline_vision8gpu/${STEP}"
)

pending=(ours baseline)
echo "=== watching for step ${STEP} checkpoints: ${pending[*]} ==="

while [[ ${#pending[@]} -gt 0 ]]; do
  next_pending=()
  for name in "${pending[@]}"; do
    dir="${CKPT_DIRS[$name]}"
    if [[ -d "$dir" ]]; then
      echo "--- $name step ${STEP} found at $dir, uploading -> qqyang/zihiao_real_test:${HF_DEST_ROOT}/${name}/${STEP} ---"
      if hf upload qqyang/zihiao_real_test "$dir" "${HF_DEST_ROOT}/${name}/${STEP}" \
          --repo-type dataset --token "$HF_TOKEN"; then
        echo "$name step ${STEP} uploaded OK"
      else
        echo "$name step ${STEP} upload FAILED, will retry next poll" >&2
        next_pending+=("$name")
      fi
    else
      next_pending+=("$name")
    fi
  done
  pending=("${next_pending[@]}")
  if [[ ${#pending[@]} -gt 0 ]]; then
    echo "$(date +%T) still waiting on: ${pending[*]}"
    sleep 120
  fi
done

echo
echo "=== WATCH_UPLOAD_10K_COMPLETE (both runs' step ${STEP} checkpoint uploaded) ==="
