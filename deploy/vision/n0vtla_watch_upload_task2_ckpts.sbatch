#!/bin/bash
#SBATCH --job-name=n0vtla_watch_t2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_watch_t2_%j.log
#
# Task2 mirror of n0vtla_watch_upload_task1_new_ckpts.sbatch: sweeps both
# task2_posttrain_{ours,baseline}_vision8gpu checkpoint dirs every 120s and uploads+deletes
# any numeric checkpoint dir found (catches the already-sitting ours/{2000,4000,6000} and
# baseline/{2000,4000} plus every future save), into a NEW dedicated Task2 HF folder separate
# from Task1's. Stops once both training jobs (562173=ours, 562174=baseline) are no longer
# RUNNING/PENDING, then does one final sweep. 8h budget: ours needs ~3.8h and baseline ~6.2h
# of training left as of writing (baseline is slower, ~42 vs ~58 steps/min), plus margin.
# Bounded downside on timeout: every checkpoint already swept was uploaded-then-deleted
# immediately, so nothing is lost, just needs one more manual pass for stragglers.
#
# Submit with: sbatch deploy/vision/n0vtla_watch_upload_task2_ckpts.sbatch
set -uo pipefail

module load Miniforge3/25.3.0-3
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
# See n0vtla_task2_build_canonical.sbatch: source/conda activate weren't reliably switching
# the interpreter on some compute nodes in this non-interactive sbatch context, so bypass it
# and prepend the env's bin/ to PATH directly.
N0VTLA_ENV=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$N0VTLA_ENV/bin:$PATH"

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

echo "=== HF auth check (must be qqyang, never the cached labmate login) ==="
WHOAMI=$(hf auth whoami)
echo "hf auth whoami: $WHOAMI"
if [[ "$WHOAMI" != *"qqyang"* ]]; then
  echo "ERROR: expected qqyang, got '$WHOAMI' -- aborting (env.sh not sourced first?)" >&2
  exit 1
fi

HF_DEST_ROOT="n0vtla_wetlab_posttrain/post_train_task2_vision8gpu"
OURS_JOBID=562173
BASELINE_JOBID=562174

declare -A CKPT_DIRS=(
  [ours]="checkpoints/vtla_tactile_posttrain/task2_posttrain_ours_vision8gpu"
  [baseline]="checkpoints/vtla_tactile_posttrain/task2_posttrain_baseline_vision8gpu"
)

sweep_and_upload() {
  for name in ours baseline; do
    ckpt_root="${CKPT_DIRS[$name]}"
    [[ -d "$ckpt_root" ]] || continue
    for d in "$ckpt_root"/*/; do
      [[ -d "$d" ]] || continue
      step_name=$(basename "$d")
      [[ "$step_name" =~ ^[0-9]+$ ]] || continue
      echo "--- $name step ${step_name} found at $d, uploading -> qqyang/zihiao_real_test:${HF_DEST_ROOT}/${name}/${step_name} ---"
      if hf upload qqyang/zihiao_real_test "$d" "${HF_DEST_ROOT}/${name}/${step_name}" \
          --repo-type dataset --token "$HF_TOKEN"; then
        echo "$name step ${step_name} uploaded OK -- deleting local copy at $d"
        rm -rf "$d"
        echo "$name step ${step_name} deleted from VISION"
      else
        echo "$name step ${step_name} upload FAILED, will retry next poll" >&2
      fi
    done
  done
}

jobs_still_active() {
  local states
  states=$(sacct -j "${OURS_JOBID},${BASELINE_JOBID}" --format=JobID,State --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$1 !~ /\./ {print $2}')
  echo "$states" | grep -qE "RUNNING|PENDING"
}

echo "=== sweeping until training jobs ${OURS_JOBID} (ours) and ${BASELINE_JOBID} (baseline) finish ==="
while jobs_still_active; do
  sweep_and_upload
  echo "$(date +%T) training still active, sleeping 120s"
  sleep 120
done

echo "=== training jobs finished -- final sweep to catch the last checkpoint ==="
sleep 15
sweep_and_upload

echo
echo "=== WATCH_UPLOAD_TASK2_COMPLETE ==="
