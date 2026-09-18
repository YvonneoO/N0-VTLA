#!/bin/bash
#SBATCH --job-name=n0vtla_watch_upload_new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_watch_upload_new_%j.log
#
# Generalized successor to n0vtla_watch_upload_task1_10k.sbatch: instead of one fixed step,
# sweeps BOTH runs' checkpoint dirs every 120s and uploads+deletes ANY numeric checkpoint dir
# found -- catches the already-sitting ours/12000 plus every future save (14000/16000/18000/...)
# for both task1_posttrain_ours_vision8gpu and task1_posttrain_baseline_vision8gpu.
#
# Stop condition: keeps sweeping until BOTH training jobs (560688=ours, 560650=baseline) are no
# longer RUNNING/PENDING per sacct, then does one final sweep (a checkpoint write can land just
# as/after the job exits) before exiting. NOT watching for a literal "20000" checkpoint dir --
# train_pytorch.py's save condition is `global_step % save_interval == 0 or
# global_step == num_train_steps - 1`, and since the loop only reaches index num_train_steps-1
# (19999) for num_train_steps=20000, the true final checkpoint is named 19999, never 20000.
#
# 1h45m budget: ours needed ~1.4h and baseline ~1.2h of training left as of writing, plus margin
# for the last checkpoint write + upload after training exits. Bounded downside if it still times
# out: every checkpoint already swept was uploaded-then-deleted immediately (nothing lost), so a
# timeout only means whatever's left unswept needs one more manual pass, not a redo.
#
# Submit with: sbatch deploy/vision/n0vtla_watch_upload_task1_new_ckpts.sbatch
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
OURS_JOBID=560688
BASELINE_JOBID=560650

declare -A CKPT_DIRS=(
  [ours]="checkpoints/vtla_tactile_posttrain/task1_posttrain_ours_vision8gpu"
  [baseline]="checkpoints/vtla_tactile_posttrain/task1_posttrain_baseline_vision8gpu"
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
echo "=== WATCH_UPLOAD_NEW_COMPLETE ==="
