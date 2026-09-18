#!/bin/bash
#SBATCH --job-name=n0vtla_upload_t1_vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_upload_t1_vision_%j.log
#
# Submitted with --dependency=afterok:<ours_jobid>:<baseline_jobid> so SLURM itself holds
# this job until BOTH task1_posttrain_ours_vision8gpu and task1_posttrain_baseline_vision8gpu
# finish successfully -- no polling of squeue/sacct needed. CPU-only (pure file I/O + HF
# upload). 1h budget (shorter walltime schedules faster / cheaper on the shared queue): 10
# checkpoint steps per run (save_interval=2000 over num_train_steps=20000 -> 2000..20000),
# 20 checkpoint dirs total across both runs -- upload of a single N0-VTLA checkpoint's
# model+optimizer+metadata has comfortably fit under an hour in past runs on this cluster.
#
# Uploads EVERY checkpoint step dir for both runs (nothing deleted here -- deletion of the
# <10k intermediate steps only happens in a separate, manually-triggered step after the
# upload is confirmed). qqyang/zihiao_real_test is a PRE-EXISTING dataset repo (see the HF
# repo registry in Ego2Contact_project/rules/vision-workflow.md) -- --repo-type MUST be
# dataset, never model.
#
# Submit with:
#   sbatch --dependency=afterok:<ours_jobid>:<baseline_jobid> \
#     deploy/vision/n0vtla_upload_task1_vision_all_ckpts.sbatch
set -euo pipefail

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

upload_all() {
  local ckpt_root="$1"
  local dest_sub="$2"
  echo
  echo "=== checkpoints present under $ckpt_root ==="
  ls -1 "$ckpt_root"
  for d in "$ckpt_root"/*/; do
    name=$(basename "$d")
    if [[ "$name" =~ ^[0-9]+$ ]]; then
      echo "--- uploading step $name -> qqyang/zihiao_real_test:${HF_DEST_ROOT}/${dest_sub}/${name} ---"
      hf upload qqyang/zihiao_real_test "$d" "${HF_DEST_ROOT}/${dest_sub}/${name}" \
        --repo-type dataset --token "$HF_TOKEN"
    fi
  done
}

upload_all "checkpoints/vtla_tactile_posttrain/task1_posttrain_ours_vision8gpu" "ours"
upload_all "checkpoints/vtla_tactile_posttrain/task1_posttrain_baseline_vision8gpu" "baseline"

echo
echo "=== UPLOAD_COMPLETE (all checkpoint steps for both runs, folder: ${HF_DEST_ROOT}) ==="
