#!/bin/bash
#SBATCH --job-name=n0vtla_upload_t4_dual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_upload_t4_dual_%j.log
#
# Watcher: submit with --dependency=afterany:<ours job>:<baseline job>; uploads every saved checkpoint
# step dir (model.safetensors + optimizer.pt + metadata.pt + assets/ with the state/action norm stats)
# of both Task4 dual-arm post-train runs, plus the per-hand tactile normalization the client needs,
# to qqyang/zihiao_real_test (dataset; pre-existing repo, never --repo-type model), following the
# Task1 layout:  n0vtla_wetlab_posttrain/post_train_task4_dual_vision8gpu/{ours,baseline}/<step>/
# afterany (not afterok) so a run that stopped early still gets what it saved uploaded; the log says
# which steps exist. Nothing is deleted here. Re-running just re-uploads the same paths.
#   sbatch --dependency=afterany:687128:687129 deploy/vision/n0vtla_watch_upload_task4_dual_ckpts.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
HF=$YQQ/envs/n0vtla/bin/hf
cd $YQQ/N0-VTLA

WHOAMI=$($HF auth whoami)
echo "hf auth whoami: $WHOAMI"
[[ "$WHOAMI" == *"qqyang"* ]] || { echo "ERROR: expected qqyang" >&2; exit 1; }

REPO=qqyang/zihiao_real_test
ROOT=n0vtla_wetlab_posttrain/post_train_task4_dual_vision8gpu

echo "=== tactile normalization (must travel with the checkpoints) ==="
$HF upload $REPO $YQQ/data/n0vtla/wetlab_tactile_norm_task4_dual.json "$ROOT/wetlab_tactile_norm_task4_dual.json" \
  --repo-type dataset --token "$HF_TOKEN"

upload_all() {
  local ckpt_root="$1" sub="$2"
  echo; echo "=== $ckpt_root ==="; ls -1 "$ckpt_root"
  for d in "$ckpt_root"/*/; do
    name=$(basename "$d")
    if [[ "$name" =~ ^[0-9]+$ ]]; then
      echo "--- $sub step $name ---"
      $HF upload $REPO "$d" "$ROOT/$sub/$name" --repo-type dataset --token "$HF_TOKEN"
    fi
  done
}
upload_all checkpoints/vtla_tactile_posttrain/task4_dual_posttrain_ours_vision8gpu ours
upload_all checkpoints/vtla_tactile_posttrain/task4_dual_posttrain_baseline_vision8gpu baseline

echo "https://huggingface.co/datasets/$REPO/tree/main/$ROOT"
echo T4_UPLOAD_COMPLETE
