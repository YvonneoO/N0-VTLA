#!/bin/bash
#SBATCH --job-name=n0vtla-merge-scaling
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/merge_scaling_%j.log
#
# Merges each Stage-1 scaling checkpoint (trainable-only delta, step 14000) onto n0-vtla-base
# into a COMPLETE model.safetensors that post-train can load via VTLA_PRETRAINED_CHECKPOINT
# (see scripts/merge_stage1_into_base_checkpoint.py for why the merge is required).
# Output: yqq/data/n0vtla/n0-vtla-base_plus_stage1_scaling_<label>_14000, next to the existing
# n0-vtla-base_plus_stage1_human_14000 (= the 100%-data one post-train "ours" already uses).
#   sbatch --export=ALL,LABELS="20pct 40pct 60pct" deploy/vision/n0vtla_merge_stage1_scaling_ckpts_siyuan.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
cd $YQQ/N0-VTLA_scaling_extract

LABELS="${LABELS:-20pct 40pct 60pct}"
STEP="${STEP:-14000}"

for label in $LABELS; do
  out="$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_scaling_${label}_${STEP}"
  echo "=== merging $label -> $out ==="
  $YQQ/envs/n0vtla/bin/python scripts/merge_stage1_into_base_checkpoint.py \
    --base-checkpoint $YQQ/data/n0vtla/n0-vtla-base \
    --stage1-checkpoint "checkpoints/vtla_stage1_predictor_pretrain/stage1_online_${label}/${STEP}" \
    --output "$out"
  ls -la "$out"
done
echo MERGE_SCALING_DONE
