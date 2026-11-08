#!/bin/bash
#SBATCH --job-name=n0vtla-merge-s1
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/merge_s1_%j.log
#
# Merge ONE Stage-1 (mid-train) trainable-only checkpoint onto n0-vtla-base -> a complete model.safetensors that
# post-train loads via VTLA_PRETRAINED_CHECKPOINT / PRETRAINED (scripts/merge_stage1_into_base_checkpoint.py says why).
# Env: STAGE1 (step dir, required), OUT (required), CLONE (checkout with the merge script, default yqq/N0-VTLA_egotouch).
# EgoTouch mid-train egt_20k, last step:
#   sbatch --export=ALL,STAGE1=$YQQ/N0-VTLA_egotouch/checkpoints/vtla_stage1_predictor_pretrain/egt_20k/20000,OUT=$YQQ/data/n0vtla/n0-vtla-base_plus_stage1_egotouch_20000 <this>
set -euo pipefail
: "${STAGE1:?}"; : "${OUT:?}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd "${CLONE:-$YQQ/N0-VTLA_egotouch}"
for f in model.safetensors metadata.pt; do [[ -s "$STAGE1/$f" ]] || { echo "MRG|missing $STAGE1/$f"; exit 1; }; done
$YQQ/envs/n0vtla/bin/python scripts/merge_stage1_into_base_checkpoint.py \
  --base-checkpoint $YQQ/data/n0vtla/n0-vtla-base --stage1-checkpoint "$STAGE1" --output "$OUT"
ls -la "$OUT"
echo "MRG|merged $STAGE1 -> $OUT"
echo MERGE_STAGE1_DONE
