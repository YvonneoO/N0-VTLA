#!/bin/bash
#SBATCH --job-name=n0vtla-stage1-scaling-eval
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/stage1_scaling_eval_%j.log
#
# Scores ONE Stage-1 scaling-study checkpoint (latest step in its exp dir) on the
# shared, QC'd "held_out" episode split -- see compute_stage1_scaling_held_out_eval.py's
# own docstring for why this must be a SEPARATE held-out pass, not each run's own
# training loss. Single GPU: this is a forward-only inference pass (no DDP, no
# backward), so 8 GPUs would just be wasted allocation for a ~1hr job.
#
# Runs from yqq/N0-VTLA_scaling_extract, same as the training launcher, so it always
# sees the same code/manifest state those runs used.
#
# Required: EXP_NAME (e.g. stage1_online_20pct) and DATA_FRACTION (e.g. 0.2) via
# --export=ALL,...
#   BASELINE=1 (with DATA_FRACTION=0.0) scores n0-vtla-base itself instead of a mid-train delta.
#   sbatch --export=ALL,EXP_NAME=stage1_online_20pct,DATA_FRACTION=0.2 \
#       scripts/vision_stage1_scaling_held_out_eval_siyuan.sbatch
set -euo pipefail

source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
export PYTHONDONTWRITEBYTECODE=1

: "${DATA_FRACTION:?DATA_FRACTION is required, e.g. 0.2}"
if [[ "${BASELINE:-0}" == "1" ]]; then
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-base}"
  CKPT_ARGS=(--baseline)
else
  : "${EXP_NAME:?EXP_NAME is required, e.g. stage1_online_20pct}"
  CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-${EXP_NAME#stage1_online_}}"
  CKPT_ARGS=(--checkpoint-dir "checkpoints/vtla_stage1_predictor_pretrain/${EXP_NAME}")
fi
# v2 reports (with pool retrieval + recon baselines) go to their own dir so the v1 JSONs stay intact.
OUT_DIR="${OUT_DIR:-/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/stage1_scaling_held_out_v2}"

cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA_scaling_extract

/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla/bin/python \
  scripts/compute_stage1_scaling_held_out_eval.py \
  "${CKPT_ARGS[@]}" \
  --checkpoint-label "$CHECKPOINT_LABEL" \
  --data-fraction "$DATA_FRACTION" \
  --episode-list-json /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/itw_scaling_splits.json \
  --episode-list-key held_out \
  --raw-root /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/raw \
  --normalization /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/normalization_pad30_per_task_scale.json \
  --base-checkpoint /scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla/n0-vtla-base \
  --output "${OUT_DIR}/${CHECKPOINT_LABEL}.json" \
  --overwrite

echo VISION_STAGE1_SCALING_EVAL_DONE
