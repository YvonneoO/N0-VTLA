#!/bin/bash
#SBATCH --job-name=n0vtla-s1-ptdata-eval
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/stage1_ptdata_eval_%j.log
#
# One (checkpoint, split) of the Stage-1 scaling checkpoints scored on Task1 (tuberack)
# post-train data -- see scripts/compute_stage1_scaling_posttrain_data_eval.py.
# Required: CKPT_LABEL (e.g. 20pct | base), CKPT_DIR (complete model dir), SPLIT (train|val).
# Optional: STRIDE(=5) MAX_SAMPLES(=0 none) OUT_DIR TAG (appended to output names, e.g. _smoke).
#   sbatch --export=ALL,CKPT_LABEL=base,CKPT_DIR=$YQQ/data/n0vtla/n0-vtla-base,SPLIT=val \
#       deploy/vision/n0vtla_stage1_scaling_posttrain_eval_siyuan.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

: "${CKPT_LABEL:?}"; : "${CKPT_DIR:?}"; : "${SPLIT:?train|val}"
STRIDE="${STRIDE:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TAG="${TAG:-}"
DATA=$YQQ/data/n0vtla
OUT_DIR="${OUT_DIR:-$YQQ/data/stage1_scaling_ptdata_eval}"

cd $YQQ/N0-VTLA_scaling_extract
ASSET_DIR=assets/vtla_stage1_predictor_pretrain/task1_tuberack_train
mkdir -p "$ASSET_DIR"
cp -n $DATA/task1_tuberack_train_norm_stats.json "$ASSET_DIR/norm_stats.json"

export VTLA_ASSET_ID=task1_tuberack_train
export VTLA_DATASET_PATH=$DATA/canonical_wetlab_task1_${SPLIT}

$YQQ/envs/n0vtla/bin/python scripts/compute_stage1_scaling_posttrain_data_eval.py \
  --checkpoint "$CKPT_DIR" --checkpoint-label "$CKPT_LABEL" \
  --dataset-root "$VTLA_DATASET_PATH" --split-label "$SPLIT" \
  --stride "$STRIDE" --max-samples "$MAX_SAMPLES" \
  --output "$OUT_DIR/${CKPT_LABEL}_${SPLIT}${TAG}.json" \
  --embeddings-out "$OUT_DIR/${CKPT_LABEL}_${SPLIT}${TAG}.pt" --overwrite
echo VISION_S1_PTDATA_EVAL_DONE
