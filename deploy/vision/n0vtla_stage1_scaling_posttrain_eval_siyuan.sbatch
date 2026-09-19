#!/bin/bash
#SBATCH --job-name=n0vtla-s1-ptdata-eval
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/stage1_ptdata_eval_%j.log
#
# ONE Stage-1 scaling checkpoint scored on Task1 (tuberack) post-train data, BOTH splits in
# one 1-GPU job (val first, then train) -- see scripts/compute_stage1_scaling_posttrain_data_eval.py.
# Each split is its own python process because VTLA_DATASET_PATH is read at config import.
# Required: CKPT_LABEL (e.g. 20pct | base), CKPT_DIR (complete model dir).
# Optional: SPLITS(="val train"), STRIDE_VAL(=2) STRIDE_TRAIN(=5), MAX_VAL(=0 none) MAX_TRAIN(=6000),
#           OUT_DIR, TAG (appended to output names, e.g. _smoke).
#   sbatch --export=ALL,CKPT_LABEL=base,CKPT_DIR=$YQQ/data/n0vtla/n0-vtla-base \
#       deploy/vision/n0vtla_stage1_scaling_posttrain_eval_siyuan.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

: "${CKPT_LABEL:?}"; : "${CKPT_DIR:?}"
SPLITS="${SPLITS:-val train}"
STRIDE_VAL="${STRIDE_VAL:-2}"; STRIDE_TRAIN="${STRIDE_TRAIN:-5}"
MAX_VAL="${MAX_VAL:-0}"; MAX_TRAIN="${MAX_TRAIN:-6000}"
TAG="${TAG:-}"
DATA=$YQQ/data/n0vtla
OUT_DIR="${OUT_DIR:-$YQQ/data/stage1_scaling_ptdata_eval}"

cd $YQQ/N0-VTLA_scaling_extract
ASSET_DIR=assets/vtla_stage1_predictor_pretrain/task1_tuberack_train
mkdir -p "$ASSET_DIR"
cp -n $DATA/task1_tuberack_train_norm_stats.json "$ASSET_DIR/norm_stats.json"
export VTLA_ASSET_ID=task1_tuberack_train

for SPLIT in $SPLITS; do
  if [[ "$SPLIT" == "val" ]]; then STRIDE=$STRIDE_VAL; MAXS=$MAX_VAL; else STRIDE=$STRIDE_TRAIN; MAXS=$MAX_TRAIN; fi
  export VTLA_DATASET_PATH=$DATA/canonical_wetlab_task1_${SPLIT}
  echo "=== $CKPT_LABEL / $SPLIT (stride $STRIDE, cap $MAXS) ==="
  $YQQ/envs/n0vtla/bin/python scripts/compute_stage1_scaling_posttrain_data_eval.py \
    --checkpoint "$CKPT_DIR" --checkpoint-label "$CKPT_LABEL" \
    --dataset-root "$VTLA_DATASET_PATH" --split-label "$SPLIT" \
    --stride "$STRIDE" --max-samples "$MAXS" \
    --output "$OUT_DIR/${CKPT_LABEL}_${SPLIT}${TAG}.json" \
    --embeddings-out "$OUT_DIR/${CKPT_LABEL}_${SPLIT}${TAG}.pt" --overwrite
done
echo VISION_S1_PTDATA_EVAL_DONE
