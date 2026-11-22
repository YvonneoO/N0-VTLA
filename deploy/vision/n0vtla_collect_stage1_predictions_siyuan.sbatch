#!/bin/bash
#SBATCH --job-name=n0vtla-s1-preds
#SBATCH --partition=def
#SBATCH --account=prj-02-uq-llms-for-reasoning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_s1_preds_%j.log
#
# Collect per-sample predictions (z, z*, current-tactile embedding, DINOv2 features, physical
# change statistics) of base + the 20/40/60/80% mid-train deltas + the 100% reference for ONE
# task's post-train data (val split, then train split), in a single pass per split. One output
# file per (task, split): $OUT_DIR/<task>_<split>.pt. See scripts/collect_stage1_multi_ckpt_predictions.py.
#   TASK = task1 | task2 | task3 | sv2
#   SMOKE=1 -> val only, 48 samples, output suffix _smoke
#   sbatch --export=ALL,TASK=task1 deploy/vision/n0vtla_collect_stage1_predictions_siyuan.sbatch
set -euo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
: "${TASK:?TASK must be task1|task2|task3|sv2}"

DATA=$YQQ/data/n0vtla
OUT_DIR="${OUT_DIR:-$YQQ/data/stage1_predictions}"
mkdir -p "$OUT_DIR"
case "$TASK" in
  task1) NAME=task1; ASSET=task1_tuberack_train;      VAL_DIR=canonical_wetlab_task1_val;        TRAIN_DIR=canonical_wetlab_task1_train ;;
  task2) NAME=task2; ASSET=task2_softhardsort_train;  VAL_DIR=canonical_wetlab_task2_val;        TRAIN_DIR=canonical_wetlab_task2_train ;;
  task3) NAME=task3; ASSET=task3_whiteboard_train;    VAL_DIR=canonical_wetlab_task3_val;        TRAIN_DIR=canonical_wetlab_task3_train ;;
  sv2)   NAME=sv2;   ASSET=wetlab_smoketestv2_train;  VAL_DIR=canonical_wetlab_smoketestv2_dev;  TRAIN_DIR=canonical_wetlab_smoketestv2_train ;;
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac

cd $YQQ/N0-VTLA_scaling_extract
ASSET_DIR=assets/vtla_stage1_predictor_pretrain/$ASSET
mkdir -p "$ASSET_DIR"
cp -n $YQQ/N0-VTLA/assets/vtla_tactile_posttrain/$ASSET/norm_stats.json "$ASSET_DIR/norm_stats.json"
export VTLA_ASSET_ID=$ASSET

CK=checkpoints/vtla_stage1_predictor_pretrain
CKPTS=(--ckpt 20pct=$CK/stage1_online_20pct/14000 --ckpt 40pct=$CK/stage1_online_40pct/14000
       --ckpt 60pct=$CK/stage1_online_60pct/14000 --ckpt 80pct=$CK/stage1_online_80pct/14000
       --ckpt 100pct=$CK/stage1_online_100pct_phailab/14000)

if [[ "${SMOKE:-0}" == "1" ]]; then SPLITS="val"; TAG=_smoke; else SPLITS="val train"; TAG=""; fi
for SPLIT in $SPLITS; do
  if [[ "$SPLIT" == "val" ]]; then DIR=$VAL_DIR; STRIDE=2; MAXS=0; else DIR=$TRAIN_DIR; STRIDE=5; MAXS=6000; fi
  if [[ "${SMOKE:-0}" == "1" ]]; then STRIDE=50; MAXS=48; fi
  export VTLA_DATASET_PATH=$DATA/$DIR
  echo "=== $NAME / $SPLIT (stride $STRIDE, cap $MAXS) ==="
  python scripts/collect_stage1_multi_ckpt_predictions.py --base $DATA/n0-vtla-base "${CKPTS[@]}" \
    --dataset-root "$VTLA_DATASET_PATH" --task-label $NAME --split-label $SPLIT \
    --stride $STRIDE --max-samples $MAXS --out "$OUT_DIR/${NAME}_${SPLIT}${TAG}.pt" --overwrite
done
echo VISION_S1_PREDS_DONE
