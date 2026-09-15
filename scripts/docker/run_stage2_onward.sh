#!/usr/bin/env bash
set -euo pipefail

# Runs Stage-2 -> merge -> post-train, using exactly what setup_stage2_prereqs.sh
# downloaded. Extra args pass through to each train_*.sh call (e.g. CHECK_ONLY=1
# as an env var, or --num-train-steps=N after the script name).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME_STAGE2="${EXP_NAME_STAGE2:-stage2_align}"
EXP_NAME_POSTTRAIN="${EXP_NAME_POSTTRAIN:-tactile_posttrain}"

echo "[1/3] Stage-2 training (exp=$EXP_NAME_STAGE2)"
VTLA_PRETRAINED_CHECKPOINT="checkpoints/n0-vtla-base" \
VTLA_STAGE1_CHECKPOINT="checkpoints/n0-vtla_ts_pretrain" \
VTLA_DATASET_PATH="data/openneodata_smoke" \
VTLA_ASSET_ID="openneodata_smoke" \
EXP_NAME="$EXP_NAME_STAGE2" \
  bash train_stage2.sh "$@"

echo "[2/3] merge Stage-1 + Stage-2 onto base"
python scripts/merge_stage2_checkpoint_for_posttrain.py \
  --base-checkpoint checkpoints/n0-vtla-base \
  --stage1-checkpoint checkpoints/n0-vtla_ts_pretrain \
  --stage2-checkpoint "checkpoints/vtla_stage2_align_expert/$EXP_NAME_STAGE2" \
  --output checkpoints/merged_for_posttrain --overwrite

echo "[3/3] post-train (exp=$EXP_NAME_POSTTRAIN)"
VTLA_PRETRAINED_CHECKPOINT="checkpoints/merged_for_posttrain" \
VTLA_DATASET_PATH="data/n0vtla_wetlab_canonical_v2/train" \
VTLA_ASSET_ID="wetlab_v2_train" \
EXP_NAME="$EXP_NAME_POSTTRAIN" \
  bash train.sh "$@"

echo "done. Final checkpoint: checkpoints/vtla_tactile_posttrain/$EXP_NAME_POSTTRAIN/ on your host"
echo "(via the CHECKPOINTS_DIR you set in open_container.sh) -- send this directory back to us."
