#!/bin/bash
#SBATCH --job-name=n0vtla_t4_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t4_build_%j.log
#
# Task4 (dual-arm tube pour) raw release -> canonical LeRobot dataset + norm stats, all CPU:
#  [1/4] pad30 tactile normalization, per hand (fit_wetlab_dual_tactile_norm.py)
#  [2/4] canonical dataset, all 50 train episodes (build_wetlab_dual_canonical_dataset.py)
#  [3/4] state/action norm stats with BOTH arms made relative (compute_canonical_norm.py
#        --robot wetlab_dual_arm; the single-arm 'wetlab_right_arm' mode would leave the left arm absolute)
#  [4/4] read-back check through the real N0 input pipeline (verify_wetlab_dual_dataset.py)
# Submit: sbatch --dependency=afterok:<download job> deploy/vision/n0vtla_task4_build_canonical.sbatch
set -euo pipefail

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
source $YQQ/env.sh
PY=$YQQ/envs/n0vtla/bin/python
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
cd $YQQ/N0-VTLA
export PYTHONPATH=$PWD:$PWD/scripts

DATA_ROOT=$YQQ/data/n0vtla
RAW_DIR=$DATA_ROOT/wetlab_raw/task4_dual_arm_tube_pour_20261005
SPLIT=$RAW_DIR/training_split.json
NORM_PATH=$DATA_ROOT/wetlab_tactile_norm_task4_dual.json
OUT=$DATA_ROOT/canonical_wetlab_task4_dual_train
ASSET_ID=task4_dual_pour_train
SCRATCH=$DATA_ROOT/_scratch_task4
mkdir -p $SCRATCH

echo "=== [1/4] tactile normalization ==="
[[ -f $NORM_PATH ]] && echo "exists, skipping" || $PY scripts/fit_wetlab_dual_tactile_norm.py $RAW_DIR $SPLIT $NORM_PATH

echo "=== [2/4] canonical dataset ==="
$PY scripts/build_wetlab_dual_canonical_dataset.py $RAW_DIR $SPLIT $NORM_PATH $OUT --workers 16 --scratch-dir $SCRATCH

echo "=== [3/4] state/action norm stats ==="
$PY scripts/compute_canonical_norm.py --repo-id $OUT --robot wetlab_dual_arm \
  --train-config-name vtla_tactile_posttrain --asset-id $ASSET_ID --train-only

echo "=== [4/4] verify ==="
$PY scripts/verify_wetlab_dual_dataset.py --dataset $OUT --asset-id $ASSET_ID
echo TASK4_CANONICAL_BUILD_COMPLETE
