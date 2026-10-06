#!/bin/bash
#SBATCH --job-name=n0vtla_t4_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t4_smoke_%j.log
#
# 10-step plumbing smoke for the dual-arm Task4 post-train, 1 H200, per-GPU batch 8 (= the per-GPU
# batch of the real 8-GPU bs64 run, so its peak memory is representative). 3 RGB + 2 tactile views.
#   sbatch --export=ALL,VARIANT=ours deploy/vision/n0vtla_smoke_task4_dual.sbatch
set -uo pipefail
: "${VARIANT:?VARIANT must be ours or baseline}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
source activate $YQQ/envs/n0vtla
export PATH="$YQQ/envs/n0vtla/bin:$PATH"
DATA_ROOT=$YQQ/data/n0vtla
case "$VARIANT" in
  ours)     export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000" ;;
  baseline) export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base" ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task4_dual_train"
export VTLA_ASSET_ID=task4_dual_pour_train
export NPROC_PER_NODE=1
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"

echo "=== preflight ==="
CHECK_ONLY=1 EXP_NAME=task4_smoke_${VARIANT}_preflight bash train.sh || { echo "!!! preflight failed"; exit 1; }
echo "=== 10 steps, batch-size=8, 1 H200 ==="
EXP_NAME=task4_smoke_${VARIANT} bash train.sh --batch-size=8 --num-train-steps=10 --save-interval=10
S=$?
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
[ "$S" = "0" ] && echo "=== SMOKE PASSED ===" || echo "=== SMOKE FAILED (exit $S) ==="
rm -rf checkpoints/vtla_tactile_posttrain/task4_smoke_${VARIANT} checkpoints/vtla_tactile_posttrain/task4_smoke_${VARIANT}_preflight
exit $S
