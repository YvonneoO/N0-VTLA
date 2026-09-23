#!/bin/bash
#SBATCH --job-name=n0vtla_smoke_t3_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_smoke_t3_scaling_%j.log
#
# Single-GPU smoke test for the task3 scaling post-train launcher (n0vtla_posttrain_scaling_siyuan.sbatch),
# using the 20pct merged checkpoint. Verifies the PATH-prepend fix for the `source activate` bug that made
# jobs 569092-569095 fail in ~5s with ModuleNotFoundError (torch/transformers) -- schedules far faster than
# an 8-GPU job so it doesn't have to wait behind the real chain.
#   sbatch deploy/vision/n0vtla_smoke_task3_scaling_siyuan.sbatch
set -uo pipefail
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
export CONDA_PREFIX="$YQQ/envs/n0vtla"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$YQQ/envs/n0vtla/bin:$PATH"

DATA_ROOT=$YQQ/data/n0vtla
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export NPROC_PER_NODE=1
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task3_train"
export VTLA_ASSET_ID=task3_whiteboard_train
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base_plus_stage1_scaling_20pct_14000"
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"

echo "=== which python/torchrun (must be under envs/n0vtla) ==="
which python; which torchrun
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

echo "=== preflight (CHECK_ONLY=1) ==="
CHECK_ONLY=1 EXP_NAME=task3_smoke_scaling_preflight bash train.sh
[ "$?" != "0" ] && { echo "!!! preflight failed"; exit 1; }
echo "preflight OK"

echo "=== real smoke: 10 steps, batch-size=8, 1 H200, 20pct scaling checkpoint ==="
EXP_NAME=task3_smoke_scaling_20pct bash train.sh --batch-size=8 --num-train-steps=10 --save-interval=10
STATUS=$?
[ "$STATUS" = "0" ] && echo "=== SMOKE TEST PASSED ===" || echo "=== SMOKE TEST FAILED (exit $STATUS) ==="
rm -rf checkpoints/vtla_tactile_posttrain/task3_smoke_scaling_20pct
exit $STATUS
