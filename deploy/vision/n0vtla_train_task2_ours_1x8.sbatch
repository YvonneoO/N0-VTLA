#!/bin/bash
#SBATCH --job-name=n0vtla_t2_ours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t2_ours_%j.log
#
# Task2 mirror of n0vtla_train_task1_ours_1x8.sbatch -- same everything, dataset swapped.
#
# Submit with: sbatch --dependency=afterok:<smoke_ours_jobid> deploy/vision/n0vtla_train_task2_ours_1x8.sbatch
set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
N0VTLA_ENV=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla
export CONDA_PREFIX="$N0VTLA_ENV"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$N0VTLA_ENV/bin:$PATH"

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export NPROC_PER_NODE=8
export EXP_NAME=task2_posttrain_ours_vision8gpu
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task2_train"
export VTLA_ASSET_ID=task2_softhardsort_train
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000"

export HF_HOME="/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/models/huggingface"
python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== starting task2_posttrain_ours on VISION, 1 node x 8 H200 GPUs ==="
bash train.sh --batch-size=64 --save-interval=2000
echo "=== TRAINING SCRIPT EXITED ==="
