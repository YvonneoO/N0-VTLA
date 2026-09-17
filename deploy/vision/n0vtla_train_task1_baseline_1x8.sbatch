#!/bin/bash
#SBATCH --job-name=n0vtla_t1_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t1_base_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export NPROC_PER_NODE=8
export EXP_NAME=task1_posttrain_baseline_vision8gpu
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task1_train"
export VTLA_ASSET_ID=task1_tuberack_train
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base"

echo "=== starting task1_posttrain_baseline (official base) on VISION, 1 node x 8 H200 GPUs ==="
bash train.sh --batch-size=64 --save-interval=2000
echo "=== TRAINING SCRIPT EXITED ==="
