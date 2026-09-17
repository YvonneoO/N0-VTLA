#!/bin/bash
#SBATCH --job-name=n0vtla_t1_ours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t1_ours_%j.log

set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla

# First run on this cluster: nothing is cached yet (DINOv2 config, PaliGemma tokenizer).
# Compute nodes have real internet access -- let train.sh's preflight fetch what it needs
# instead of its own HF_HUB_OFFLINE=1 default, which would fail with nothing cached.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export NPROC_PER_NODE=8
export EXP_NAME=task1_posttrain_ours_vision8gpu
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task1_train"
export VTLA_ASSET_ID=task1_tuberack_train
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000"

# See n0vtla_smoke_task1_officialbase.sbatch -- train.sh's preflight hardcodes
# local_files_only=True for the DINOv2 config check, so pre-warm it first.
export HF_HOME="/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/models/huggingface"
python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== starting task1_posttrain_ours on VISION, 1 node x 8 H200 GPUs ==="
bash train.sh --batch-size=64 --save-interval=2000
echo "=== TRAINING SCRIPT EXITED ==="
