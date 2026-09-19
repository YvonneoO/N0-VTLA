#!/bin/bash
#SBATCH --job-name=n0vtla_pt_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_pt_scaling_%j.log
#
# Post-train ONE (task, Stage-1 data fraction) cell of the scaling study, 10k steps.
# Same recipe as task1_posttrain_ours_vision8gpu (n0vtla_train_task1_ours_1x8.sbatch: bs64 on
# 8xH200, lr schedule left at the config default so the 10k checkpoint is comparable to the
# 10k checkpoint of the 20k-step 100% run) -- only the pretrained checkpoint, dataset and
# num-train-steps change.
#   TASK  = task1 | sv2   (sv2 = smoke_test_v2, 59 train episodes)
#   LABEL = 20pct | 40pct | 60pct | 80pct
#   sbatch --export=ALL,TASK=task1,LABEL=20pct deploy/vision/n0vtla_posttrain_scaling_siyuan.sbatch
set -uo pipefail
: "${TASK:?TASK must be task1 or sv2}"
: "${LABEL:?LABEL must be e.g. 20pct}"

YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
source activate $YQQ/envs/n0vtla

DATA_ROOT=$YQQ/data/n0vtla
case "$TASK" in
  task1) export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task1_train"; export VTLA_ASSET_ID=task1_tuberack_train ;;
  sv2)   export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_smoketestv2_train"; export VTLA_ASSET_ID=wetlab_smoketestv2_train ;;
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base_plus_stage1_scaling_${LABEL}_14000"
export EXP_NAME="${TASK}_posttrain_s1_${LABEL}_10k"
export NPROC_PER_NODE=8
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"

python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== $EXP_NAME: ckpt=$VTLA_PRETRAINED_CHECKPOINT data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID ==="
bash train.sh --batch-size=64 --num-train-steps=10000 --save-interval=5000
echo "=== TRAINING SCRIPT EXITED ($EXP_NAME) ==="
