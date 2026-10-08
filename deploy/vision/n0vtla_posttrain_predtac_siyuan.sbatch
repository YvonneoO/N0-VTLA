#!/bin/bash
#SBATCH --job-name=n0vtla_predtac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_predtac_%x_%j.log
#
# N0-VTLA post-train with PREDICTED tactile (video -> fine-tuned v2 DiT -> cell-to-taxel decoder -> glove-format tactile -> canonical dataset,
# see Ego2Contact scripts/tools/robot_ft_predtac_build.sbatch). Same recipe as task1_posttrain_ours_vision8gpu / sv2 (bs64 on 8 GPUs, config
# default lr schedule, init = Stage-1 human-data mid-train merged into the base) -- only the tactile VIDEOS of the dataset differ (state, action,
# RGB, episodes, norm_stats are identical to the real-tactile datasets, so the same asset id is used).
#   TASK   = task1 | sv2
#   SUFFIX = predtac (dataset suffix written by robot_ft_predtac_build.sbatch)
#   STEPS  = 20000 (smoke: 20)     SAVE = 5000      RESUME=1 to resume
#   sbatch --job-name=task1 --export=ALL,TASK=task1 deploy/vision/n0vtla_posttrain_predtac_siyuan.sbatch
set -uo pipefail
: "${TASK:?TASK must be task1 or sv2}"
SUFFIX="${SUFFIX:-predtac}"
YQQ=/scratch/project/prj-02-uq-llms-for-reasoning/yqq
cd $YQQ/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source $YQQ/env.sh
export CONDA_PREFIX="$YQQ/envs/n0vtla"
export CONDA_DEFAULT_ENV=n0vtla
export PATH="$YQQ/envs/n0vtla/bin:$PATH"

DATA_ROOT=$YQQ/data/n0vtla
case "$TASK" in
  task1) export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task1_${SUFFIX}_train"; export VTLA_ASSET_ID=task1_tuberack_train ;;
  sv2)   export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_smoketestv2_${SUFFIX}_train"; export VTLA_ASSET_ID=wetlab_smoketestv2_train ;;
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac
[ -d "$VTLA_DATASET_PATH" ] || { echo "dataset $VTLA_DATASET_PATH missing"; exit 2; }
export VTLA_PRETRAINED_CHECKPOINT="${PRETRAINED:-$DATA_ROOT/n0-vtla-base_plus_stage1_human_14000}"
export EXP_NAME="${EXP_NAME_OVERRIDE:-${TASK}_posttrain_ours_${SUFFIX}_vision8gpu}"
export NPROC_PER_NODE=8
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME="$YQQ/N0-VTLA/models/huggingface"
ulimit -n 65536 2>/dev/null || true

$YQQ/envs/n0vtla/bin/python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== $EXP_NAME: ckpt=$VTLA_PRETRAINED_CHECKPOINT data=$VTLA_DATASET_PATH asset=$VTLA_ASSET_ID steps=${STEPS:-20000} ==="
bash train.sh --batch-size=64 --num-train-steps=${STEPS:-20000} --save-interval=${SAVE:-5000} ${RESUME:+--resume}
echo "=== TRAINING SCRIPT EXITED ($EXP_NAME) ==="
