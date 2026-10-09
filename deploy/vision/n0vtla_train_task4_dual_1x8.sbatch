#!/bin/bash
#SBATCH --job-name=n0vtla_t4_dual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_t4_dual_%j.log
#
# Post-train the dual-arm Task4 (50 min-pour episodes, canonical 32-dim bimanual layout) on 1 node x 8 H200.
# Same recipe as task1_posttrain_{ours,baseline}_vision8gpu (bs64, vtla_tactile_posttrain config, 20k steps,
# save every 2k); only data/asset/ckpt change. Needs the build from n0vtla_task4_build_canonical.sbatch.
#   VARIANT=ours      -> n0-vtla-base_plus_stage1_human_14000 (full human-data mid-train merged into base)
#   VARIANT=baseline  -> n0-vtla-base (no mid-train)
#   sbatch --export=ALL,VARIANT=ours deploy/vision/n0vtla_train_task4_dual_1x8.sbatch
# Optional DATA_NAME / ASSET_ID / EXP_TAG select another build (default: the per-run-segment build). The
# one-episode-per-trajectory build (n0vtla_task4_build_canonical.sbatch WHOLE=1):
#   sbatch --export=ALL,VARIANT=ours,DATA_NAME=canonical_wetlab_task4_dual_traj_train,ASSET_ID=task4_dual_pour_traj_train,EXP_TAG=traj_ <this file>
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
export VTLA_DATASET_PATH="$DATA_ROOT/${DATA_NAME:-canonical_wetlab_task4_dual_train}"
export VTLA_ASSET_ID="${ASSET_ID:-task4_dual_pour_train}"
export EXP_NAME="task4_dual_posttrain_${VARIANT}_${EXP_TAG:-}vision8gpu"
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
bash train.sh --batch-size=64 --save-interval=2000
echo "=== TRAINING SCRIPT EXITED ($EXP_NAME) ==="
