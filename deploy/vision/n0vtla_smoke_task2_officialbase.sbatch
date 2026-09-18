#!/bin/bash
#SBATCH --job-name=n0vtla_smoke_t2_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/logs/n0vtla_smoke_t2_base_%j.log
#
# Task2 mirror of n0vtla_smoke_task1_officialbase.sbatch -- same everything, dataset swapped.
#
# Submit with: sbatch --dependency=afterok:<normstats_jobid> deploy/vision/n0vtla_smoke_task2_officialbase.sbatch
set -uo pipefail
cd /scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA

module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
source /scratch/project/prj-02-uq-llms-for-reasoning/yqq/env.sh
source activate /scratch/project/prj-02-uq-llms-for-reasoning/yqq/envs/n0vtla

DATA_ROOT=/scratch/project/prj-02-uq-llms-for-reasoning/yqq/data/n0vtla
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export NPROC_PER_NODE=1
export VTLA_DATASET_PATH="$DATA_ROOT/canonical_wetlab_task2_train"
export VTLA_ASSET_ID=task2_softhardsort_train
export VTLA_PRETRAINED_CHECKPOINT="$DATA_ROOT/n0-vtla-base"

export HF_HOME="/scratch/project/prj-02-uq-llms-for-reasoning/yqq/N0-VTLA/models/huggingface"
echo "=== pre-warming DINOv2 config cache at $HF_HOME ==="
python -c "
from transformers import AutoConfig
AutoConfig.from_pretrained('facebook/dinov2-base')
print('DINOv2 config cached OK')
"

echo "=== preflight (CHECK_ONLY=1) ==="
CHECK_ONLY=1 EXP_NAME=task2_smoke_officialbase_vision_preflight bash train.sh
PREFLIGHT_STATUS=$?
if [ "$PREFLIGHT_STATUS" != "0" ]; then
  echo "!!! preflight failed"
  exit 1
fi
echo "preflight OK"

echo "=== real smoke: 10 steps, batch-size=8, 1 H200 ==="
EXP_NAME=task2_smoke_officialbase_vision bash train.sh --batch-size=8 --num-train-steps=10 --save-interval=10
SMOKE_STATUS=$?

if [ "$SMOKE_STATUS" = "0" ]; then
  echo "=== SMOKE TEST PASSED ==="
else
  echo "=== SMOKE TEST FAILED (exit $SMOKE_STATUS) ==="
fi
rm -rf checkpoints/vtla_tactile_posttrain/task2_smoke_officialbase_vision
exit $SMOKE_STATUS
